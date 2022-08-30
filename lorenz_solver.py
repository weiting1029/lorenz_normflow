import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import scipy
from scipy import stats
from scipy.integrate import solve_ivp
from scipy import linalg
from scipy.stats import multivariate_normal
import multiprocessing as mp
import numexpr as ne
from functools import partial


@jit(nopython=True, cache=True)
def l96_truth_step(X, Y, h, F, b, c):
    """
    Calculate the time increment in the X and Y variables for the Lorenz '96 "truth" model.

    Args:
        X (1D ndarray): Values of X variables at the current time step
        Y (1D ndarray): Values of Y variables at the current time step
        h (float): Coupling constant
        F (float): Forcing term
        b (float): Spatial scale ratio
        c (float): Time scale ratio

    Returns:
        dXdt (1D ndarray): Array of X increments, dYdt (1D ndarray): Array of Y increments
    """
    K = X.size
    J = Y.size // K
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(K):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F - h * c / b * np.sum(Y[k * J: (k + 1) * J])
    for j in range(J * K):
        dYdt[j] = -c * b * Y[(j + 1) % (J * K)] * (Y[(j + 2) % (J * K)] - Y[j - 1]) - c * Y[j] + h * c / b * X[
            int(j / J)]
    return dXdt, dYdt


@jit(nopython=True, cache=True)
def run_lorenz96_truth(x_initial, y_initial, h, f, b, c, time_step, num_steps, burn_in, skip):
    """
    Integrate the Lorenz '96 "truth" model forward by num_steps.

    Args:
        x_initial (1D ndarray): Initial X values.
        y_initial (1D ndarray): Initial Y values.
        h (float): Coupling constant.
        f (float): Forcing term.
        b (float): Spatial scale ratio
        c (float): Time scale ratio
        time_step (float): Size of the integration time step in MTU
        num_steps (int): Number of time steps integrated forward.
        burn_in (int): Number of time steps not saved at beginning
        skip (int): Number of time steps skipped between archival

    Returns:
        X_out [number of timesteps, X size]: X values at each time step,
        Y_out [number of timesteps, Y size]: Y values at each time step
    """
    archive_steps = (num_steps - burn_in) // skip  ##number of steps to be saved
    x_out = np.zeros((archive_steps, x_initial.size))
    y_out = np.zeros((archive_steps, y_initial.size))
    steps = np.arange(num_steps)[burn_in::skip]
    times = steps * time_step
    x = np.zeros(x_initial.shape)
    y = np.zeros(y_initial.shape)
    # Calculate total Y forcing over archive period using trapezoidal rule
    y_trap = np.zeros(y_initial.shape)
    x[:] = x_initial
    y[:] = y_initial
    y_trap[:] = y_initial
    k1_dxdt = np.zeros(x.shape)
    k2_dxdt = np.zeros(x.shape)
    k3_dxdt = np.zeros(x.shape)
    k4_dxdt = np.zeros(x.shape)
    k1_dydt = np.zeros(y.shape)
    k2_dydt = np.zeros(y.shape)
    k3_dydt = np.zeros(y.shape)
    k4_dydt = np.zeros(y.shape)
    i = 0

    if burn_in == 0:
        x_out[i] = x
        y_out[i] = y
        i += 1
    for n in range(1, num_steps):
        # if (n * time_step) % 1 == 0:
        #     print(n, n * time_step)
        k1_dxdt[:], k1_dydt[:] = l96_truth_step(x, y, h, f, b, c)
        k2_dxdt[:], k2_dydt[:] = l96_truth_step(x + k1_dxdt * time_step / 2,
                                                y + k1_dydt * time_step / 2,
                                                h, f, b, c)
        k3_dxdt[:], k3_dydt[:] = l96_truth_step(x + k2_dxdt * time_step / 2,
                                                y + k2_dydt * time_step / 2,
                                                h, f, b, c)
        k4_dxdt[:], k4_dydt[:] = l96_truth_step(x + k3_dxdt * time_step,
                                                y + k3_dydt * time_step,
                                                h, f, b, c)
        x += (k1_dxdt + 2 * k2_dxdt + 2 * k3_dxdt + k4_dxdt) / 6 * time_step
        y += (k1_dydt + 2 * k2_dydt + 2 * k3_dydt + k4_dydt) / 6 * time_step
        if n >= burn_in and n % skip == 0:
            x_out[i] = x
            y_out[i] = (y + y_trap) / skip
            i += 1
        elif n % skip == 1:
            y_trap[:] = y
        else:
            y_trap[:] += y
    return x_out, y_out, times, steps


def gnr_synthetic_data(X_out, Y_out, times, N, L, K):
    """
    :param X_out: trajectories of large-scale variables X_k
    :param Y_out: trajectories of small-scale variables Y_l,k
    :param times: in MTU times
    :param N: number of realizations
    :param J: number of fast variables
    :param K: number of slow variables
    :return: synthetic noisy data for inversions
    """
    T = X_out.shape[0]
    Y_bar = np.zeros(X_out.shape)
    Y_square = np.square(Y_out)
    Y_square_bar = np.zeros(X_out.shape)
    for i in range(K):
        start_index = i * L
        end_index = (i + 1) * L
        Y_bar[:, i] = np.mean(Y_out[:, start_index:end_index], axis=1)
        Y_square_bar[:, i] = np.mean(Y_square[:, start_index:end_index], axis=1)
    big_array = np.concatenate((np.reshape(times, (T, 1)), X_out, Y_bar, Y_square_bar), axis=1)

    df = pd.DataFrame(big_array)
    df = df.groupby(np.arange(len(df)) // 100).mean()

    traject_array = np.array(df)
    synthetic_array = np.zeros([N, 5])
    synthetic_array[:, 0] = np.mean(traject_array[:, 1:9], axis=1)
    synthetic_array[:, 1] = np.mean(traject_array[:, 9:17], axis=1)
    synthetic_array[:, 2] = np.square(np.mean(traject_array[:, 1:9], axis=1))
    synthetic_array[:, 3] = np.multiply(np.mean(traject_array[:, 1:9], axis=1),
                                        np.mean(traject_array[:, 9:17], axis=1))
    synthetic_array[:, 4] = np.mean(traject_array[:, 17:25], axis=1)

    s_p = np.std(synthetic_array, axis=0)
    sigma_p = 5 * s_p
    # sigma_p = 0.01*np.ones(5)
    mean = np.zeros(5)
    cov = np.diag(sigma_p ** 2)
    measurement_noise = rng.multivariate_normal(mean, cov, N)
    noisy_synth = synthetic_array + measurement_noise

    synthetic_data = pd.DataFrame(synthetic_array, columns=['X', 'Y_bar', 'X^2', 'X*Y_bar', 'Y_bar^2'])
    noisy_synthetic_data = pd.DataFrame(noisy_synth, columns=['X', 'Y_bar', 'X^2', 'X*Y_bar', 'Y_bar^2'])

    return synthetic_data, noisy_synthetic_data


def forward_model_fi(x0, y0, h, F, b, c, time_step, num_steps, burn_in, skip, N):
    K = 8
    L = 32
    X = np.zeros(K)
    Y = np.zeros(L * K)
    X[0] = x0
    Y[0] = y0

    X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps, burn_in, skip)

    T = X_out.shape[0]
    Y_bar = np.zeros(X_out.shape)
    Y_square = np.square(Y_out)
    Y_square_bar = np.zeros(X_out.shape)
    for i in range(K):
        start_index = i * L
        end_index = (i + 1) * L
        Y_bar[:, i] = np.mean(Y_out[:, start_index:end_index], axis=1)
        Y_square_bar[:, i] = np.mean(Y_square[:, start_index:end_index], axis=1)
    big_array = np.concatenate((np.reshape(times, (T, 1)), X_out, Y_bar, Y_square_bar), axis=1)

    df = pd.DataFrame(big_array)
    df = df.groupby(np.arange(len(df)) // 100).mean()

    traject_array = np.array(df)
    synthetic_array = np.zeros([N, 5])
    synthetic_array[:, 0] = np.mean(traject_array[:, 1:9], axis=1)
    synthetic_array[:, 1] = np.mean(traject_array[:, 9:17], axis=1)
    synthetic_array[:, 2] = np.square(np.mean(traject_array[:, 1:9], axis=1))
    synthetic_array[:, 3] = np.multiply(np.mean(traject_array[:, 1:9], axis=1),
                                        np.mean(traject_array[:, 9:17], axis=1))
    synthetic_array[:, 4] = np.mean(traject_array[:, 17:25], axis=1)

    return synthetic_array


def parallel_fm_j(x0, y0, time_step, num_steps, burn_in, skip, theta_prev, theta_dm, N, j):
    CTHETA_j = np.outer(theta_dm[j, :], theta_dm[j, :])  # dim:(p,p)
    h_j = theta_prev[j, 0]
    F_j = theta_prev[j, 1]
    # c_j = np.exp(theta_prev[j, 2])
    # b_j = theta_prev[j, 3]
    b_j = theta_prev[j, 2]
    c_j = 10

    forward_eva_j = forward_model_fi(x0, y0, h_j, F_j, b_j, c_j, time_step, num_steps,
                                     burn_in, skip, N)

    return CTHETA_j, forward_eva_j


def prior_theta(m_theta, sigma_theta, L):
    theta_0 = rng.multivariate_normal(m_theta, sigma_theta, L)
    return theta_0


def prior_initial(m_z0, sigma_z0, L):
    z0 = rng.multivariate_normal(m_z0, sigma_z0, L)
    return z0


def eks_fixed_initial(data, max_itr, J, x0, y0, m_theta, sigma_theta, SIGMA, time_step, num_steps, burn_in, skip, N,
                      pool):
    eps = np.finfo(float).eps
    p = m_theta.shape[0]
    theta_0 = prior_theta(m_theta, sigma_theta, J)
    print(theta_0)
    weight_matrix = linalg.inv(linalg.sqrtm(SIGMA))  ##W = SIGMA^(-1/2)
    data_matrix = np.repeat(data.to_numpy()[None, :], J, axis=0).reshape(-1, 5)
    # theta_prev =
    theta_new = theta_0
    THETA = np.zeros([max_itr * J, p])
    for i in range(max_itr):
        theta_prev = theta_new  # dim: (J, 4)
        THETA[i * J:(i + 1) * J, :] = theta_new
        theta_mean = np.mean(theta_prev, axis=0)
        theta_dm = theta_prev - theta_mean * np.ones(theta_prev.shape)
        CTHETA = np.zeros([p, p])
        forward_eva = np.zeros([N * J, 5])
        # for j in range(J):
        #     CTHETA = CTHETA + np.outer(theta_dm[j, :], theta_dm[j, :])  # dim:(p,p)
        #     h_j = theta_prev[j, 0]
        #     F_j = theta_prev[j, 1]
        #     c_j = np.exp(theta_prev[j, 2])
        #     b_j = theta_prev[j, 3]
        #     start_index = j * 30
        #     end_index = (j + 1) * 30
        #     forward_eva[start_index:end_index, :] = forward_model_fi(x0, y0, h_j, F_j, b_j, c_j, time_step, num_steps,
        #                                                              burn_in, skip)

        # distance_j = data - forward_eva[start_index:end_index, :]

        parallel_fm = partial(parallel_fm_j, x0, y0, time_step, num_steps, burn_in, skip, theta_prev, theta_dm, N)
        CTHETA_list, forward_eva_list = zip(*pool.map(parallel_fm, [j for j in range(J)]))
        arr_CTHETA = np.array(CTHETA_list)
        arr_forward_eva = np.array(forward_eva_list)

        CTHETA = np.mean(arr_CTHETA, axis=0)
        forward_eva = arr_forward_eva.reshape(-1, 5)
        forward_mean = np.mean(forward_eva, axis=0) * np.ones(forward_eva.shape)  ##?
        g_demeaned = np.matmul(forward_eva - forward_mean, weight_matrix.T)  # dim: NJ x  5
        data_dm = np.matmul(forward_eva - data_matrix, weight_matrix.T)  # dim: NJ x 5
        # norm = np.mean(np.multiply(forward_eva - forward_mean, forward_eva - data_matrix))
        # delta_t = 1/(norm+eps)
        #
        norm = np.mean(np.multiply(g_demeaned, data_dm))
        delta_t = 1 / (norm + eps)

        for j in range(J):
            start_index = j * N
            end_index = (j + 1) * N
            temp_matrix = np.repeat(data_dm[start_index:end_index, :][None, :], J, axis=0).reshape(-1, 5)
            dot_product = np.sum(np.multiply(g_demeaned, temp_matrix), axis=1)  # (NJ,1)
            vec_product = np.mean(dot_product.reshape(-1, N), axis=1).reshape(J, )  # (J,1) sum over i

            theta_weighted = np.mean(np.multiply(theta_prev, np.repeat(vec_product, p).reshape(J, p)),
                                     axis=0)  # dim: (p,1) sum over j
            # theta_weighted = (np.matmul(theta_prev.T, vec_product).reshape([p, ]))/J

            v = theta_prev[j] - delta_t * theta_weighted
            A = delta_t * np.matmul(CTHETA, linalg.inv(sigma_theta)) + np.identity(p)  # dim: (p,p)
            random_W = rng.multivariate_normal(np.zeros(p), np.identity(p))
            theta_new[j] = linalg.solve(A, v) + np.sqrt(2 * delta_t) * np.matmul(linalg.sqrtm(CTHETA), random_W)
            print(theta_new[j])

    return THETA

    # norm = np.mean(np.multiply(g_demeaned, temp_matrix))

    # theta_new[j] = np.inner( , data_dm[start_index:end_index,:])


# def eks_unknown_initial(N, J, m_theta, sigma_theta, m_z0, sigma_z0):
#     theta_0 = prior_theta(m_theta, sigma_theta, J)
#     z_0 = prior_initial(m_z0, sigma_z0, J)
#     for i in range(N):
#

ne.set_vml_num_threads(8)


def main():
    K = 8
    L = 32
    X = np.zeros(K)
    Y = np.zeros(L * K)
    X[0] = 1
    Y[0] = 0.1
    h = 1
    b = 10.0
    c = 10.0
    F = 10.0

    global rng
    rng = np.random.default_rng(12345)

    T = 10
    time_step = 0.001  # delta_t
    num_steps = 360000  # integration_steps 3600/0.001->1000/0,01
    burn_in = 60000  # 600/0.001
    skip = 100  # 1/0.001

    num_steps_test = 36000  # integration_steps 3600/0.001
    burn_in_test = 6000  # 600/0.001
    skip_test = 1000  # 1/0.001

    N = int((num_steps - burn_in) / (skip * T))

    X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps, burn_in, skip)
    # X_out = np.array(pd.read_csv('data/X_out.csv'))[:, 1:]
    # Y_out = np.array(pd.read_csv('data/Y_out.csv'))[:, 1:]
    # times = np.arange(3000)
    # data_out = process_lorenz_data(X_out, times, steps, J, F, dt=time_step, x_skip=1, t_skip=10, u_scale=1)

    synth_data_long, noisy_synth_data_long = gnr_synthetic_data(X_out, Y_out, times, 30, L, K)

    N = 4
    noisy_synth_data = noisy_synth_data_long[:4]
    synth_data = synth_data_long[:4]
    # synth_data_long, noisy_synth_data_long = gnr_synthetic_data(X_out, Y_out, times, N, L, K)

    ######## EKS #######
    pool = mp.Pool(4)
    x0 = 1
    y0 = 0.1
    # m_theta = np.array([0, 10, 2, 8]) #(h,F,logc, b)
    # sigma_theta = np.diag([1, 3, 0.1, 3])
    m_theta = np.array([0.0, 10.0, 9.0]) #(h,F,logc, b)
    sigma_theta = np.diag([1.0, 1.0, 2.0])

    SIGMA = np.asarray(noisy_synth_data_long.cov())
    max_itr = 50
    J = 100

    T = 10
    time_step = 0.001  # delta_t
    num_steps = 100000  # integration_steps 3600/0.001->1000/0,01
    burn_in = 60000  # 600/0.001
    skip = 100  # 1/0.001

    theta_test = eks_fixed_initial(noisy_synth_data, max_itr, J, x0, y0, m_theta, sigma_theta, SIGMA, time_step,
                                   num_steps, burn_in, skip, N, pool)
    pool.close()
    pool.join()

    # theta_test = 0

    return synth_data, noisy_synth_data, theta_test


if __name__ == "__main__":
    synth_data, noisy_synth_data, theta_test = main()
    print(synth_data.describe())
    print(np.mean(theta_test, axis=0))
    h_eks_sample = theta_test[:, 0]
    F_eks_sample = theta_test[:, 1]
    # c_eks_sample = np.exp(theta_test[:, 2])
    b_eks_sample = theta_test[:, 2]

