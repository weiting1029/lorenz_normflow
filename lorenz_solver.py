import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import scipy
from scipy import stats
from scipy.integrate import solve_ivp
from scipy import linalg
from numpy.random import multivariate_normal


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


def gnr_synthetic_data(X_out, Y_out, times, N, J, K):
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
        start_index = i * J
        end_index = (i + 1) * J - 1
        Y_bar[:, i] = np.mean(Y_out[:, start_index:end_index], axis=1)
        Y_square_bar[:, i] = np.mean(Y_square[:, start_index:end_index], axis=1)
    big_array = np.concatenate((np.reshape(times, (T, 1)), X_out, Y_bar, Y_square_bar), axis=1)

    df = pd.DataFrame(big_array)
    df = df.groupby(np.arange(len(df)) // 100).mean()

    traject_array = np.array(df)
    synthetic_array = np.zeros([30, 5])
    synthetic_array[:, 0] = np.mean(traject_array[:, 1:9], axis=1)
    synthetic_array[:, 1] = np.mean(traject_array[:, 9:17], axis=1)
    synthetic_array[:, 2] = np.square(np.mean(traject_array[:, 1:9], axis=1))
    synthetic_array[:, 3] = np.multiply(np.mean(traject_array[:, 1:9], axis=1),
                                        np.mean(traject_array[:, 9:17], axis=1))
    synthetic_array[:, 4] = np.mean(traject_array[:, 17:25], axis=1)

    s_p = np.std(synthetic_array, axis=0)
    sigma_p = 1.5 * s_p
    mean = np.zeros(5)
    cov = np.diag(sigma_p ** 2)
    measurement_noise = multivariate_normal(mean, cov, 30)
    noisy_synth = synthetic_array + measurement_noise

    synthetic_data = pd.DataFrame(synthetic_array, columns=['X', 'Y_bar', 'X^2', 'X*Y_bar', 'Y_bar^2'])
    noisy_synthetic_data = pd.DataFrame(noisy_synth, columns=['X', 'Y_bar', 'X^2', 'X*Y_bar', 'Y_bar^2'])

    return synthetic_data, noisy_synthetic_data


def forward_model_fi(x0, y0, h, F, b, c, time_step, num_steps, burn_in, skip):
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
        end_index = (i + 1) * L - 1
        Y_bar[:, i] = np.mean(Y_out[:, start_index:end_index], axis=1)
        Y_square_bar[:, i] = np.mean(Y_square[:, start_index:end_index], axis=1)
    big_array = np.concatenate((np.reshape(times, (T, 1)), X_out, Y_bar, Y_square_bar), axis=1)

    df = pd.DataFrame(big_array)
    df = df.groupby(np.arange(len(df)) // 100).mean()

    traject_array = np.array(df)
    synthetic_array = np.zeros([30, 5])
    synthetic_array[:, 0] = np.mean(traject_array[:, 1:9], axis=1)
    synthetic_array[:, 1] = np.mean(traject_array[:, 9:17], axis=1)
    synthetic_array[:, 2] = np.square(np.mean(traject_array[:, 1:9], axis=1))
    synthetic_array[:, 3] = np.multiply(np.mean(traject_array[:, 1:9], axis=1),
                                        np.mean(traject_array[:, 9:17], axis=1))
    synthetic_array[:, 4] = np.mean(traject_array[:, 17:25], axis=1)

    return synthetic_array


def prior_theta(m_theta, sigma_theta, J):
    theta_0 = multivariate_normal(m_theta, sigma_theta, J)
    return theta_0


def prior_initial(m_z0, sigma_z0, J):
    z0 = multivariate_normal(m_z0, sigma_z0, J)
    return z0


def eks_fixed_initial(data, N, J, x0, y0, m_theta, sigma_theta, SIGMA, time_step, num_steps, burn_in, skip):
    p = m_theta.shape[0]
    theta_0 = prior_theta(m_theta, sigma_theta, J)
    weight_matrix = linalg.inv(linalg.sqrtm(SIGMA))  ##W = SIGMA^(-1/2)
    # theta_prev =
    theta_new = theta_0
    for i in range(N):
        theta_prev = theta_new #dim: (J, 4)
        theta_mean = np.mean(theta_prev, axis=0)
        theta_dm = theta_prev - theta_mean * np.ones(theta_prev.shape)
        CTHETA = np.zeros([p, p])
        inner = 0
        forward_eva = np.zeros([30 * J, 5])
        for j in range(J):
            CTHETA = CTHETA + np.outer(theta_dm[j, :], theta_dm[j, :]) #dim:(p,p)

            h_j = theta_prev[j, 0]
            F_j = theta_prev[j, 1]
            c_j = np.exp(theta_prev[j, 2])
            b_j = theta_prev[j, 3]
            start_index = j * 30
            end_index = (j + 1) * 30 - 1
            forward_eva[start_index:end_index, :] = forward_model_fi(x0, y0, h_j, F_j, b_j, c_j, time_step, num_steps,
                                                                     burn_in, skip)

            # distance_j = data - forward_eva[start_index:end_index, :]

        forward_mean = np.mean(forward_eva, axis=0) * np.ones(forward_eva.shape)  ##?
        CTHETA = CTHETA / J
        data_matrix = np.repeat(data, J, axis=0)

        g_demeaned = np.matmul(forward_eva - forward_mean, weight_matrix.T)  # dim: 30J x  5
        data_dm = np.matmul(forward_eva - data_matrix, weight_matrix.T)  # dim: 30J x 5
        delta_t = 1

        for j in range(J):
            start_index = j * 30
            end_index = (j + 1) * 30 - 1
            temp_matrix = np.repeat(data_dm[start_index:end_index, :], J, axis=0)
            dot_product = np.sum(np.multiply(g_demeaned, temp_matrix), axis=1) # (30J,1)
            vec_product = np.mean(dot_product.reshape(-1, 30), axis=1) #(J,1)

            theta_weighted = np.mean(np.multiply(theta_prev, np.repeat(vec_product, p, axis=1))) #dim: (p,1)

            v = theta_prev[j] - delta_t*theta_weighted
            A = delta_t * np.matmul(CTHETA, linalg.inv(sigma_theta)) + np.identity(p) #dim: (p,p)
            theta_new[j] = linalg.solve(A, v)

    return theta_new




            # norm = np.mean(np.multiply(g_demeaned, temp_matrix))

            # theta_new[j] = np.inner( , data_dm[start_index:end_index,:])


# def eks_unknown_initial(N, J, m_theta, sigma_theta, m_z0, sigma_z0):
#     theta_0 = prior_theta(m_theta, sigma_theta, J)
#     z_0 = prior_initial(m_z0, sigma_z0, J)
#     for i in range(N):
#


def process_lorenz_data(X_out, times, steps, J, F, dt, x_skip, t_skip, u_scale):
    """
    Sample from Lorenz model output and reformat the data into a format more amenable to machine learning.


    Args:
        X_out (ndarray): Lorenz 96 model output
        J (int): number of Y variables per X variable
        x_skip (int): number of X variables to skip when sampling the data
        t_skip (int): number of time steps to skip when sampling the data

    Returns:
        combined_data: pandas DataFrame
    """
    x_series_list = []
    # y_series_list = []
    # y_prev_list = []
    ux_series_list = []
    ux_prev_series_list = []
    u_series_list = []
    u_prev_series_list = []
    x_s = np.arange(0, X_out.shape[1], x_skip)
    t_s = np.arange(2, X_out.shape[0] - 1, t_skip)
    t_p = t_s - 1
    time_list = []
    step_list = []
    x_list = []
    K = X_out.shape[1]
    for k in x_s:
        x_series_list.append(X_out[t_s, k: k + 1])
        ux_series_list.append((-X_out[t_s, k - 1] * (X_out[t_s, k - 2] - X_out[t_s, (k + 1) % K]) - X_out[t_s, k] + F) -
                              (X_out[t_s + 1, k] - X_out[t_s, k]) / dt)
        ux_prev_series_list.append((-X_out[t_p, k - 1] * (X_out[t_p, k - 2] - X_out[t_p, (k + 1) % K]) - X_out[t_p, k]
                                    + F) - (X_out[t_s, k] - X_out[t_p, k]) / dt)
        # y_series_list.append(Y_out[t_s, k * J: (k + 1) * J])
        # y_prev_list.append(Y_out[t_p, k * J: (k + 1) * J])
        # u_series_list.append(np.expand_dims(u_scale * Y_out[t_s, k * J: (k+1) * J].sum(axis=1), 1))
        # u_prev_series_list.append(np.expand_dims(u_scale * Y_out[t_p, k * J: (k+1) * J].sum(axis=1), 1))
        time_list.append(times[t_s])
        step_list.append(steps[t_s])
        x_list.append(np.ones(time_list[-1].size) * k)
    x_cols = ["X_t"]
    # y_cols = ["Y_t+1_{0:d}".format(y) for y in range(J)]
    # y_p_cols = ["Y_t_{0:d}".format(y) for y in range(J)]
    # u_cols = ["Uy_t", "Uy_t+1", "Ux_t", "Ux_t+1"]
    u_cols = ["Ux_t", "Ux_t+1"]
    combined_data = pd.DataFrame(np.vstack(x_series_list), columns=x_cols)
    combined_data.loc[:, "time"] = np.concatenate(time_list)
    combined_data.loc[:, "step"] = np.concatenate(step_list)
    combined_data.loc[:, "x_index"] = np.concatenate(x_list)
    combined_data.loc[:, "u_scale"] = u_scale
    combined_data.loc[:, "Ux_t+1"] = np.concatenate(ux_series_list)
    combined_data.loc[:, "Ux_t"] = np.concatenate(ux_prev_series_list)
    # combined_data.loc[:, "Uy_t+1"] = np.concatenate(u_series_list)
    # combined_data.loc[:, "Uy_t"] = np.concatenate(u_prev_series_list)
    # combined_data = pd.concat([combined_data, pd.DataFrame(np.vstack(y_prev_list), columns=y_p_cols),
    #                           pd.DataFrame(np.vstack(y_series_list), columns=y_cols)], axis=1)
    out_cols = ["x_index", "step", "time", "u_scale"] + x_cols + u_cols  # + y_p_cols + y_cols
    return combined_data.loc[:, out_cols]


def save_lorenz_output(X_out, Y_out, times, steps, model_attrs, out_file):
    """
    Write Lorenz 96 truth model output to a netCDF file.

    Args:
        X_out (ndarray): X values from the model run
        Y_out (ndarray): Y values from the model run
        times (ndarray): time steps of model in units of MTU
        steps (ndarray): integer integration step values
        model_attrs (dict): dictionary of model attributes
        out_file: Name of the netCDF file

    Returns:

    """
    data_vars = dict()
    data_vars["time"] = xr.DataArray(times, dims=["time"], name="time", attrs={"long_name": "integration time",
                                                                               "units": "MTU"})
    data_vars["step"] = xr.DataArray(steps, dims=["time"], name="step", attrs={"long_name": "integration step",
                                                                               "units": ""})
    data_vars["lorenz_x"] = xr.DataArray(X_out, coords={"time": data_vars["time"], "x": np.arange(X_out.shape[1])},
                                         dims=["time", "x"], name="lorenz_X", attrs={"long_name": "lorenz_x",
                                                                                     "units": ""})
    data_vars["lorenz_y"] = xr.DataArray(Y_out, coords={"time": times, "y": np.arange(Y_out.shape[1])},
                                         dims=["time", "y"], name="lorenz_Y", attrs={"long_name": "lorenz_y",
                                                                                     "units": ""})
    l_ds = xr.Dataset(data_vars=data_vars, attrs=model_attrs)
    l_ds.to_netcdf(out_file, "w", encoding={"lorenz_x": {"zlib": True, "complevel": 2},
                                            "lorenz_y": {"zlib": True, "complevel": 2}})
    return


def main():
    K = 8
    J = 32
    X = np.zeros(K)
    Y = np.zeros(J * K)
    X[0] = 1
    Y[0] = 0.1
    h = 1
    b = 10.0
    c = 10.0
    F = 10.0

    T = 100
    time_step = 0.001  # delta_t
    num_steps = 3600000  # integration_steps 3600/0.001
    burn_in = 600000  # 600/0.001
    skip = 1000  # 1/0.001

    num_steps_test = 36000  # integration_steps 3600/0.001
    burn_in_test = 6000  # 600/0.001
    skip_test = 1000  # 1/0.001

    N = (num_steps - burn_in) / (skip * T)

    X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps, burn_in, skip)
    # X_out = np.array(pd.read_csv('data/X_out.csv'))
    # Y_out = np.array(pd.read_csv('data/Y_out.csv'))
    # times = np.arange(3000)
    # data_out = process_lorenz_data(X_out, times, steps, J, F, dt=time_step, x_skip=1, t_skip=10, u_scale=1)

    synth_data, noisy_synth_data = gnr_synthetic_data(X_out, Y_out, times, N, J, K)

    return synth_data, noisy_synth_data


if __name__ == "__main__":
    synth_data, noisy_synth_data = main()
