import numpy as np
# from numba import jit
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
from lorenz_solver import l96_truth_step, run_lorenz96_truth, gnr_synthetic_data


#
# class L96_forward_model:
#     def __init__(self, theta, initial_state, K, L):
#         self.theta = theta
#         self.initial_state = initial_state
#         self.K = K
#         self.L = L
#
# def natural_forward_model(self, time_step, num_steps, burn_in, skip, N):
#     K = self.K
#     L = self.L
#     x0, y0 = self.initial_state
#     h, F, b, c = self.theta
#     return forward_model_fi(x0, y0, h, F, b, c, time_step, num_steps, burn_in, skip, N)

#     # def surrogate_stochastic_model(self, ):


def natural_forward_model(K, L, x0, y0, h, F, b, c, time_step, num_steps, burn_in, skip, N):
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


# def surrogate_forward_model():


def parallel_nfm_j(K, L, x0, y0, time_step, num_steps, burn_in, skip, theta_sample, N, j):
    h_j = theta_sample[j, 0]
    F_j = theta_sample[j, 1]
    # c_j = np.exp(theta_prev[j, 2])
    # b_j = theta_prev[j, 3]
    b_j = theta_sample[j, 2]
    c_j = 10

    forward_eva_j = natural_forward_model(K, L, x0, y0, h_j, F_j, b_j, c_j, time_step, num_steps,
                                          burn_in, skip, N)

    # u_matrix =




    return forward_eva_j


global rng
rng = np.random.default_rng(2022)


class new_ces:
    def __init__(self, theta_prior_mean, theta_prior_sigma, initial_prior_mean, initial_prior_sigma, true_initial,
                 Sigma_y, data):
        self.theta_prior_mean = theta_prior_mean
        self.theta_prior_sigma = theta_prior_sigma
        self.initial_prior_mean = initial_prior_mean
        self.initial_prior_sigma = initial_prior_sigma
        self.true_initial = true_initial
        self.Sigma_m = Sigma_y
        self.data = data

    # def construct_forward_function(self):

    def prior_theta(self, J):
        # rng = np.random.default_rng(12345)
        theta_0 = rng.multivariate_normal(self.theta_prior_mean, self.theta_prior_sigma, J)
        return theta_0

    def prior_initial(self, J):
        # rng = np.random.default_rng(12345)
        z0 = rng.multivariate_normal(self.initial_prior_mean, self.initial_prior_sigma, J)
        return z0

    def calibrate_shortcut(self, K, L, true_m_theta, true_sigma_theta, sample_size, time_step, num_steps, burn_in, skip,
                           N, pool):
        approx_sample = rng.multivariate_normal(true_m_theta, true_sigma_theta, sample_size)
        x0, y0 = self.true_initial
        parallel_fm = partial(parallel_nfm_j, K, L, x0, y0, time_step, num_steps, burn_in, skip, approx_sample, N)
        forward_eva_list = pool.map(parallel_fm, [j for j in range(sample_size)])
        forward_eva = np.array(forward_eva_list).reshape(-1, 5)

        return approx_sample, forward_eva

    # def emulate_autoreg(self, ):

    # def emulate_autoregressive(self):

    # def sample_mcmc(self):


ne.set_vml_num_threads(8)


def main():
    ###synthetic data generation###
    K = 8
    L = 32
    X = np.zeros(K)
    Y = np.zeros(L * K)
    X[0] = 1
    Y[0] = 0.1
    h = 1.0
    b = 10.0
    c = 10.0
    F = 10

    T = 10
    time_step = 0.001  # delta_t
    num_steps = 360000  # integration_steps 3600/0.001->1000/0,01
    burn_in = 60000  # 600/0.001
    skip = 100  # 1/0.001

    num_steps_test = 36000  # integration_steps 3600/0.001
    burn_in_test = 6000  # 600/0.001
    skip_test = 1000  # 1/0.001

    N = int((num_steps - burn_in) / (skip * T))

    # X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, h_bad, F_bad, b_bad, c, time_step, num_steps, burn_in, skip)

    X_out, Y_out, times, steps = run_lorenz96_truth(X, Y, h, F, b, c, time_step, num_steps, burn_in, skip)
    synth_data_long, noisy_synth_data_long = gnr_synthetic_data(X_out, Y_out, times, 30, L, K)

    #####CES######
    N = 2  # the number of time intervals
    noisy_synth_data = noisy_synth_data_long[:N]
    synth_data = synth_data_long[:N]

    pool = mp.Pool(4)
    x0 = 1
    y0 = 0.1

    theta_prior_mean = np.array([0.0, 10.0, 12.0])
    theta_prior_sigma = np.diag([1.0, 5.0, 5.0])

    initial_prior_mean = np.zeros(2)
    initial_prior_sigma = np.identity(2)

    true_initial = [1, 0.1]

    SIGMA = np.asarray(noisy_synth_data_long.cov())

    ces_test = new_ces(theta_prior_mean, theta_prior_sigma, initial_prior_mean, initial_prior_sigma, true_initial,
                       SIGMA, data=noisy_synth_data)

    true_m_theta = np.array([1, 10, 10])
    true_sigma_theta = np.diag([0.5, 2, 2])
    sample_size = 20
    T = 10
    time_step = 0.001  # delta_t
    num_steps = 80000  # integration_steps 3600/0.001->1000/0,01
    burn_in = 60000  # 600/0.001
    skip = 100  # 1/0.001

    sample_test, forward_eva_test = ces_test.calibrate_shortcut(K, L, true_m_theta, true_sigma_theta, sample_size,
                                                                time_step, num_steps, burn_in, skip, N, pool)

    pool.close()
    pool.join()

    return synth_data_long, noisy_synth_data_long, sample_test, forward_eva_test


if __name__ == "__main__":
    synth_data, noisy_synth_data, sample_test, forward_eva_test = main()
    # print(synth_data.describe())
