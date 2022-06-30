import numpy as np
from scipy.stats import linregress

def step(fn, year, cds, param_list):
    inv_y = 1 / cds
    exp_cx = np.stack([fn(y, param_list) for y in year])
    n_time, n_param = exp_cx.shape
    intercept_list = []
    slope_list = []
    std_list = []
    for p in range(n_param):
        lin = linregress(exp_cx[:,p], inv_y)
        intercept, slope = lin.intercept, lin.slope
        du = (cds - (1 / intercept) / (1 + slope / intercept * exp_cx[:,p])) ** 2
        std = np.sqrt(np.sum(du) / (n_time - 1))
        intercept_list.append(intercept)
        slope_list.append(slope)
        std_list.append(std)
    return intercept_list, slope_list, std_list


def optimize(estimate_fn, year, cds):
    c_arr = np.array([2 ** (-p) for p in range(21)])
    a_inv, b_a, std = step(estimate_fn, year, cds, c_arr)
    c_opt = c_arr[np.argmin(std)]

    for k in range(1, 6):
        c_arr = np.array([c_opt * 2 ** (p / 4 ** k) for p in range(4, -5, -1)])
        a_inv, b_a, std = step(estimate_fn, year, cds, c_arr)
        a_opt = 1 / a_inv[np.argmin(std)]
        b_opt = a_opt * b_a[np.argmin(std)]
        c_opt = c_arr[np.argmin(std)]
        std_opt = np.min(std)
    return a_opt, b_opt, c_opt, std_opt


if __name__ == "__main__":
    year = [2010, 2011, 2012, 2013, 2013, 2014, 2014, 2015, 2015, 2017, 2017, 2018]
    cd_val = [0.362, 0.378, 0.36, 0.354000002, 0.36500001, 0.352, 0.356, 0.347, 0.345, 0.386, 0.349, 0.366]

    year = np.array(year)
    cds = np.array(cd_val)

    estimate_fn = lambda year, c: np.exp(-c * (year - 1900))

    # training
    a_opt, b_opt, c_opt, std_opt = optimize(estimate_fn, year, cds)
    print(a_opt, b_opt, c_opt, std_opt)

    # predict
    year_pred = np.arange(2010, 2020)
    trend = a_opt / (1 + b_opt * estimate_fn(year_pred, c_opt))
    
