# -*- coding: utf-8 -*-
"""
Created on Wed May  7 23:11:29 2025

@author: archit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import my_functions as mf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
np.random.seed(22)
import sys

""" FUNCTIONS """

def loading_bar(progress, total, length=30):
    percent = int(100 * progress / total)
    filled = int(length * progress / total)
    bar = '█' * filled + '-' * (length - filled)
    sys.stdout.write(f'\rRunning Simulations: |{bar}| {percent}% --- {progress}/{total}')
    sys.stdout.flush()

def random_walk(T):
    e = np.random.normal(0,0.5,T)
    z = np.cumsum(e)
    return z

def adf_fixed_lag0(series):
    return adfuller(series,maxlag = 2, regression="c", autolag=None)[0]

def sadf(series, min_window):
    T = len(series)
    stats = []
    for end in range(min_window, T + 1):
        sub_series = series[:end]
        stat = adf_fixed_lag0(sub_series)
        stats.append(stat)
    return max(stats)

def sadf_critical_values_from_simulation(T, B=1000):
    r0 = 0.01 + 1.8 / np.sqrt(T)
    min_window = int(np.floor(r0 * T))
    stats = []
    for _ in range(B):
        series = random_walk(T)
        stat = sadf(series, min_window)
        stats.append(stat)
    stats = np.sort(stats)
    return {
        "90%": np.percentile(stats, 90),
        "95%": np.percentile(stats, 95),
        "99%": np.percentile(stats, 99),
        "all": stats
    }
    
def generate_bubble_series_from_z(z, r, beta=1.0, delta=0.05, eps=None):

    T = len(z)
    tau1 = int(np.floor(r * T))
    
    y = np.zeros(T)
    
    for t in range(1, T):
        if t <= tau1:
            y[t] = beta * z[t] + eps[t]
        else:
            y[t] = (1 + delta) * y[t-1] + eps[t]
    
    return y

def sadf_series(series, min_window):
    T = len(series)
    stats = []
    for end in range(min_window, T + 1):
        sub_series = series[:end]
        stat = adf_fixed_lag0(sub_series)
        stats.append(stat)
    return stats

 

def unique_rounded_uniform(low, high, size, *, decimals=1):

    step = 10 ** (-decimals)
    grid_lo = np.ceil(low / step) * step
    grid_hi = np.floor(high / step) * step
    grid_points = np.around(np.arange(grid_lo, grid_hi + step/2, step), decimals)
    admissible = grid_points[np.abs(grid_points) >= 0.2]

    if size > admissible.size:
        raise ValueError("Not enough admissible unique values.")

    return np.random.choice(admissible, size=size, replace=False)



def simulate_sadf_distribution(B, T, num_series, coint_frac, num_of_rw, multi, common_fac, e):
    """Simulates SADF null distribution and returns critical values and plot."""

    # Generate multipliers
    multipliers = multi

    # Parameters
    min_win = int((0.01 + 1.8 / np.sqrt(T)) * T)
    split_point = int(T * coint_frac)

    # Create cointegrated series
    e = e
    common_fac = common_fac
    
    mat = np.zeros((T, num_series))
    mat = (mat + common_fac) * multipliers + e

    # Pre-break OLS
    Y = mat[:split_point, 0]
    X = mat[:split_point, 1:]
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const)
    results = model.fit()
    params = results.params
    a = params[0]  # Intercept
    b = params[1:] # All slopes
    pre_resid = Y - (a + X @ b)

    # Simulation loop
    no_of_random_ts = int(num_of_rw * mat.shape[1])
    sadf_stats = []

    for i in range(B):
        innov = np.random.normal(0, 1, size=(T - split_point, no_of_random_ts))
        last_vals = mat[split_point - 1, :no_of_random_ts]
        random_walk_matrix = np.cumsum(innov, axis=0) + last_vals
        mat[split_point:, :no_of_random_ts] = random_walk_matrix
        Y = mat[split_point:, 0]
        X = mat[split_point:, 1:]
        post_resid = Y - (a + X @ b)
        sadf_stats.append(sadf(np.concatenate([pre_resid, post_resid]), min_win))
        loading_bar(i + 1, B)

    # Compute critical values
    sadf_stats = np.array(sadf_stats)
    res = {
        "90%": np.percentile(sadf_stats, 90),
        "95%": np.percentile(sadf_stats, 95),
        "99%": np.percentile(sadf_stats, 99),
        "all": sadf_stats
    }

    # Plot distribution
    plt.figure(figsize=(6, 4))
    plt.hist(res['all'], bins=40, density=True, alpha=0.7, color='steelblue')
    plt.axvline(res['90%'], color='orange', linestyle='--', label='90%')
    plt.axvline(res['95%'], color='red', linestyle='--', label='95%')
    plt.axvline(res['99%'], color='purple', linestyle='--', label='99%')
    plt.title(f"SADF Null Distribution (T={T})")
    plt.xlabel("SADF Statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    del mat
    # Return critical values DataFrame
    return pd.DataFrame({ "Critical Value": [res["90%"], res["95%"], res["99%"]] },
                        index=["90%", "95%", "99%"])





def run_sadf_bubble_check(
    delta, T, coint_frac, num_of_rw,
    common_fac, multipliers, innov, e,
    crit_val_95, crit_val_99, num_series
):
    # Parameters
    min_win = int((0.01 + 1.8 / np.sqrt(T)) * T)
    split_point = int(T * coint_frac)

    # Create cointegrated series
    mat = np.zeros((T, num_series))
    mat = (mat + common_fac) * multipliers + e

    # Pre-break OLS
    Y = mat[:split_point, 0]
    X = mat[:split_point, 1:]
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const)
    results = model.fit()
    params = results.params
    a = params[0]  # Intercept
    b = params[1:] # All slopes
    pre_resid = Y - (a + X @ b)

    print("beta = ",b)
    print("alpha = ", np.round(a,4))
    print(f"ADF on residuals  stat = {adfuller(pre_resid, autolag='AIC')[0]:.3f}")

    # Bubble Phase
    no_of_random_ts = int(num_of_rw * mat.shape[1])

    bubble_matrix = np.zeros((T - split_point, no_of_random_ts))
    bubble_matrix[0, :] = mat[split_point - 1, :no_of_random_ts]

    for t in range(1, T - split_point):
        bubble_matrix[t, :] = (1 + delta) * bubble_matrix[t - 1, :] + innov[t, :]

    mat[split_point:, :no_of_random_ts] = bubble_matrix

    Y = mat[split_point:, 0]
    X = mat[split_point:, 1:]
    post_resid = Y - (a + X @ b)
    new_res = np.concatenate([pre_resid, post_resid])
    sadf_stats = sadf_series(new_res, min_win)

    # Plotting
    plt.figure(figsize=(15, 15))

    # Subplot 1: Pre-break time series
    plt.subplot(5, 1, 1)
    plt.plot(mat[:split_point, :], color="magenta")
    plt.plot(common_fac[:split_point], color='black', alpha=1, label = "Common Factor")
    plt.legend()
    plt.title(f'First {coint_frac*100:.0f}% of the series')

    # Subplot 2: Pre-break residuals
    plt.subplot(5, 1, 2)
    plt.plot(pre_resid, color='limegreen')
    plt.axhline(0, color='black', lw=0.7)
    plt.title('Residuals (pre-break)')

    # Subplot 3: Post-break time series
    plt.subplot(5, 1, 3)
    plt.plot(mat[:, :no_of_random_ts], color="blue", alpha=0.7)
    plt.plot(mat[:, no_of_random_ts:], color="red")
    plt.title('Time Series after Explosion')

    # Subplot 4: Full residuals (pre + post)
    plt.subplot(5, 1, 4)
    plt.plot(new_res, color='red')
    plt.title('Residuals (after explosion)')

    # Subplot 5: SADF Path
    plt.subplot(5, 1, 5)
    plt.plot(sadf_stats, label='SADF path', color='black')
    plt.axhline(crit_val_95, color='blue', linestyle='--', label='95% Critical Value')
    plt.axhline(crit_val_99, color='red', linestyle='--', label='99% Critical Value')
    for t, stat in enumerate(sadf_stats):
        if stat > crit_val_95:
            plt.axvline(t, color='orange', alpha = 0.6)
            break
            
    
    plt.title("SADF Recursive Path with Bubble Detection")
    plt.xlabel("Time")
    plt.ylabel("SADF Statistic")
    plt.legend()

    plt.tight_layout()
    plt.show()

    if max(sadf_stats) > crit_val_99:
        print(f"Time series is Explosive. SADF Value is: {np.round(max(sadf_stats), 2)}")
    elif max(sadf_stats) > crit_val_95:
        print(f"Time series is Explosive. SADF Value is: {np.round(max(sadf_stats), 2)}")
    else:
        print(f"Time series is Non Explosive. SADF Value is: {np.round(max(sadf_stats), 2)}")
    print("")   
    
    if max(sadf_stats) > crit_val_95:
        print(f"Bubble starts at t={t}")
    return np.round(max(sadf_stats),4)


import cupy as cp



def simulate_sadf_distribution_gpu(B, T, num_series, coint_frac, num_of_rw, multi, common_fac, e):
    """Simulates SADF null distribution and returns critical values and plot."""

    # Generate multipliers
    multipliers = multi

    # Parameters
    min_win     = int((0.01 + 1.8 / np.sqrt(T)) * T)
    split_point = int(T * coint_frac)

    # Create cointegrated series (on CPU, as before)
    e           = e
    common_fac  = common_fac
    mat         = (common_fac * multipliers) + e

    # Pre-break OLS (unchanged)
    Y           = mat[:split_point, 0]
    X           = mat[:split_point, 1:]
    X_const     = sm.add_constant(X)
    results     = sm.OLS(Y, X_const).fit()
    a, b        = results.params[0], results.params[1:]
    pre_resid   = Y - (a + X @ b)

    # ————— GPU-accelerated simulation loop —————
    no_of_random_ts = int(num_of_rw * mat.shape[1])
    sadf_stats      = []

    # 1) Push the whole mat onto the GPU
    mat_gpu = cp.asarray(mat)

    for i in range(B):
        # 2) generate innovations on the GPU
        shape      = (T - split_point, no_of_random_ts)
        innov_gpu  = cp.random.standard_normal(shape)

        # 3) grab “last” values from GPU mat
        last_gpu   = mat_gpu[split_point - 1, :no_of_random_ts]

        # 4) build the new random-walk block entirely on GPU
        rw_gpu     = cp.cumsum(innov_gpu, axis=0) + last_gpu

        # 5) write it back into mat_gpu
        mat_gpu[split_point:, :no_of_random_ts] = rw_gpu

        # 6) move just the slices you need back to CPU for residual + SADF
        Y_post = cp.asnumpy(mat_gpu[split_point:, 0])
        X_post = cp.asnumpy(mat_gpu[split_point:, 1:])

        post_resid = Y_post - (a + X_post @ b)
        full_resid = np.concatenate([pre_resid, post_resid])

        sadf_stats.append(sadf(full_resid, min_win))
        loading_bar(i + 1, B)

    # (optional) free GPU memory early
    del mat_gpu

    # ————— back to unchanged CPU code —————
    sadf_stats = np.array(sadf_stats)
    res = {
        "90%": np.percentile(sadf_stats, 90),
        "95%": np.percentile(sadf_stats, 95),
        "99%": np.percentile(sadf_stats, 99),
        "all": sadf_stats
    }

    # Plot distribution
    plt.figure(figsize=(6, 4))
    plt.hist(res['all'], bins=40, density=True, alpha=0.7, color='steelblue')
    plt.axvline(res['90%'], color='orange', linestyle='--', label='90%')
    plt.axvline(res['95%'], color='red', linestyle='--', label='95%')
    plt.axvline(res['99%'], color='purple', linestyle='--', label='99%')
    plt.title(f"SADF Null Distribution (T={T})")
    plt.xlabel("SADF Statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(
        {"Critical Value": [res["90%"], res["95%"], res["99%"]]},
        index=["90%", "95%", "99%"]
    )

def run_sadf_bubble_check_2(
    delta, T, coint_frac, num_of_rw,
    common_fac, multipliers, innov, e,
    crit_val_95, crit_val_99, num_series
):
    # Parameters
    min_win = int((0.01 + 1.8 / np.sqrt(T)) * T)
    split_point = int(T * coint_frac)

    # Create cointegrated series
    mat = np.zeros((T, num_series))
    mat = (mat + common_fac) * multipliers + e

    # Pre-break OLS
    Y = mat[:split_point, 0]
    X = mat[:split_point, 1:]
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const)
    results = model.fit()
    params = results.params
    a = params[0]  # Intercept
    b = params[1:] # All slopes
    pre_resid = Y - (a + X @ b)

    # print("beta = ",b)
    # print("alpha = ", np.round(a,4))
    # print(f"ADF on residuals  stat = {adfuller(pre_resid, autolag='AIC')[0]:.3f}")

    # Bubble Phase
    no_of_random_ts = int(num_of_rw * mat.shape[1])

    bubble_matrix = np.zeros((T - split_point, no_of_random_ts))
    bubble_matrix[0, :] = mat[split_point - 1, :no_of_random_ts]

    for t in range(1, T - split_point):
        bubble_matrix[t, :] = (1 + delta) * bubble_matrix[t - 1, :] + innov[t, :]

    mat[split_point:, :no_of_random_ts] = bubble_matrix

    Y = mat[split_point:, 0]
    X = mat[split_point:, 1:]
    post_resid = Y - (a + X @ b)
    new_res = np.concatenate([pre_resid, post_resid])
    sadf_stats = sadf_series(new_res, min_win)

    # # Plotting
    # plt.figure(figsize=(15, 15))

    # # Subplot 1: Pre-break time series
    # plt.subplot(5, 1, 1)
    # plt.plot(mat[:split_point, :], color="magenta")
    # plt.plot(common_fac[:split_point], color='black', alpha=1, label = "Common Factor")
    # plt.legend()
    # plt.title(f'First {coint_frac*100:.0f}% of the series')

    # # Subplot 2: Pre-break residuals
    # plt.subplot(5, 1, 2)
    # plt.plot(pre_resid, color='limegreen')
    # plt.axhline(0, color='black', lw=0.7)
    # plt.title('Residuals (pre-break)')

    # # Subplot 3: Post-break time series
    # plt.subplot(5, 1, 3)
    # plt.plot(mat[:, :no_of_random_ts], color="blue", alpha=0.7)
    # plt.plot(mat[:, no_of_random_ts:], color="red")
    # plt.title('Time Series after Explosion')

    # # Subplot 4: Full residuals (pre + post)
    # plt.subplot(5, 1, 4)
    # plt.plot(new_res, color='red')
    # plt.title('Residuals (after explosion)')

    # # Subplot 5: SADF Path
    # plt.subplot(5, 1, 5)
    # plt.plot(sadf_stats, label='SADF path', color='black')
    # plt.axhline(crit_val_95, color='blue', linestyle='--', label='95% Critical Value')
    # plt.axhline(crit_val_99, color='red', linestyle='--', label='99% Critical Value')
    # for t, stat in enumerate(sadf_stats):
    #     if stat > crit_val_95:
    #         plt.axvline(t, color='orange', alpha = 0.6)
    #         break
            
    
    # plt.title("SADF Recursive Path with Bubble Detection")
    # plt.xlabel("Time")
    # plt.ylabel("SADF Statistic")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    # if max(sadf_stats) > crit_val_99:
    #     print(f"Time series is Explosive. SADF Value is: {np.round(max(sadf_stats), 2)}")
    # elif max(sadf_stats) > crit_val_95:
    #     print(f"Time series is Explosive. SADF Value is: {np.round(max(sadf_stats), 2)}")
    # else:
    #     print(f"Time series is Non Explosive. SADF Value is: {np.round(max(sadf_stats), 2)}")
    # print("")   
    
    # if max(sadf_stats) > crit_val_95:
    #     print(f"Bubble starts at t={t}")
    return np.round(max(sadf_stats),4)

