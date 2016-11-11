"""A variant of Savitzky Golay filter by Chen et al 2004 R.S.E. to
find the upper envelop of a time series.

Zhan Li, zhanli86@bu.edu
Created: Wed Oct 19 16:35:43 EDT 2016
"""

import sys
import argparse

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def sgFilterUpEnv(ts, n_iters=100, window0=7, polyorder0=3, debug=False):
    ts_len = len(ts)
    ts_flag = np.logical_not(np.isnan(ts))
    ts_valid = ts[ts_flag]
    ts_valid_ind = np.where(ts_flag)[0]
    # linear interpolation
    x = np.arange(ts_len)[ts_flag]
    x = np.insert(x, 0, ts_valid_ind[-1]-ts_len)
    x = np.append(x, ts_len+ts_valid_ind[0])
    y = np.insert(ts[ts_flag], 0, ts_valid[-1])
    y = np.append(y, ts_valid[0])
    lin_interp_func = interp1d(x, y)
    x = np.arange(ts_len)
    ts0 = lin_interp_func(x)
    
    if debug:
        fig, ax = plt.subplots()
        ax.plot(x, ts, 'ok', label='ts original')
        ax.plot(x, ts0, '-k', label='t0')
    
    # get long term trend
    ts_tr = savgol_filter(ts0, window0, polyorder0, mode='wrap')
    # ts_tr = np.ones(ts_len)*np.median(ts_valid)
    if debug:
        ax.plot(x, ts_tr, '-r', label='ts trend')
    
    # calculate weight for each sample
    weight = np.ones(ts_len)
    ts_diff = np.fabs(ts0 - ts_tr)
    tmp_flag = ts0<ts_tr
    weight[tmp_flag] = 1 - ts_diff[tmp_flag] / np.max(ts_diff)

    # update TS 
    ts_k = np.copy(ts0)
    ts_k[tmp_flag] = ts_tr[tmp_flag]
    # refit to get the first TS of our iteration
    window = 7
    polyorder = 3
    ts_kp1 = savgol_filter(ts_k, window, polyorder, mode='wrap')
    if debug:
        ax.plot(x, ts_kp1, '--', label='ts 1')
    
    f_k = np.sum(np.fabs((ts_kp1-ts0)[ts_flag]) * weight[ts_flag])
    f_k_down = True
    # start iteration
    f_k_rec = np.zeros(n_iters+1)
    f_k_rec[0] = f_k
    for i in range(n_iters):
        f_km1 = f_k
        ts_final = np.copy(ts_kp1)
        ts_new = ts_kp1
        tmp_flag = ts0 >= ts_kp1
        ts_new[tmp_flag] = ts0[tmp_flag]
        ts_kp1 = savgol_filter(ts_new, window, polyorder, mode='wrap')
        
        f_k_rec[i+1] = np.sum(np.fabs((ts_kp1-ts0)[ts_flag]) * weight[ts_flag])
#         if f_k_down and (f_k_rec[i+1]>f_k_rec[i]):
#             break
    f_k_final = f_k_rec[i]
    
    if debug:
        ax.plot(x, ts_final, '-.k', label='ts final')
        plotly_fig = plotly.tools.mpl_to_plotly(fig)
        plotly_fig['layout']['showlegend'] = True
        plotly_fig['layout']['legend'] = dict(orientation="h")
        iplot(plotly_fig)
    return ts_final, f_k_rec

def main(cmdargs):
    pass

def getCmdArgs():
    pass

if __name__=='__main__':
    cmdargs = getCmdArgs()
    main(cmdargs)
