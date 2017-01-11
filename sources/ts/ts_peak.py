import os
import sys
import itertools
import glob
import warnings
import argparse

import types

import numpy as np
from osgeo import gdal, gdal_array, osr, ogr
import scipy as sp
from scipy import optimize as spopt
import uncertainties as unc
from uncertainties import umath

import geo_ts as gt

def quadFunc(x, a, p1, p2):
    # y = a*(x**2 + p1*x + p2)
    return np.poly1d([a, a*p1, a*p2])(x)

def quadFuncUnc(x, a, p1, p2):
    return a * x**2 + a*p1 * x + a*p2

def quadPeakConstrained(x, y, max_iter=100):    
    peak_x = 250

    max_y = np.max(y)
    a_0 = -1
    p1_0 = -2*peak_x
    p2_0 = max_y/a_0 - (-0.5*p1_0)**2 - (-0.5*p1_0)*p1_0
    p0 = np.array([a_0, p1_0, p2_0])
    plbd = [-np.inf, -2*280, -np.inf]
    pubd = [0, -2*220, np.inf]
    for i in range(max_iter):       
        max_y = np.max(y)
        try:
            popt, pcov = spopt.curve_fit(quadFunc, x, y, 
                                               p0 = p0, 
                                               bounds=(plbd, pubd))

            # Use the uncertainties package to get the uncertainty of each parameter 
            # from the covariance matrix of the parameter estimates
            unc_p = unc.correlated_values(popt, pcov)        
            quad_coef = popt
            quad_coef_std = np.array([pv.std_dev for pv in unc_p])

            quad_func = lambda x: quadFunc(x, *popt)
            quad_func_unc = lambda x: quadFuncUnc(x, *unc_p)

            peak_x = -0.5 * popt[1]
            peak_x_std = (-0.5 * unc_p[1]).std_dev

            unc_x = unc.ufloat(peak_x, peak_x_std)
            unc_peak_y = quad_func_unc(unc_x)
            peak_y = unc_peak_y.nominal_value
            peak_y_std = unc_peak_y.std_dev

            unc_yfit = [quad_func_unc(xv) for xv in x]
            yfit_nom, yfit_std = zip(*[(uyv.nominal_value, uyv.std_dev) for uyv in unc_yfit])
            yfit_nom = np.array(yfit_nom)
            yfit_std = np.array(yfit_std)

            # see if the difference between fitted y and measured y is larger than the 3*std_dev
            tmp_flag = abs(yfit_nom-y) < 3*yfit_std
    #         tmp_flag = abs(yfit_nom-y) < np.inf
    #         tmp_flag = (yfit_nom-y) > -3*yfit_std
            if np.sum(np.logical_not(tmp_flag)) == 0:
                break
            x = x[tmp_flag]
            y = y[tmp_flag]
        except (RuntimeError, np.linalg.LinAlgError) as e:
            # if str(e).find("Optimal parameters not found: ") == -1:
            #     raise e
            # else:
            #     peak_x = 0
            #     peak_x_std = 0
            #     peak_y = 0
            #     peak_y_std = 0
            #     quad_coef = np.zeros_like(p0)
            #     quad_coef_std = np.zeros_like(p0)
            #     quad_func = None
            #     quad_func_unc = None
            #     return peak_x, peak_x_std, peak_y, peak_y_std, \
            #            quad_coef, quad_coef_std, quad_func, quad_func_unc, max_y
            peak_x = 0
            peak_x_std = 0
            peak_y = 0
            peak_y_std = 0
            quad_coef = np.zeros_like(p0)
            quad_coef_std = np.zeros_like(p0)
            quad_func = None
            quad_func_unc = None
            return peak_x, peak_x_std, peak_y, peak_y_std, \
                   quad_coef, quad_coef_std, quad_func, quad_func_unc, max_y
    
    return peak_x, peak_x_std, peak_y, peak_y_std, \
           quad_coef, quad_coef_std, quad_func, quad_func_unc, max_y

def runQuadPeakConstrained(ts_input, ix, iy, ibands,  
                           no_data=-9999, min_cnt=6, max_iter=100,
                           ts_doy=None):
    """Call function quadPeakConstrained with input raster file of time
    series *ts_file* at the image location (*ix*, *iy*) with *ix* as
    the sample/column and *iy* as the line/row, with first sample and
    line being 0. *ibands* is the indices to bands in ts_input to be
    used, with first band being 1.

    """
    # loading data is the most time-consuming step. So we will load
    # the whole time series data before entering this function that
    # would be called repeatedly.
    if isinstance(ts_input, types.StringTypes):
        ts_data = gt.readPixelsGdal(ts_input, [ix], [iy], band=ibands)
    else:
        ts_data = ts_input[iy, ix, ibands-1]

    if ts_doy is None:
        x = ibands.astype(float)
    else:
        x = ts_doy[ibands-1].astype(float)
    y = np.squeeze(ts_data)
    cond_list = [y!=no_data, y>0.1, y<=1.0]
    tmp_idx = np.where(reduce(np.logical_and, cond_list))[0]
    x = x[tmp_idx]
    y = y[tmp_idx]

    if len(x) >= min_cnt:
        qPC_out = quadPeakConstrained(x, y, max_iter=max_iter)
        # ix, iy, peak_x, peak_x_std, peak_y, peak_y_std, max_y
        fit_output = (ix, iy, qPC_out[0], qPC_out[1], qPC_out[2], qPC_out[3], qPC_out[8])
    else:
        fit_output = (ix, iy, no_data, no_data, no_data, no_data, no_data)

    return fit_output

def fitTsPeak(ts_file, out_peak_file, no_data=-9999, beg_doy=None, end_doy=None):
    # get the meta profile of the input raster file of time series
    ts_meta = gt.getRasterMetaGdal(ts_file)
    # Settings for raster reading/writing and TS fitting
    if beg_doy is None:
        beg_doy = 1
    if end_doy is None:
        end_doy = ts_meta['RasterCount']

    # band_names = ts_meta['Metadata']['ENVI']['band_names']

    ts_doy = np.arange(beg_doy, end_doy, 1, dtype=np.int)
    bands = np.arange(0, len(ts_doy), 1, dtype=np.int) + 1
    # load time series beteween beg_doy and end_doy of ALL pixels. 
    sys.stdout.write('Reading time series data at once ... ')
    sys.stdout.flush()
    ts_data_all = gt.readPixelsGdal(ts_file, band=ts_doy).reshape((ts_meta['RasterYSize'], ts_meta['RasterXSize'], -1))
    sys.stdout.write("done\n")

    results = [runQuadPeakConstrained(ts_data_all, ix, iy, bands, ts_doy=ts_doy) for ix in range(ts_meta['RasterXSize']) for iy in range(ts_meta['RasterYSize'])]
#    results = [runQuadPeakConstrained(ts_data_all, ix, iy, bands, ts_doy=ts_doy) for ix in range(5) for iy in range(2)]

    ix, iy, peak_x, peak_x_std, peak_y, peak_y_std, max_y = zip(*results)
    out_data_dict = dict(peak_doy=peak_x, peak_doy_std=peak_x_std,
                         peak_ts=peak_y, peak_ts_std=peak_y_std, max_ts=max_y)
    out_arr_dict = dict()
    for od_key, od in out_data_dict.items():
        out_arr = np.zeros((ts_meta['RasterYSize'], ts_meta['RasterXSize']), dtype=np.float32) + no_data
        out_arr[iy, ix] = od
        out_arr_dict[od_key] = out_arr

    # Save array of peak values to a raster file
    out_nband = len(out_arr_dict)
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(out_peak_file, ts_meta['RasterXSize'], ts_meta['RasterYSize'], out_nband, \
                           gdal_array.NumericTypeCodeToGDALTypeCode(out_arr.dtype.type))
    for i, oa_key in enumerate(["peak_doy", "peak_doy_std", "peak_ts", "peak_ts_std", "max_ts"]):
        oa = out_arr_dict[oa_key]
        out_band = out_ds.GetRasterBand(i+1)
        out_band.WriteArray(oa)
        out_band.SetDescription('{0:s} doy {1:d} to {2:d}'.format(oa_key.lower(), beg_doy, end_doy))
        out_band.SetNoDataValue(no_data)
        out_band.FlushCache()
    out_ds.SetGeoTransform(ts_meta['GeoTransform'])
    out_ds.SetProjection(ts_meta['ProjectionRef'])
    out_ds = None

    sys.stdout.write("Finish ts_peak.py for {0:s}".format(ts_file))

def main(cmdargs):
    fitTsPeak(cmdargs.input, cmdargs.output, \
              no_data=cmdargs.no_data, beg_doy=cmdargs.beg_doy, end_doy=cmdargs.end_doy)

def getCmdArgs():
    p = argparse.ArgumentParser(description="Find the peaks of time series of images")

    p.add_argument("-i", "--input", dest="input", required=True, default=None, help="Input raster file of multiple bands as time series of images")
    p.add_argument("--no_data", dest="no_data", required=False, type=float, default=-9999, help="The no-data value of the input raster file. Default: -9999")

    p.add_argument("--beg_doy", dest="beg_doy", required=False, type=int, default=None, help="The beginning DOY (INCLUSIVE) of the temporal section of a time series to search for peak. Default: 1")
    p.add_argument("--end_doy", dest="end_doy", required=False, type=int, default=None, help="The ending DOY (EXCLUSIVE) of the temporal section of a time series to search for peak. Default: number of bands in the input raster")

    p.add_argument("-o", "--output", dest="output", required=True, default=None, help="Output raster file of peaks of time series")

    cmdargs = p.parse_args()

    return cmdargs

if __name__=='__main__':
    cmdargs = getCmdArgs()
    main(cmdargs)
