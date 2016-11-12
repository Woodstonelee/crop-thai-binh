import os
import sys
import itertools
import glob
import warnings

import numpy as np
from osgeo import gdal, gdal_array, osr, ogr

import geo_ts as gt

def quadPeak(x, y, no_data=-9999, max_iter=100, diff_thresh=1):
    tmp_ind = np.where(y!=no_data)[0]
    x0 = x[tmp_ind]
    y0 = y[tmp_ind]
    
    x = np.copy(x0)
    y = np.copy(y0)
    for i in range(max_iter):    
        coef = np.polyfit(x, y, 2)
        quad_func = np.poly1d(coef)
        xfit = x0
        yfit = quad_func(xfit)
        ydiff = yfit - y0
        tmp_flag = ydiff < diff_thresh
        if np.sum(tmp_flag) == 0:
            break
        x = x0[tmp_flag]
        y = y0[tmp_flag]

    peak_x = -0.5*coef[1]/coef[0]
    if coef[0] < 0:
        peak_y = quad_func(peak_x)
    else:
        peak_y = no_data
    
    max_y = np.max(y)
    return peak_x, peak_y, quad_func, max_y

def fitTsPeak(ts_file, out_peak_file):
    # Settings for raster reading/writing and TS fitting
    no_data = -9999
    beg_doy = 225
    end_doy = 300

    # Read TS data and do quadratic fitting
    ts_meta = gt.getRasterMetaGdal(ts_file)
    sys.stdout.write('Reading time series data at once ... \n')
    sys.stdout.flush()
    ts_data_all = gt.readPixelsGdal(ts_file)

    peak_arr = np.zeros((ts_meta['RasterYSize'], ts_meta['RasterXSize']), dtype=np.float32) + no_data
    x = np.arange(ts_meta['RasterXSize'], dtype=np.int32)
    ts_x = np.arange(beg_doy-1, end_doy)
    for row in range(ts_meta['RasterYSize']):
        sys.stdout.write('Processing line {0:d}\n'.format(row+1))
        sys.stdout.flush()

        # y = np.zeros_like(x) + row
        # ts_data = gt.readPixelsGdal(ts_file, x, y)
        ts_data = ts_data_all[row*ts_meta['RasterXSize']:(row+1)*ts_meta['RasterXSize'], :]

        # # Quadratic fitting        
        # peak_x, peak_y, _, max_y = zip(*[quadPeak(ts_x, ts_data[i, ts_x.astype(int)], no_data=no_data) for i in range(len(x))])
        # peak_x = np.array(peak_x)
        # peak_y = np.array(peak_y)
        # max_y = np.array(max_y)
        # valid_ind = reduce(np.logical_and, (peak_x>beg_doy-1, peak_x<end_doy-1, peak_y>=max_y, peak_y!=no_data))
        # peak_y[np.where(np.logical_not(valid_ind))[0]] = no_data
        # peak_arr[y, x] = peak_y

        # Simple maximum
        peak_arr[row, :] = np.max(ts_data, axis=1)

    # Save array of peak values to a raster file
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(out_peak_file, peak_arr.shape[1], peak_arr.shape[0], 1, \
                           gdal_array.NumericTypeCodeToGDALTypeCode(peak_arr.dtype.type))
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(peak_arr)
    out_band.SetDescription('Peak of time series from \"{0:s}\" by quadratic fitting to the section between DOY {1:d} and {2:d}'.format(os.path.basename(ts_file), beg_doy, end_doy))
    out_band.SetNoDataValue(no_data)
    out_band.FlushCache()

    out_ds.SetGeoTransform(ts_meta['GeoTransform'])
    out_ds.SetProjection(ts_meta['ProjectionRef'])

    out_ds = None

def main():
    # ts_file = '/home/zhan/Windows-Shared/Workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/predicted_NDVI'
    # out_peak_file = '/home/zhan/Windows-Shared/Workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/predicted_NDVI_quadfit_peak.img'

    # ts_file = '/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/predicted_NDVI'
    # out_peak_file = '/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/predicted_NDVI_quadfit_peak.img'

    ts_file = '/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/fit_NDVI'
    out_peak_file = '/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/predicted_NDVI_max_peak.img'

    fitTsPeak(ts_file, out_peak_file)

if __name__=='__main__':
    main()
