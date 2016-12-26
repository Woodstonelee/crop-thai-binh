"""Save a 2D array in a MATLAB mat file to a single-band ENVI raster
file.

Zhan Li, zhanli86@bu.edu
Created: Tue Oct 18 14:29:48 EDT 2016

"""

import sys
import os
import argparse

import numpy as np
from scipy.io import loadmat

from osgeo import gdal, gdal_array, osr

gdal.AllRegister()

def mat2Raster(mat_file, matvar_name, profile_mat_file, raster_file, fmt='ENVI'):
    mat_dict = loadmat(profile_mat_file)
    n, g, p = 0, 0, 0
    for k in mat_dict:
        if k.find('NoDataValue') > -1:
            nodata_key = k
            n = n+1
        elif k.find('GeoTransform') > -1:
            geo_key = k
            g = g+1
        elif k.find('ProjectionRef') > -1:
            proj_key = k
            p = p+1
    raster_profile = {'NoDataValue':mat_dict[nodata_key][0][0] if (type(mat_dict[nodata_key][0]) is np.ndarray) else mat_dict[nodata_key][0], \
                      'GeoTransform':mat_dict[geo_key][0], \
                      'ProjectionRef':mat_dict[proj_key][0].encode('ascii', 'ignore')}
    
    mat_dict = loadmat(mat_file)
    for k in mat_dict:
        if k.find(matvar_name) > -1:
            img_key = k
            break
    img_array = mat_dict[img_key]
    raster_profile['Description'] = img_key
    writeRaster(img_array, raster_file, raster_profile, fmt=fmt)

def writeRaster(img_array, out_file, raster_profile, fmt='ENVI'):
    img_array = img_array.astype(np.float32)
    
    driver = gdal.GetDriverByName(fmt)
    out_ds = driver.Create(out_file, img_array.shape[1], img_array.shape[0], 1, \
                           gdal_array.NumericTypeCodeToGDALTypeCode(img_array.dtype.type))
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(img_array)
    out_band.SetDescription(raster_profile['Description']) # set band name
    out_band.SetNoDataValue(raster_profile['NoDataValue'])
    out_band.FlushCache()
    
    out_ds.SetGeoTransform(raster_profile['GeoTransform'])
    out_ds.SetProjection(raster_profile['ProjectionRef'])

    out_ds = None

def main(cmdargs):
    # mat_file = '/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/mat-files/smoothn_predicted_ndvi_b360.mat'
    # profile_mat_file = '/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/mat-files/predicted_ndvi_profile.mat'
    # raster_file = '/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/mat-files/raster-files/smoothn_predicted_ndvi_b360.bin'
    # fmt = 'ENVI'

    mat_file = cmdargs.matf
    profile_mat_file = cmdargs.pfmat
    raster_file = cmdargs.raster
    fmt = cmdargs.of

    matvar_name = '.'.join(os.path.basename(mat_file).split('.')[0:-1])

    mat2Raster(mat_file, matvar_name, profile_mat_file, raster_file, fmt=fmt)

def getCmdArgs():
    p = argparse.ArgumentParser(description='Save a 2D array in a MATLAB mat file to a raster file')

    p.add_argument('-i', '--input', dest='matf', required=True, default=None, help='Input mat file of the 2D array')
    p.add_argument('-o', '--output', dest='raster', required=True, default=None, help='Output file name of the raster')
    p.add_argument('--profile_mat', dest='pfmat', required=True, default=None, help='Mat file of the raster profie')
    p.add_argument('--of', dest='of', required=False, default='ENVI', choices=['ENVI'], help='Format of the output raster file')

    cmdargs = p.parse_args()

    return cmdargs

if __name__ == '__main__':
    cmdargs = getCmdArgs()
    main(cmdargs)
