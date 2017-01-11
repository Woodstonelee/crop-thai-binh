import os
import sys
import itertools
import glob
import warnings
import argparse

import numpy as np
from osgeo import gdal, gdal_array, osr, ogr

import geo_ts as gt

def mergeTimeSeries(img, priority, nodata):
    """Merge valid values from multiple time series together to one time
    series according to their priority.
    
    Parameters: **img**, *list of numpy 2D array*
                    List of input image arrays to be merged
                **priority**, *numpy 1D array like*
                    Priority of each image array
                **nodata**, *numpy 1D array like*
                    No-data values for each image array

    Returns:    **merged**, *numpy 2D array*
                    Merged image array
    """
    if np.unique([len(img), len(priority), len(nodata)]).size > 1:
        raise RuntimeError("Number of given image arrays, priority values and no-data values are not the same!")
    
    nodata_priority = np.min(priority) - 1
#    invalid_flag = [dat==nd for dat, nd in itertools.izip(img, nodata)]
    priority_arr = [np.ones_like(dat)*pv for dat, pv in itertools.izip(img, priority)]
    priority_arr = [np.where(dat==nd, np.ones_like(dat)*nodata_priority, pa) for dat, nd, pa in itertools.izip(img, nodata, priority_arr)]
    priority_top = np.max(np.array(priority_arr), axis=0)
    # for each pixel in each image, a flag to indicate if this pixel is included in our merge calcualtion. 
    select_flag = [pa==priority_top for pa in priority_arr]
    # determine the weight of each pixel in each image according to
    # how many images for each pixel location to be merged
    img_weight = np.sum(np.array(select_flag), axis=0)
    tmp_flag = img_weight>0
    img_weight[tmp_flag] = 1./img_weight[tmp_flag]
    merged = [sf*img_weight*dat for sf, dat in itertools.izip(select_flag, img)]
    merged = np.sum(np.array(merged), axis=0)
    
    return merged

def parseInMetaFile(meta_file):
    """Parse the input meta data CSV file.
    
    Parameters: **meta_file**, *str*:
                    File name of input CSV file of meta data.

    Returns:    **doy**, *numpy 1D array*:
                    Day of year
                **priority**, *numpy 1D array*:
                    Priority
                **band**, *numpy 1D array*:
                    Index to band to read in a raster file with first
                    band being 1. 
                **no_data**, *numpy 1D array*:
                    Value of no-data pixels in a band
                **raster_file_name**, *numpy 1D array*:
                    File names of raster files
    """
    inmeta = np.genfromtxt(meta_file, delimiter=",", dtype=None,
                           skip_header=1, 
                           names=None, filling_values=np.nan,
                           unpack=False)
    return inmeta['f0'], inmeta['f1'], inmeta['f2'], inmeta['f3'], inmeta['f4']
    

def stackRaster2TimeSeries(raster_fname, doy, priority, band_idx, no_data, \
                           out_fname, out_format):
    """Stack a series of bands from multiple rasters to a multi-band
    raster of time series according to the priority of each band of
    the input rasters.

    """
    # First get the meta info about all the bands from all the rasters
    # and double check if their dimensions are all the same.
    band_meta_list = [gt.getRasterMetaGdal(fname, band_idx=bd) for fname, bd in zip(raster_fname, band_idx)]
    img_xsize_list, img_ysize_list, img_nband_list = \
        zip(*[(bm['RasterXSize'], bm['RasterYSize'], bm['RasterCount'])
              for bm in band_meta_list])
    if np.unique(img_xsize_list).size > 1 or np.unique(img_ysize_list).size > 1:
        raise RuntimeError("Dimensions of bands are not all the same!")
    if np.sum(band_idx > np.array(img_nband_list)) > 1:
        raise RuntimeError("Some given indices to band are larger than the number of bands in the corresponding raster files!")

    uniq_doy, uniq_cnt = np.unique(doy, return_counts=True)
    sort_idx = np.argsort(uniq_doy)
    uniq_doy = uniq_doy[sort_idx]
    uniq_cnt = uniq_cnt[sort_idx]
    
    # set up output raster file
    out_type = np.zeros(1,dtype=float).dtype.type
    out_img_xsize = img_xsize_list[0]
    out_img_ysize = img_ysize_list[0]
    out_raster_profile = band_meta_list[0]
    out_no_data = -9999
    driver = gdal.GetDriverByName(out_format)
    out_ds = driver.Create(out_fname, out_img_xsize, out_img_ysize, len(uniq_doy), \
                           gdal_array.NumericTypeCodeToGDALTypeCode(out_type))
    for i, (d, cnt) in enumerate(zip(uniq_doy, uniq_cnt)):
        curr_idx = np.where(doy==d)[0]
        if cnt>1:
            print "Need to merge for {0:d}".format(d)
            # merge the two bands
            dat_list = [gt.readPixelsGdal(raster_fname[ci], \
                                          band=band_idx[[ci]]).reshape(img_ysize_list[ci], img_xsize_list[ci]) \
                        for ci in curr_idx]
            priority_list = priority[curr_idx]
            no_data_list = no_data[curr_idx]
            out_dat = mergeTimeSeries(dat_list, priority_list, no_data_list).astype(out_type)
        else:
            # Read this band and later directly write it to the output
            # raster of time series.
            out_dat = gt.readPixelsGdal(raster_fname[curr_idx[0]], band=band_idx[curr_idx])
            out_dat = out_dat.astype(out_type).reshape(out_img_ysize, out_img_xsize)

        out_band = out_ds.GetRasterBand(i+1)
        out_band.WriteArray(out_dat)
        # set band name
        out_band.SetDescription("DOY_{0:03d}".format(d))
        # set no data value
        out_band.SetNoDataValue(out_no_data)
        out_band.FlushCache()

    out_ds.SetGeoTransform(out_raster_profile['GeoTransform'])
    out_ds.SetProjection(out_raster_profile['ProjectionRef'])

    out_ds = None


def main(cmdargs):
    doy, priority, band_idx, no_data, raster_fname = parseInMetaFile(cmdargs.input)
    stackRaster2TimeSeries(raster_fname, doy, priority, band_idx, no_data, \
                           cmdargs.output, cmdargs.outformat)

def getCmdArgs():
    p = argparse.ArgumentParser(description="Stack rasters and create a raster of time series data")
    p.add_argument("-i", "--input", dest="input", required=True, default=None, help="Input CSV file of meta data providing five columns in order: ('doy', 'priority', 'band', 'no_data', 'raster_file_name'), where 'doy' as day of year; 'priority' as the priority of a band to decide which band data to use for a pixel if valid values from multiple DOY exist, by favoring band with higher priority value; 'band' as the index to the band to use from a raster with 1 being the first band; 'no_data' as the no-data value in a band; 'raster_file_name' as the file name of a raster.")

    p.add_argument("-o", "--output", dest="output", required=True, default=None, help="Output raster file of the time series data")
    p.add_argument("--of", dest="outformat", required=False, default="ENVI", help="Format of output raster. Default: ENVI")

    cmdargs = p.parse_args()

    return cmdargs

if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
