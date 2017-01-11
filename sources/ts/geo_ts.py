import os
import sys
import itertools
import glob
import warnings

import numpy as np
from osgeo import gdal, gdal_array, osr, ogr

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# import seaborn as sns
# sns.set_context("paper")
# sns.set_style("whitegrid")
# dpi = 300

# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=False) # run at the start of every ipython notebook to use plotly.offline
#                      # this injects the plotly.js source files into the notebook
# import plotly.tools

gdal.AllRegister()

def readPixelGdal(filename, x, y):
    """ Reads in a pixel of data from an images using GDAL
    Args:
      filename (str): filename to read from
      x (int): column
      y (int): row
    Returns:
      np.ndarray: 1D array (nband) containing the pixel data
    """
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    if ds is None:
        sys.stderr.write("Failed to open the file {0:s}\n".format(filename))
        return None
    
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
        ds.GetRasterBand(1).DataType)

    dat = np.empty(ds.RasterCount, dtype=dtype)
    for i in range(ds.RasterCount):
        dat[i] = ds.GetRasterBand(i + 1).ReadAsArray(x, y, 1, 1)
        
    ds = None

    return dat

def readPixelsGdal(filename, x=None, y=None, band=None):
    """ Reads in multiple pixel of data from an images using GDAL
    Args:
      filename (str): filename to read from
      x (numpy 1D array): columns, with first pixel being 0
      y (numpy 1D array): rows, with first pixel being 0
      band (numpy 1D array): band indices, with first band being 1
    Returns:
      np.ndarray: 2D array (npixel, nband) containing the pixel data
    """
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    if ds is None:
        sys.stderr.write("Failed to open the file {0:s}\n".format(filename))
        return None
    
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
        ds.GetRasterBand(1).DataType)

    if band is None:
        band = np.arange(ds.RasterCount, dtype=np.int) + 1
    else:
        band = np.array(band).astype(int)

    if ((x is not None) and (y is not None)):
        x = np.array(x).astype(int)
        y = np.array(y).astype(int)
        dat = np.empty((len(x), len(band)), dtype=dtype)
        for i,ib in enumerate(band):
            tmp = ds.GetRasterBand(ib).ReadAsArray()
            dat[:, i] = tmp[y, x]
    else:
        dat = np.empty((ds.RasterXSize*ds.RasterYSize, len(band)), dtype=dtype)
        for i,ib in enumerate(band):
            tmp = ds.GetRasterBand(ib).ReadAsArray()
            dat[:, i] = tmp.flatten()

    ds = None

    return dat

# get time series and doy from a list of single-band image files
def getTsFromImgs(imgfiles, x, y):
    """
    Parameters: **imgfiles**, *list of str*
                  List of file names to the single-band image files
                **x, y**, *numpy 1D array*
                  Columns (samples or x) and rows (lines or y)
    """
    n = len(imgfiles)
    ts_data = np.zeros((len(x), n))        
    for n, imgf in enumerate(imgfiles):
        ts_data[:, n] = readPixelsGdal(imgf, x, y)[:, 0]
    fnames = [os.path.basename(imgf) for imgf in imgfiles]
    return fnames, ts_data

def getRasterMetaGdal(filename, band_idx=1):
        """Retrieve the meta information of an image using GDAL
        Args:
          filename: str
              file name to read from
          band_idx: int, default 1
              index to the band to retrieve meta data for, with the first being 1.
        Returns:
          meta: dict
              returned meta data records about the image
        """
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            sys.stderr.write("Failed to open the file {0:s}\n".format(filename))
            return None

        ncol = ds.RasterXSize
        nrow = ds.RasterYSize
        nbands = ds.RasterCount

        geo_trans = ds.GetGeoTransform()

        proj_ref = ds.GetProjectionRef()

        metadata = {dm:ds.GetMetadata(dm) for dm in ds.GetMetadataDomainList()}

        # bands = [ds.GetRasterBand(idx+1) for idx in range(nbands)]
        # nodata = [-9999 if b.GetNoDataValue() is None else b.GetNoDataValue() for b in bands]
        # dtype = [gdal_array.GDALTypeCodeToNumericTypeCode(b.DataType) for b in bands]

        band = ds.GetRasterBand(band_idx)
        nodata = -9999 if band.GetNoDataValue() is None else band.GetNoDataValue()
        dtype = gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)

        ds = None
        
        return {"RasterXSize":ncol, "RasterYSize":nrow, "RasterCount":nbands, \
                "DataType":dtype, "NoDataValue":nodata, \
                "GeoTransform":geo_trans, "ProjectionRef":proj_ref, \
                "Metadata":metadata}

def proj2Pixel(geoMatrix, x, y, ret_int=True):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location, with 0 being the first pixel, of 
    a geospatial coordinate in the projection system.
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    if ret_int:
        sample = int((x - ulX) / xDist)
        line = int((y - ulY) / yDist)
    else:
        sample = (x - ulX) / xDist
        line = (y - ulY) / yDist
    return (sample, line)

def pixel2Proj(geoMatrix, sample, line):
    """
    Covnert pixel location (sample, line), 
    with 0 being the first pixel, to the 
    geospatial coordinates in the projection system, 
    with (x, y) being the UL corner of the pixel.
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    x = sample * xDist + ulX
    y = line * yDist + ulY
    return (x, y)

def geo2Proj(filename, lon, lat):
    """
    Convert geographic coordinates to projected coordinates 
    given the input raster file
    """
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    out_sr = osr.SpatialReference()
    out_sr.ImportFromWkt(ds.GetProjectionRef())
    in_sr = out_sr.CloneGeogCS()
    coord_trans = osr.CoordinateTransformation(in_sr, out_sr)
    return coord_trans.TransformPoint(lon, lat)[0:2]

def proj2Geo(filename, x, y):
    """
    Convert projected coordinates to geographic coordinates
    given the input raster file    
    """
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    in_sr = osr.SpatialReference()
    in_sr.ImportFromWkt(ds.GetProjectionRef())
    out_sr = in_sr.CloneGeogCS()
    coord_trans = osr.CoordinateTransformation(in_sr, out_sr)
    return coord_trans.TransformPoint(x, y)[0:2]

def geo2Pixel(filename, lon, lat, ret_int=True):
    """
    Convert geographic coordinates to image coordinates 
    given the input raster file
    """
    ds_meta = getRasterMetaGdal(filename)
    proj_coord = geo2Proj(filename, lon, lat)
    return proj2Pixel(ds_meta["GeoTransform"], proj_coord[0], proj_coord[1], ret_int)

def pixel2Geo(filename, sample, line):
    """
    Convert the image coordinates to geographic coordinates given the input raster file.
    """
    ds_meta = getRasterMetaGdal(filename)
    proj_coord = pixel2Proj(ds_meta["GeoTransform"], sample, line)
    return proj2Geo(filename, proj_coord[0], proj_coord[1])

def plotTs(doy_list, ts_data_list, meta_all, \
           style_list=None, plot_kw_dict_list=None, \
           fig_kw_dict=dict(figsize=(8, 4)), \
           ax_kw_dict=dict(xlabel="DOY", ylabel="Value", xlim=(-1.5, 366.5)), \
           use_plotly=False, save_fig=False, \
           select_doy=None, nodata=-9999):
    
    # valid_flag_list = [np.logical_and(ts_data!=nodata, ts_data>0) for ts_data in ts_data_list]
    valid_flag_list = [ts_data!=nodata for ts_data in ts_data_list]
    tmp_list = [np.sum(valid_flag, axis=1)>0 for valid_flag in valid_flag_list]
    select_keys_list = [tmp[tmp].index for tmp in tmp_list]
    ymin = np.nanmin([np.nanmin(ts_data[valid_flag]) for ts_data, valid_flag in itertools.izip(ts_data_list, valid_flag_list)])
    ymax = np.nanmax([np.nanmax(ts_data[valid_flag]) for ts_data, valid_flag in itertools.izip(ts_data_list, valid_flag_list)])
    
    if ymin is np.nan or ymax is np.nan:
        warnings.warn("No valid values in all the given time series")
        return None
    
    # colors = mpl.colors.cnames.keys()[len(doy_list)]
    
    # sort doy first
    # doy_sort_ind_list = [np.argsort(doy_arr) for doy_arr in doy_list]

    fig_key_list = []
    for ik, point_key in enumerate(meta_all.index):
        if "num" in fig_kw_dict:
            # fig, ax = plt.subplots(**fig_kw_dict)
            fig = plt.figure(**fig_kw_dict)
            ax = fig.add_subplot(111)
            fig_key = fig_kw_dict["num"]
        else:
            # fig, ax = plt.subplots(num=point_key, **fig_kw_dict)
            fig = plt.figure(num=point_key, **fig_kw_dict)
            ax = fig.add_subplot(111)
            fig_key = point_key
        for n, (doy, ts_data) in enumerate(itertools.izip(doy_list, ts_data_list)):
            if not (point_key in select_keys_list[n]):
                continue
            flag = valid_flag_list[n].loc[point_key, :]
            flag = np.where(flag)[0]
            sort_ind = np.argsort(doy[flag])
            ts_ind = ts_data.columns.get_values()[flag]
            if style_list is not None:
                if plot_kw_dict_list is not None:
                    ax.plot(doy[flag][sort_ind], ts_data.loc[point_key, ts_ind].iloc[sort_ind], style_list[n], **plot_kw_dict_list[n])
                else:
                    ax.plot(doy[flag][sort_ind], ts_data.loc[point_key, ts_ind].iloc[sort_ind], style_list[n])
            else:
                if plot_kw_dict_list is not None:
                    ax.plot(doy[flag][sort_ind], ts_data.loc[point_key, ts_ind].iloc[sort_ind], **plot_kw_dict_list[n])
                else:
                    # ax.plot(doy[np.nonzero(flag)], ts_data.loc[point_key, flag])
                    ax.plot(doy[flag][sort_ind], ts_data.loc[point_key, ts_ind].iloc[sort_ind])

        if not ("ylim" in ax_kw_dict):
            ax.set_ylim(ymin, ymax)
        if not ("xlim" in ax_kw_dict):
            ax.set_xlim(-1.5, 366.5)
        if not ("title" in ax_kw_dict):
            ax.set_title("{0:s}, {1:s}".format(point_key, meta_all.loc[point_key]))
        plt.setp(ax, **ax_kw_dict)

        if not use_plotly:
            if select_doy is not None:
                for d in select_doy:
                    ax.plot(np.array([d, d]), ax.get_ylim(), '--k')
            ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    
        if not (fig_key in fig_key_list):
            fig_key_list.append(fig_key)

    for fig_key in fig_key_list:
        fig = plt.figure(num=fig_key)
        if use_plotly:
            plotly_fig = plotly.tools.mpl_to_plotly(fig)
            plotly_fig['layout']['showlegend'] = True
            plotly_fig['layout']['legend'] = dict(orientation="h")
            iplot(plotly_fig)
        else:
            plt.show()

        if save_fig:
            if "ylabel" in ax_kw_dict:
                ylabel = ax_kw_dict['ylabel']
            else:
                ylabel = "Value"
            plt.savefig("../figures/ts_{0:s}_{1:s}.png".format(ylabel.lower(), fig_key.replace(" ", "_").lower()), \
                        dpi = dpi, bbox_inches="tight", pad_inches=0.)

def drawImages():
    """Export the designated bands of given raster files to easy-to-look
    images (png format) for quick preview.

    Parameters: **

    """
    pass
