"""Convert the bands of a raster file to indiviudal MATLAB mat files
to interface with MATLAB scripts. Each band of the raster file will be
saved in separate individual mat files.

Zhan Li, zhan.li@bu.edu
Created: Mon Oct 17 11:33:42 EDT 2016
"""

import sys
import argparse

from scipy.io import savemat

from osgeo import gdal, gdal_array, osr

gdal.AllRegister()

def readRaster(filename, band=None):
    """ Reads in a band from an images using GDAL
    Args:
      filename (str): filename to read from
      band (np.ndarray): 1D array of indices to the bands to read, 
          with first being 1, if None, read all bands
    Returns:
      dat (sequence of numpy 2D array): each list element is a band from the input image file.
      band (numpy 1d array): list of bands read to the *dat*
    """
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    if ds is None:
        sys.stderr.write("Failed to open the file {0:s}\n".format(filename))
        return None

    if band is None:
        band = np.arange(ds.RasterCount, dtype=np.int)+1

    dat = [None for b in band]
    for n, b in enumerate(band):
        dat[n] = ds.GetRasterBand(b).ReadAsArray()

    ds = None

    return dat, band

def getRasterProfile(filename):
    """Retrieve the meta information of an image using GDAL
    Args:
      filename: str
          file name to read from
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

    bands = [ds.GetRasterBand(idx+1) for idx in range(nbands)]
    nodata = [-9999 if b.GetNoDataValue() is None else b.GetNoDataValue() for b in bands]
    dtype = [gdal_array.GDALTypeCodeToNumericTypeCode(b.DataType).__name__ for b in bands]

    return {"RasterXSize":ncol, "RasterYSize":nrow, "RasterCount":nbands, \
            "DataType":dtype, "NoDataValue":nodata, \
            "GeoTransform":geo_trans, "ProjectionRef":proj_ref, \
            "Metadata":metadata}


def raster2Mat(raster_file, mat_file, matvar_prefix, export_profile=False):
    raster_profile = getRasterProfile(raster_file)
    outf_prefix = '.'.join(mat_file.split('.')[0:-1]) + '_'
    ndigits = len(str(raster_profile['RasterCount']))
    fmtstr = '{{0:0{0:d}d}}'.format(ndigits)
    outf_name_list = [outf_prefix+'b'+fmtstr.format(i+1) for i in range(raster_profile['RasterCount'])]
    matvar_name_list = [matvar_prefix+'_b'+fmtstr.format(i+1) for i in range(raster_profile['RasterCount'])]
    if export_profile:
        outf_profile = outf_prefix+'profile'
        # save raster profile to a mat file after flattening the dict of raster profile
        tmp = flatDict(raster_profile)
        # add the given prefix to the raster profile keys
        raster_profile_mat = {'{0:s}_{1:s}'.format(matvar_prefix, k):tmp[k] for k in tmp}
        savemat(outf_profile, raster_profile_mat, format='5')
    for b in range(raster_profile['RasterCount']):
        savemat(outf_name_list[b], {matvar_name_list[b]:readRaster(raster_file, [b+1])[0][0]}, format='5')

def flatDict(var_dict):
    """Flatten a nested dictionary variable, i.e. if an element of
    *var_dict* is also a dict variable, the elements of this dict
    element will be unfolded and its elements will be added to the
    *var_dict* with names consisted of names in the sequence of dicts
    delimited by '_'. For example:

    var_dict={'A':1, 'B':{'b1':2, 'b2':3}} will return

    {'A':1, 'B_b1':2, 'B_b2':3}
    """
    new_dict=dict()
    for k, v in var_dict.iteritems():
        if type(v) is dict:
            for kk, vv in v.iteritems():
                new_dict['{0:s}_{1:s}'.format(k,kk)] = vv
        else:
            new_dict[k] = v
    recursion = False
    for k, v in new_dict.iteritems():
        if type(v) is dict:
            recursion = True
            break
    if recursion:
        return flatDict(new_dict)
    else:
        return new_dict

def main(cmdargs):
    raster2Mat(cmdargs.img, cmdargs.matf, cmdargs.matv, cmdargs.exp_pf)

def getCmdArgs():
    p = argparse.ArgumentParser(description='Convert bands of a raster to MATLAB mat file')

    p.add_argument('-i', '--input', dest='img', required=True, default=None, help='Input raster image file from which bands are exported to mat file')
    p.add_argument('-o', '--output', dest='matf', required=True, default=None, help='Output file name as the prefix for the exported mat file')
    p.add_argument('--var_prefix', dest='matv', required=True, default=None, help='Prefix to the name of variables saved in mat file')
    p.add_argument('--export_profile', dest='exp_pf', default=False, action='store_true', help='If set, export the raster profile to a mat file. Default: false, do not export')

    cmdargs = p.parse_args()

    return cmdargs

if __name__=='__main__':
    cmdargs = getCmdArgs()
    main(cmdargs)
