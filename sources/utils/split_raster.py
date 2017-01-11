#!/usr/bin/env python
import os
import sys
import argparse

from osgeo import gdal

def main(cmdargs):
    if os.path.isfile(cmdargs.outdir):
        raise RuntimeError("{0:s} is a file while the program asks for a directory!".format(cmdargs.outdir))
    if not os.path.exists(cmdargs.outdir):
        os.mkdir(cmdargs.outdir)
    
    dset = gdal.Open(cmdargs.inraster)

    width = dset.RasterXSize
    height = dset.RasterYSize

    tw = cmdargs.tilexsize
    th = cmdargs.tileysize

    print "Input raster dimension = {0:d} x {1:d}\n".format(width, height)

    dset = None

    ndigit_w = len(str(width))
    ndigit_h = len(str(height))
    ndigit_tw = len(str(tw))
    ndigit_th = len(str(th))
    fname_base, fname_ext = os.path.splitext(os.path.basename(cmdargs.inraster))
    fname_format_str = "{4:s}_{{0:0{0:d}d}}_{{1:0{1:d}d}}_{{2:0{2:d}d}}_{{3:0{3:d}d}}{5:s}".format(ndigit_w, ndigit_h, ndigit_tw, ndigit_th, fname_base, fname_ext)
    for i in range(0, width, tw):
        for j in range(0, height, th):
            w = min(i+tw, width) - i
            h = min(j+th, height) - j
            out_fname = os.path.join(cmdargs.outdir, 
                                     fname_format_str.format(i, j, w, h))
            gdaltran_string = "gdal_translate -of ENVI -srcwin {0:d} {1:d} {2:d} {3:d} {4:s} {5:s}".format(i, j, w, h, cmdargs.inraster, out_fname)
            sys.stdout.write("{0:s}\n".format(gdaltran_string))
            sys.stdout.flush()
            os.system(gdaltran_string)

def getCmdArgs():
    p = argparse.ArgumentParser(description="Split a raster to multiple tiles of smaller raster files")
    
    p.add_argument("-i", "--input", dest="inraster", required=True, default=None, help="Input raster file to be split")
    p.add_argument("-O", "--output_directory", dest="outdir", required=True, default=None, help="Output directory for those tiles from raster split")
    p.add_argument("--tile_xsize", dest="tilexsize", type=int, required=True, default=None, help="Tile size along X coordinate direction (columns)")
    p.add_argument("--tile_ysize", dest="tileysize", type=int, required=True, default=None, help="Tile size along Y coordinate direction (rows)")

    cmdargs = p.parse_args()
    return cmdargs

if __name__ == "__main__":
    cmdargs = getCmdArgs()
    main(cmdargs)
