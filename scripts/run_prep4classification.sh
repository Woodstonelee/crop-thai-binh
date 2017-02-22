#!/bin/bash
#$ -pe omp 4
#$ -l mem_total=16
#$ -l h_rt=24:00:00
#$ -N run-prep4classification
#$ -V
#$ -m ae
#$ -o ./run-sh-dump/$JOB_NAME.o$JOB_ID
#$ -e ./run-sh-dump/$JOB_NAME.e$JOB_ID

# convert all ALOS into the same projection as Landsat and resample to 30 m.
ALOS_IMGS=(\
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073936/IMG-HH-ALOS2062703200-150722-WBDR2.1GUD.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073936/IMG-HV-ALOS2062703200-150722-WBDR2.1GUD.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073937/IMG-HV-ALOS2070983200-150916-WBDR2.1GUD.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073937/IMG-HH-ALOS2070983200-150916-WBDR2.1GUD.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073938/IMG-HH-ALOS2075123200-151014-WBDR2.1GUD.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073938/IMG-HV-ALOS2075123200-151014-WBDR2.1GUD.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073935/IMG-HV-ALOS2058563200-150624-WBDR2.1GUD.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073935/IMG-HH-ALOS2058563200-150624-WBDR2.1GUD.tif" \
)

# printf "%s\n" ${ALOS_IMGS[@]} | xargs -I{} echo {} | sed s/.tif// | xargs -I{} gdalwarp -overwrite -srcnodata 0 -t_srs "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/dest_srs.wkt" -tr 30 30 -r cubic {}.tif {}".UTM48N.30M.tif"

# stack all ALOS together
NEW_ALOS_IMGS=()
for f in ${ALOS_IMGS[@]}; do
  NEW_ALOS_IMGS+=("${f%.tif}.UTM48N.30M.tif")
done

ALOS_STACK="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/thai_binh_alos_stack.utm48n.30m.tif"
gdal_merge.py -n 0 -a_nodata 0 -o ${ALOS_STACK} -separate ${NEW_ALOS_IMGS[@]}
# Convert DN to dB
gdal_calc.py --format="GTiff" --overwrite --type="Float32" --allBands=A -A ${ALOS_STACK} --outfile=${ALOS_STACK%".tif"}".db.tif" --calc="nan_to_num(10*log10(A.astype(float)**2/1.9952623149688828e+08))*(A>0) + (A<=0)*-9999" --NoDataValue=-9999

# # generate a shape file to crop the Landsat data
# CUT_SHP="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/thai_binh_alos_stack.utm48n.30m.tif.shp"
# gdaltindex ${CUT_SHP} ${ALOS_STACK}

# # cut the Landsat rasters
# # NDVI time series from SG fitting
# IMAGES=`for i in {001..365}; do echo /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-ts-sg/fitSG_NDVI_126046.2015${i}; done`
# printf "%s\n" ${IMAGES[@]} | xargs -I{} gdalwarp -of "ENVI" -overwrite -r cubic -srcnodata -9999 -dstnodata -9999 -cutline ${CUT_SHP} -crop_to_cutline {} {}"_alos_precut.bin"
# printf "%s\n" ${IMAGES[@]} | xargs -I{} gdal_calc.py --format="GTiff" --overwrite --allBands=A -A {}"_alos_precut.bin" --outfile={}"_alos_cut.tif" --calc="nan_to_num(A)*(logical_and(nan_to_num(A)>0, nan_to_num(A)<1))+(logical_or(nan_to_num(A)<=0, nan_to_num(A)>=1))*-9999" --NoDataValue=-9999

# IMAGES=(\
# "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LC82015191" \
# "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015103_filled_GNSPI_filled_GNSPI" \
# "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015183_filled_GNSPI_filled_GNSPI" \
# )
# # WTF, these three images are crap, containing NaN values, nodata not being common -9999 for Landsat but 0!
# # Clean up these three images first before cutting them to ALOS extent

# # # Interesting gdalwarp bug, the nodata value is not correctly excluded in the resampling procedure
# # printf "%s\n" ${IMAGES[@]} | xargs -I{} gdal_calc.py --overwrite --allBands=A -A {} --outfile={}"_clean.tif" --calc="nan_to_num(A)*(logical_and(nan_to_num(A)>0, nan_to_num(A)<1))+(reduce(logical_or,(nan_to_num(A)<=0, nan_to_num(A)>=1)))*-9999" --NoDataValue=-9999
# # printf "%s\n" ${IMAGES[@]} | xargs -I{} gdalwarp -overwrite -r cubic -srcnodata -9999 -dstnodata -9999 -cutline ${CUT_SHP} -crop_to_cutline {}"_clean.tif" {}"_alos_cut.tif"

# printf "%s\n" ${IMAGES[@]} | xargs -I{} gdal_calc.py --format="ENVI" --overwrite --allBands=A -A {} --outfile={}"_clean.bin" --calc="nan_to_num(A)*(logical_and(nan_to_num(A)>0, nan_to_num(A)<1))+(logical_or(nan_to_num(A)<=0, nan_to_num(A)>=1))*-9999" --NoDataValue=-9999
# printf "%s\n" ${IMAGES[@]} | xargs -I{} gdalwarp -of "ENVI" -overwrite -r cubic -srcnodata -9999. -dstnodata -9999 -cutline ${CUT_SHP} -crop_to_cutline {}"_clean.bin" {}"_alos_precut.bin"
# printf "%s\n" ${IMAGES[@]} | xargs -I{} gdal_calc.py --format="GTiff" --overwrite --allBands=A -A {}"_alos_precut.bin" --outfile={}"_alos_cut.tif" --calc="nan_to_num(A)*(logical_and(nan_to_num(A)>0, nan_to_num(A)<1))+(logical_or(nan_to_num(A)<=0, nan_to_num(A)>=1))*-9999" --NoDataValue=-9999
