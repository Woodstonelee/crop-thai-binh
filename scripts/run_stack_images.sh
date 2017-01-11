#!/bin/bash
# #$ -pe omp 4
# #$ -l mem_total=8
# #$ -l h_rt=24:00:00
# #$ -N run-stack-images
# #$ -V
# #$ -m ae
# #$ -t 1:4

# SGE_TASK_ID=1

IMAGES=(\
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LC82015191" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015103_filled_GNSPI_filled_GNSPI" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015183_filled_GNSPI_filled_GNSPI" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073935/IMG-HH-ALOS2058563200-150624-WBDR2.1GUD.UTM48N.30M.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073936/IMG-HH-ALOS2062703200-150722-WBDR2.1GUD.UTM48N.30M.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073937/IMG-HH-ALOS2070983200-150916-WBDR2.1GUD.UTM48N.30M.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073938/IMG-HH-ALOS2075123200-151014-WBDR2.1GUD.UTM48N.30M.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073935/IMG-HV-ALOS2058563200-150624-WBDR2.1GUD.UTM48N.30M.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073936/IMG-HV-ALOS2062703200-150722-WBDR2.1GUD.UTM48N.30M.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073937/IMG-HV-ALOS2070983200-150916-WBDR2.1GUD.UTM48N.30M.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073938/IMG-HV-ALOS2075123200-151014-WBDR2.1GUD.UTM48N.30M.tif" \
)

printf "%s\n" ${IMAGES[@]} | xargs -I {} gdal_merge.py -of "ENVI" {}

# gdal_merge.py
