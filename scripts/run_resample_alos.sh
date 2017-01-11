#!/bin/bash
#$ -pe omp 4
#$ -l mem_total=8
#$ -l h_rt=24:00:00
#$ -N run-resample-alos
#$ -V
#$ -m ae
#$ -t 1:4

# SGE_TASK_ID=1

alos_hh_imgfiles=(\
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073935/IMG-HH-ALOS2058563200-150624-WBDR2.1GUD.UTM48N.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073936/IMG-HH-ALOS2062703200-150722-WBDR2.1GUD.UTM48N.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073937/IMG-HH-ALOS2070983200-150916-WBDR2.1GUD.UTM48N.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073938/IMG-HH-ALOS2075123200-151014-WBDR2.1GUD.UTM48N.tif" \
                   )

alos_hv_imgfiles=(\
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073935/IMG-HV-ALOS2058563200-150624-WBDR2.1GUD.UTM48N.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073936/IMG-HV-ALOS2062703200-150722-WBDR2.1GUD.UTM48N.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073937/IMG-HV-ALOS2070983200-150916-WBDR2.1GUD.UTM48N.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/0000073938/IMG-HV-ALOS2075123200-151014-WBDR2.1GUD.UTM48N.tif" \
                   )

src_file=${alos_hh_imgfiles[$SGE_TASK_ID-1]}
dst_file=${src_file/%".tif"/".30M.tif"}
gdalwarp -tr 30 30 -r bilinear -srcnodata 0 -dstnodata -9999 ${src_file} ${dst_file}

src_file=${alos_hv_imgfiles[$SGE_TASK_ID-1]}
dst_file=${src_file/%".tif"/".30M.tif"}
gdalwarp -tr 30 30 -r bilinear -srcnodata 0 -dstnodata -9999 ${src_file} ${dst_file}
