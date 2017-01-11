#!/bin/bash
#$ -pe omp 4
#$ -l mem_total=4
#$ -l h_rt=24:00:00
#$ -N run-mosaic-imgs
#$ -V
#$ -m ae

IMGFILES=(
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0000_0000_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0000_0400_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0000_0800_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0000_1200_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0000_1600_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0000_2000_400_023_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0400_0000_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0400_0400_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0400_0800_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0400_1200_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0400_1600_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0400_2000_400_023_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0800_0000_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0800_0400_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0800_0800_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0800_1200_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0800_1600_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_0800_2000_400_023_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1200_0000_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1200_0400_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1200_0800_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1200_1200_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1200_1600_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1200_2000_400_023_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1600_0000_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1600_0400_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1600_0800_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1600_1200_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1600_1600_400_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_1600_2000_400_023_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_2000_0000_170_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_2000_0400_170_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_2000_0800_170_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_2000_1200_170_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_2000_1600_170_400_peak.bin" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/thai_binh_ndvi_ts_at_2000_2000_170_023_peak.bin")

OUTFILE="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai_binh_ndvi_ts_at_peak.tif"

# gdal_merge.py -o ${OUTFILE} -of "ENVI" -n -9999 -a_nodata -9999  ${IMGFILES[@]}
gdal_merge.py -o ${OUTFILE} -of "GTiff" ${IMGFILES[@]}
