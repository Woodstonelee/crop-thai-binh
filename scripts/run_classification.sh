#!/bin/bash
#$ -pe omp 16
#$ -l mem_total=16
#$ -l h_rt=24:00:00
#$ -N run-classification
#$ -V
#$ -m ae
# #$ -t 1:5

PYCMD="/usr3/graduate/zhanli86/Workspace/src/projects/adb-paddy-rice-thai-binh/scripts/classify_image.py"

# NDVI time series from SG fitting
IMAGES=`for i in {001..365}; do echo /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-ts-sg/fitSG_NDVI_126046.2015${i}; done` 
python ${PYCMD} -i ${IMAGES} --trainF 1 --train "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-field/classification_training_data_pixel.txt" -c "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_sg_ndvi.img" --clsF "ENVI"

# Three Landsat scenes, all bands for classification
IMAGES="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LC82015191 /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015103_filled_GNSPI_filled_GNSPI /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015183_filled_GNSPI_filled_GNSPI"
python ${PYCMD} -i ${IMAGES} --trainF 1 --train "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-field/classification_training_data_pixel.txt" -c "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_lsat_scenes.img" --clsF "ENVI"
