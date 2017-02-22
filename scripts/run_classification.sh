#!/bin/bash
#$ -pe omp 8
#$ -l mem_total=16
#$ -l h_rt=12:00:00
#$ -N run-classification
#$ -V
#$ -m ae
#$ -o ./run-sh-dump/$JOB_NAME.o$JOB_ID.$TASK_ID
#$ -e ./run-sh-dump/$JOB_NAME.e$JOB_ID.$TASK_ID
#$ -t 1:4

# SGE_TASK_ID=1

PYCMD="/usr3/graduate/zhanli86/Workspace/src/projects/crop-thai-binh/sources/imgcls/classify_image.py"

IMAGES_ARR=(\
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LC82015191_alos_cut.tif /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015103_filled_GNSPI_filled_GNSPI_alos_cut.tif /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015183_filled_GNSPI_filled_GNSPI_alos_cut.tif" \
"`for i in {001..365}; do echo /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-ts-sg/fitSG_NDVI_126046.2015${i}_alos_cut.tif; done`" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/thai_binh_alos_stack.utm48n.30m.db.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LC82015191_alos_cut.tif /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015103_filled_GNSPI_filled_GNSPI_alos_cut.tif /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015183_filled_GNSPI_filled_GNSPI_alos_cut.tif /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-alos/thai_binh_alos_stack.utm48n.30m.db.tif" \
)

CLS_ARR=(\
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_lsat_alos_cut.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_sg_ndvi_alos_cut.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_alos_hh_hv.tif" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_lsat_alos_hh_hv.tif" \
)

CLS_FMTS=(\
"GTiff" \
"GTiff" \
"GTiff" \
"GTiff" \
)

DIAG_ARR=(\
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_lsat_alos_cut_diagnosis.txt" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_sg_ndvi_alos_cut_diagnosis.txt" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_alos_hh_hv_diagnosis.txt" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_lsat_alos_hh_hv_diagnosis.txt" \
)

python ${PYCMD} -i ${IMAGES_ARR[$SGE_TASK_ID-1]} --trainF 2 --train "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-field/classification_training_data_polygon_v1_alos_cut.tif" -c ${CLS_ARR[$SGE_TASK_ID-1]} --clsF ${CLS_FMTS[$SGE_TASK_ID-1]} --diagnosis ${DIAG_ARR[$SGE_TASK_ID-1]}

# # NDVI time series from SG fitting
# IMAGES=`for i in {001..003}; do echo /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-ts-sg/fitSG_NDVI_126046.2015${i}; done` \
# python ${PYCMD} -i ${IMAGES[@]1:2} --trainF 1 --train "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-field/classification_training_data_pixel.txt" -c "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_sg_ndvi.img" --clsF "GTiff"
# # Three Landsat scenes, all bands for classification
# IMAGES="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LC82015191 /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015103_filled_GNSPI_filled_GNSPI /projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-landsat-gapfilled/LE72015183_filled_GNSPI_filled_GNSPI"
# python ${PYCMD} -i ${IMAGES} --trainF 1 --train "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-field/classification_training_data_pixel.txt" -c "/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-cls/vietnam_thai_bin_cls_rf_lsat_scenes.img" --clsF "GTiff"
