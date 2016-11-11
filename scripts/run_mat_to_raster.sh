#!/bin/bash
#$ -pe omp 4
#$ -l mem_total=8
#$ -l h_rt=24:00:00
#$ -N run-mat-to-raster
#$ -V
#$ -m ae

M2RCMD="/usr3/graduate/zhanli86/Workspace/src/projects/adb-paddy-rice-thai-binh/scripts/mat_to_raster.py"

MATFILES=()
for i in {001..365}; do
    MATFILES+=("/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/mat-files/smoothn_predicted_ndvi_b${i}.mat")
done

PFMAT_FILE="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/mat-files/predicted_ndvi_profile.mat"

OUTDIR="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/mat-files/raster-files/"
OUTFILE="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts/mat-files/raster-files/smoothn_predicted_ndvi.bin"

RASTER_FILES=""
for ((i=0; i<${#MATFILES[@]}; i++)); do
    TMP=$(basename ${MATFILES[${i}]} ".mat")
    OUTFILE=${OUTDIR}"/"${TMP}".bin"
    RASTER_FILES=${RASTER_FILES}" "${OUTFILE}
    python ${M2RCMD} -i "${MATFILES[${i}]}" -o "${OUTFILE}" --profile_mat "${PFMAT_FILE}" --of "ENVI"
done

# stack images
# gdal_merge.py -o "${OUTFILE}" -of "ENVI" -a_nodata -9999 -separate ${RASTER_FILES}
