#!/bin/bash
#$ -pe omp 1
#$ -l mem_total=2
#$ -l h_rt=24:00:00
#$ -N run-ts-peak
#$ -V
#$ -m ae
#$ -o ./run-ts-peak-dump/$JOB_NAME.o$JOB_ID.$TASK_ID
#$ -e ./run-ts-peak-dump/$JOB_NAME.e$JOB_ID.$TASK_ID
#$ -t 1:36

# SGE_TASK_ID=3

TSFILES=(
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0000_0000_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0000_0400_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0000_0800_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0000_1200_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0000_1600_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0000_2000_400_023" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0400_0000_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0400_0400_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0400_0800_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0400_1200_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0400_1600_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0400_2000_400_023" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0800_0000_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0800_0400_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0800_0800_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0800_1200_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0800_1600_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_0800_2000_400_023" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1200_0000_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1200_0400_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1200_0800_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1200_1200_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1200_1600_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1200_2000_400_023" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1600_0000_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1600_0400_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1600_0800_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1600_1200_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1600_1600_400_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_1600_2000_400_023" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_2000_0000_170_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_2000_0400_170_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_2000_0800_170_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_2000_1200_170_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_2000_1600_170_400" \
"/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split/thai_binh_ndvi_ts_at_2000_2000_170_023")

OUTDIR="/projectnb/echidna/lidar/zhanli86/workspace/data/projects/kaiyu-adb-crop/vietnam-fusion-ts-new/thai-bin-ndvi-ts-at-split-peak/"
if [[ ! -d ${OUTDIR} ]]; then
  mkdir -p ${OUTDIR}
fi

CMD="/usr3/graduate/zhanli86/Workspace/src/projects/crop-thai-binh/sources/ts/ts_peak.py"

OUTFILE=${OUTDIR}/$(basename ${TSFILES[${SGE_TASK_ID}-1]})"_peak.bin"
python ${CMD} -i ${TSFILES[${SGE_TASK_ID}-1]} -o ${OUTFILE} --beg_doy 220 --end_doy 301
