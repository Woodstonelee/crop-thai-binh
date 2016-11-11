#!/bin/bash
#$ -pe omp 16
#$ -l mem_total=48
#$ -l h_rt=24:00:00
#$ -N run-ndvi-smoothn
#$ -V
#$ -m ae

ML="/usr/local/bin/matlab -nodisplay -nojvm -r "

ML_CMD="/usr3/graduate/zhanli86/Workspace/src/projects/adb-paddy-rice-thai-binh/scripts/ndvi_ts_smoothn.m"
${ML} "maxNumCompThreads(16); run ${ML_CMD}"
