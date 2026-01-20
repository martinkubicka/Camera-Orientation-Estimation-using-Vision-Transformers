#!/bin/bash
#PBS -N main_first_look_final_big_dinov2small_finetune_hdf5
#PBS -l select=1:ncpus=1:mem=256gb:ngpus=1:gpu_mem=50gb:scratch_local=256gb
#PBS -l walltime=320:00:00
#PBS -q gpu_long

DATADIR=/storage/brno2/home/xkubic45/DP/src/training_45deg/main_first_look_final_big_dinov2small_finetune_hdf5
cd $DATADIR || { echo "Failed to change directory to $DATADIR" >&2; exit 2; }
chmod +x setup.sh || { echo "Failed to change permissions" >&2; exit 3; }
./setup.sh || { echo "setup.sh failed" >&2; exit 4; }
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
