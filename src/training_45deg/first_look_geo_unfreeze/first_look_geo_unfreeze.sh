#!/bin/bash
#PBS -N first_look_geo_unfreeze_dp
#PBS -l select=1:ncpus=1:mem=256gb:ngpus=1:gpu_mem=44gb:scratch_local=256gb
#PBS -l walltime=300:00:00
#PBS -q gpu_long

DATADIR=/storage/brno2/home/xkubic45/DP/src/training_45deg/first_look_geo_unfreeze
cd $DATADIR || { echo "Failed to change directory to $DATADIR" >&2; exit 2; }
chmod +x setup.sh || { echo "Failed to change permissions" >&2; exit 3; }
./setup.sh || { echo "setup.sh failed" >&2; exit 4; }
python train.py
