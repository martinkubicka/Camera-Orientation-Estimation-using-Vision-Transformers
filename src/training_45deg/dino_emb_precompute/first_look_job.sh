#!/bin/bash
#PBS -N dinov2_emb_preprocess
#PBS -l select=1:ncpus=1:mem=256gb:ngpus=1:gpu_mem=20gb:scratch_local=256gb
#PBS -l walltime=150:00:00
#PBS -q gpu_long

DATADIR=/storage/brno2/home/xkubic45/DP/src/training_45deg/dino_emb_precompute
cd $DATADIR || { echo "Failed to change directory to $DATADIR" >&2; exit 2; }
chmod +x setup.sh || { echo "Failed to change permissions" >&2; exit 3; }
./setup.sh || { echo "setup.sh failed" >&2; exit 4; }
python get_dinov2_emb_dataset.py \
  --src ../../training/dataset/out_train_first_look_45_augmented \
  --dst ../../training/dataset/out_train_first_look_45_augmented_embed \
  --num_rows 4 --num_cols 8 --fov_h 45 --fov_v 45 --tile_w 518 --tile_h 518 \
  --model_name dinov2_vitb14 --batch_size 32

