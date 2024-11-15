#!/bin/bash

# 実験設定
EXP_DATE="1115"
EXP_ID="001"
DB_ID="1115_A"

# パスの設定
export UNET_CKPT_PATH="/root/data/checkpoints/mercari_unet/exp_1114_001/checkpoints/last.ckpt"
export FIRST_STAGE_CKPT_PATH="/root/data/checkpoints/mercari_firststage/exp_1113_001/checkpoints/last.ckpt"
export DB_PATH="/root/data/datasets/mercari/db_${DB_ID}/data.lmdb"

# ログディレクトリ
LOG_DIR="/root/data/checkpoints/mercari_clip/exp_${EXP_DATE}_${EXP_ID}"
mkdir -p ${LOG_DIR}

# 実験設定のログ
echo "Experiment Settings:" > ${LOG_DIR}/experiment_settings.txt
echo "Date: ${EXP_DATE}" >> ${LOG_DIR}/experiment_settings.txt
echo "ID: ${EXP_ID}" >> ${LOG_DIR}/experiment_settings.txt
echo "Database: ${DB_PATH}" >> ${LOG_DIR}/experiment_settings.txt
echo "UNET checkpoint: ${UNET_CKPT_PATH}" >> ${LOG_DIR}/experiment_settings.txt
echo "First stage checkpoint: ${FIRST_STAGE_CKPT_PATH}" >> ${LOG_DIR}/experiment_settings.txt

# 実験実行
cd /root/work/DisDiff
python main.py \
  --base configs/latent-diffusion/mercari-vq-4-16-dis.yaml \
  -t \
  --gpus 0, \
  -l ${LOG_DIR} \
  -n s0 \
  -s 0 \
  2>&1 | tee ${LOG_DIR}/training.log 