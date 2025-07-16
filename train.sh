#!/bin/bash

# -----------------------
# ⚙️ 실험 파라미터
# -----------------------
EXP_DIR="./exp"
EXP_NAME="my_experiment"
MODALITY="audiovisual"        # audio, video, audiovisual 선택
BATCH_SIZE=4
MAX_EPOCHS=11
DEVICE="cpu"                 # gpu 또는 cpu
NUM_DEVICES=1

# -----------------------
# ⚙️ Dataset 경로
# -----------------------
DATASET_ROOT="/Users/dkmoon/Desktop/WorkSpace/DaeKyoCNS/LipReading/Code/KM-AVSR/dataset"
TRAIN_FILE="/Users/dkmoon/Desktop/WorkSpace/DaeKyoCNS/LipReading/Code/KM-AVSR/dataset/train/train.csv"
VAL_FILE="/Users/dkmoon/Desktop/WorkSpace/DaeKyoCNS/LipReading/Code/KM-AVSR/dataset/valid/valid.csv"

MOUTH_DIR_TRAIN="/Users/dkmoon/Desktop/WorkSpace/DaeKyoCNS/LipReading/Code/KM-AVSR/dataset/train/Video"
WAV_DIR_TRAIN="/Users/dkmoon/Desktop/WorkSpace/DaeKyoCNS/LipReading/Code/KM-AVSR/dataset/train/Audio"
MOUTH_DIR_VALID="/Users/dkmoon/Desktop/WorkSpace/DaeKyoCNS/LipReading/Code/KM-AVSR/dataset/valid/Video"
WAV_DIR_VALID="/Users/dkmoon/Desktop/WorkSpace/DaeKyoCNS/LipReading/Code/KM-AVSR/dataset/valid/Audio"

# -----------------------
# ⚙️ backbone & optimizer 설정
# -----------------------
PRETRAINED_MODEL_PATH=""    # 사전학습 모델 경로 (없으면 랜덤 초기화)
TRANSFER_FRONTEND="--transfer_frontend"  # 사용하지 않으면 주석 처리
TRANSFER_ENCODER=""        # 사용하려면 "--transfer_encoder" 지정

AUDIOVISUAL_BACKBONE="conformer_av"
AUDIO_BACKBONE="conformer_audio"
VISUAL_BACKBONE="conformer_visual"

LR=0.001
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=5
LABEL_FLAG=0               # label 출력 여부 (1 or 0)

# -----------------------
# ⚙️ Trainer 고급 설정
# -----------------------
PRECISION=32
NUM_NODES=1
SYNC_BATCHNORM="--sync_batchnorm"  # 사용하지 않으면 주석 처리
NUM_SANITY_VAL_STEPS=0
ACCUMULATE_GRAD_BATCHES=1
GRADIENT_CLIP_VAL=5.0

# -----------------------
# ⚙️ Python 실행
# -----------------------
python train.py \
    --exp_dir "$EXP_DIR" \
    --exp_name "$EXP_NAME" \
    --modality "$MODALITY" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --device "$DEVICE" \
    --num_devices "$NUM_DEVICES" \
    --dataset_root "$DATASET_ROOT" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --mouth_dir_train "$MOUTH_DIR_TRAIN" \
    --wav_dir_train "$WAV_DIR_TRAIN" \
    --mouth_dir_valid "$MOUTH_DIR_VALID" \
    --wav_dir_valid "$WAV_DIR_VALID" \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    $TRANSFER_FRONTEND \
    $TRANSFER_ENCODER \
    --audiovisual_backbone "$AUDIOVISUAL_BACKBONE" \
    --audio_backbone "$AUDIO_BACKBONE" \
    --visual_backbone "$VISUAL_BACKBONE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --label_flag "$LABEL_FLAG" \
    --precision "$PRECISION" \
    --num_nodes "$NUM_NODES" \
    $SYNC_BATCHNORM \
    --num_sanity_val_steps "$NUM_SANITY_VAL_STEPS" \
    --accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES" \
    --gradient_clip_val "$GRADIENT_CLIP_VAL" \