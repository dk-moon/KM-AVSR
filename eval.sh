#!/bin/bash

# -----------------------
# ⚙️ 실험 파라미터
# -----------------------
EXP_DIR="./exp"
EXP_NAME="my_eval"
MODALITY="audiovisual"        # audio, video, audiovisual 선택
BATCH_SIZE=16
DEVICE="gpu"                 # gpu 또는 cpu
NUM_DEVICES=1

# -----------------------
# ⚙️ backbone 설정
# -----------------------
PRETRAINED_MODEL_PATH="./exp/my_experiment/last.ckpt"  # 🔥 평가할 모델 체크포인트 경로 지정
AUDIOVISUAL_BACKBONE="conformer_av"
AUDIO_BACKBONE="conformer_audio"
VISUAL_BACKBONE="conformer_visual"

LABEL_FLAG=1               # label 출력 여부 (1: 출력, 0: 출력 안함)

# -----------------------
# ⚙️ Python 실행
# -----------------------
python eval.py \
    --exp_dir "$EXP_DIR" \
    --exp_name "$EXP_NAME" \
    --modality "$MODALITY" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --num_devices "$NUM_DEVICES" \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --audiovisual_backbone "$AUDIOVISUAL_BACKBONE" \
    --audio_backbone "$AUDIO_BACKBONE" \
    --visual_backbone "$VISUAL_BACKBONE" \
    --label_flag "$LABEL_FLAG"