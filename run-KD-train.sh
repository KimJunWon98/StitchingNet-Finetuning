#!/usr/bin/env bash
set -euo pipefail

# ===========================
# StitchingNet 양자화 실험 스크립트
# ===========================

# 1) 사용할 모델 리스트
models=(
  # "timm_mobilenetv4_conv_large"
  # "mobilenet_v3_large"
  # "mobilenet_v3_custom_075"
  "mobilenet_v3_custom_050"
  # "mobilenet_v3_custom_035"
  # "mobilenet_v3_custom_025"
  # "mobilenet_v2"
  # "mobilenet_v2_custom_075"
  # "mobilenet_v2_custom_050"
  # "mobilenet_v2_custom_035"
  # "mobilenet_v2_custom_025"
  # "mobilenet_v3_small_custom_100"
  # "mobilenet_v3_small_custom_075"
  # "mobilenet_v3_small_custom_050"
  # "mobilenet_v3_small_custom_035"
  # "mobilenet_v3_small_custom_025"
)

# 2) 증강(augmentation) 모드 리스트 (0=off, 1~3=각각 다른 증강 설정) augments=(0 1 2 3)
augments=(2)

# 3) 공통 파라미터 (필요시 주석 해제하여 사용)
# EPOCHS=30
# BATCH_SIZE=64
CONFIG="config-quantization.yaml"

# 4) 실험 실행 루프
for model in "${models[@]}"; do
  for aug in "${augments[@]}"; do
    echo "=============================================="
    echo "[START] Model: ${model} | Augment mode: ${aug}"
    echo "----------------------------------------------"

    python KD-train.py \
      --model "${model}" \
      --augment "${aug}" \
      --config "${CONFIG}" \
      # --epochs "${EPOCHS}" \
      # --batch-size "${BATCH_SIZE}" 


    echo "[DONE ] Model: ${model} | Augment mode: ${aug}"
    echo "=============================================="
    echo
  done
done

# 백그라운드 실행 예시:
# nohup ./run-quantization-train.sh > experiments.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup ./run-quantization-train.sh > experiments.log 2>&1 &
