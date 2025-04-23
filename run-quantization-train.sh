#!/usr/bin/env bash
set -euo pipefail

# 1) 사용할 모델 리스트
models=(
  "mobilenet_v3_large"
  "mobilenet_v3_custom_075"
  "mobilenet_v3_custom_050"
  "mobilenet_v3_custom_025"
  "mobilenet_v2"
  "mobilenet_v2_custom_075"
  "mobilenet_v2_custom_050"
  "mobilenet_v2_custom_025"
  "mobilenet_v3_small_custom_100"
  "mobilenet_v3_small_custom_075"
  "mobilenet_v3_small_custom_050"
  "mobilenet_v3_small_custom_025"
)

# 2) augmentation 모드 리스트 (0=off, 1~3=각각 다른 설정)
augments=(0 1 2 3)

# # 3) 공통 파라미터
# EPOCHS=30
# BATCH_SIZE=64
# CONFIG="config-qat.yaml"

# 4) 실행 루프
for model in "${models[@]}"; do
  for aug in "${augments[@]}"; do
    echo "=============================================="
    echo "[START] Model: ${model} | Augment mode: ${aug}"
    echo "----------------------------------------------"

    python /workspace/hojeon/git-repo/StitchingNet-Finetuning/quantization-train.py \
      --model "${model}" \
      --augment "${aug}" \
    #   --epochs "${EPOCHS}" \
    #   --batch_size "${BATCH_SIZE}" \
    #   --config "${CONFIG}"

    echo "[DONE ] Model: ${model} | Augment mode: ${aug}"
    echo "=============================================="
    echo
  done
done
