#!/usr/bin/env bash
set -euo pipefail

# 1) 사용할 사전학습 모델 리스트 (timm/torchvision 둘 다 인식되는 이름 권장)
models=(
  "mobilenetv1_100"
  "mobilenetv2_050"
  "mobilenetv2_100"
  "mobilenetv3_small_050"
  "mobilenetv3_small_075"
  "mobilenetv3_small_100"
  "mobilenetv3_large_100"
  "mobilenetv3_rw"
  "mobilenetv4_conv_small_050"
  "mobilenetv4_conv_small"
  "mobilenetv4_conv_medium"
  "mobilenetv4_conv_large"
  "mobilenetv4_hybrid_medium"
  "mobilenetv4_hybrid_large"
  "resnet18"
  "resnet34"
)

# 2) augmentation 모드 리스트 (0=off, 1~3=각각 다른 설정)
augments=(1 2)
augments=(0)
# 3) (선택) 공통 파라미터
EPOCHS=1
BATCH_SIZE=32
# CONFIG="config-fp32.yaml"   # 새 FP32 설정 파일

# 4) 실행 루프
for model in "${models[@]}"; do
  for aug in "${augments[@]}"; do
    echo "=============================================="
    echo "[START] Model: ${model} | Augment mode: ${aug}"
    echo "----------------------------------------------"

    python FP32-train.py \
      --model "${model}" \
      --augment "${aug}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      # --config "${CONFIG}"

    echo "[DONE ] Model: ${model} | Augment mode: ${aug}"
    echo "=============================================="
    echo
  done
done

# 실행 예시 (백그라운드 로그 저장):
# nohup ./run-fp32-train.sh > experiments_fp32.log 2>&1 &
