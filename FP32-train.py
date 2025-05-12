"""train_fp32_only.py - FP32‑only training pipeline (timm pretrained models)
=====================================================================

* Trains and evaluates an image‑classification model in full‑precision (FP32).
* **No quantization, fusion, or custom architectures** – only models available
  in the `timm` model zoo (and optionally `torchvision.models`).
* Expected project directory already contains the helper modules that were
  used in the original repo (data loading, training loop, etc.).

Usage (minimal):
----------------
```bash
python train_fp32_only.py --model efficientnet_b0  # any timm model name
```

With overrides:
```bash
python train_fp32_only.py \
  --model mobilenetv3_small_050 \
  --augment 1 \
  --epochs 30 \
  --batch-size 64
```

Configuration is read from `config-fp32.yaml` by default. Command‑line
arguments override YAML values.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import wandb
import yaml
from dotenv import load_dotenv

# ---- local helper modules (must exist in the repo) -------------------
from modules.data_loader import get_defect_data_loaders
from modules.train import train_model  # generic training loop used earlier
from modules.seed import seed_everything

# ---------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------

def create_model_by_name(model_name: str, num_classes: int) -> nn.Module:
    """Return a *pretrained* model from timm with adjusted head."""

    try:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )
    except Exception as e:
        raise ValueError(
            f"Model '{model_name}' not found in timm or failed to load: {e}"
        )

    return model


def evaluate_model_performance(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
) -> Tuple[float, float, float]:
    """Compute loss, accuracy, and mean single‑sample inference time."""

    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_loss, total_samples, correct = 0.0, 0, 0
    inference_times: list[float] = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            start_time = time.time()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            inference_times.append(time.time() - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total_samples
    avg_loss = total_loss / total_samples
    mean_inference_time = float(np.mean(inference_times)) if inference_times else 0.0

    return avg_loss, accuracy, mean_inference_time


def get_model_statistics(model: nn.Module) -> Tuple[int, float]:
    """Return parameter count and model size in MB (FP32)."""

    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / (1024**2)  # float32 = 4 bytes
    return param_count, model_size_mb


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FP32 Training Pipeline (timm models)")
    parser.add_argument("--model", type=str, required=True, help="timm model name (e.g. efficientnet_b0)")
    parser.add_argument("--augment", type=int, choices=[0, 1, 2, 3], help="data‑augmentation version", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    parser.add_argument("--patience", type=int, default=None, help="early stopping patience (val loss)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --------------------------- Configuration ------------------------
    CONFIG_PATH = "config-fp32.yaml"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file '{CONFIG_PATH}' not found.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hp = config["hyperparameters"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # YAML defaults ----------------------------------------------------
    SEED = hp["seed"]
    MODEL_NAME = args.model or hp["model_name"]
    BATCH_SIZE = args.batch_size or hp["batch_size"]
    EPOCHS = args.epochs or hp["epochs"]
    PATIENCE = args.patience or hp["patience"]
    NUM_WORKERS = hp["num_workers"]
    AUG_VERSION = args.augment if args.augment is not None else int(data_cfg.get("use_augmentation", 0))

    TRAIN_RATIO = data_cfg["train_ratio"]
    VAL_RATIO = data_cfg["val_ratio"]
    DATA_ROOT = data_cfg["root"]

    PROJECT_NAME = train_cfg["project_name"]
    BASE_DIR = train_cfg["checkpoint_base_dir"]

    BASE_LR = float(train_cfg.get("base_lr", 1e-4))
    USE_SCHEDULER = train_cfg.get("use_scheduler", False)
    T_MAX = train_cfg.get("T_max", EPOCHS)
    ETA_MIN = float(train_cfg.get("eta_min", 1e-6))
    FREEZE_LAYERS = bool(train_cfg.get("freeze_layers", False))
    MAX_CHECKPOINTS = hp.get("max_checkpoints", 5)

    # --------------------------- Environment -------------------------
    seed_everything(SEED)
    load_dotenv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------- Data Loaders ------------------------
    trainloader, valloader, testloader, classes = get_defect_data_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        use_augmentation=AUG_VERSION,
    )

    # Smaller loader for single‑sample inference timing
    inference_loader = torch.utils.data.DataLoader(
        testloader.dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    # --------------------------- Model ------------------------------
    model = create_model_by_name(MODEL_NAME, num_classes=len(classes))
    if FREEZE_LAYERS:
        for p in model.parameters():
            p.requires_grad = False
    model.to(device)

    # --------------------------- Optimizer --------------------------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)
        if USE_SCHEDULER
        else None
    )
    criterion = nn.CrossEntropyLoss()

    # --------------------------- W&B -------------------------------
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=PROJECT_NAME, name=f"{MODEL_NAME}_FP32")

    param_count, model_size_mb = get_model_statistics(model)
    wandb.config.update(
        {
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "augmentation": AUG_VERSION,
            "seed": SEED,
            "freeze_layers": FREEZE_LAYERS,
            "total_params": param_count,
            "model_size_mb": model_size_mb,
        }
    )

    # --------------------------- Training ---------------------------
    checkpoint_dir = os.path.join(BASE_DIR, f"seed-{SEED}-aug-{AUG_VERSION}")
    best_val_loss = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        trainloader=trainloader,
        valloader=valloader,
        layers=[],
        model_name=MODEL_NAME,
        num_epochs=EPOCHS,
        patience=PATIENCE,
        max_checkpoints=MAX_CHECKPOINTS,
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler,
    )

    # --------------------------- Load Best Weights ------------------
    best_model_path = os.path.join(checkpoint_dir, f"{MODEL_NAME}__best.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # --------------------------- Test Evaluation --------------------
    correct = total = total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, _ in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += criterion(outputs, labels).item() * labels.size(0)

    test_loss = total_loss / total
    test_acc = correct / total

    print(f"[FP32] Test Loss={test_loss:.4f}, Acc={test_acc:.4f}")
    wandb.log({"Test Loss": test_loss, "Test Acc": test_acc})

    # --------------------------- Inference Timing -------------------
    inf_loss, inf_acc, inf_time = evaluate_model_performance(
        model, inference_loader, device=device
    )
    print(
        f"[FP32 Inference] Loss={inf_loss:.4f}, Acc={inf_acc:.4f}, "
        f"Single‑sample Time={inf_time * 1000:.3f} ms"
    )
    wandb.log(
        {
            "Infer Loss": inf_loss,
            "Infer Acc": inf_acc,
            "Infer Time (ms)": inf_time * 1000,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
