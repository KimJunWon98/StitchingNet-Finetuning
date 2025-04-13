import argparse
import copy
import datetime
import os
import re
import time
from collections import OrderedDict
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import timm
import wandb
import yaml
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, classification_report
from timm.models._efficientnet_blocks import (
    DepthwiseSeparableConv,
    InvertedResidual,
)
from torch.ao.quantization import QConfig, prepare_qat, convert
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.quantization import (
    fuse_modules,
    get_default_qat_qconfig,
    get_default_qconfig,
    prepare,
)

from modules.data_loader import get_defect_data_loaders
from modules.dataset import count_labels
from modules.evaluate import test_best_model
from modules.seed import seed_everything
from modules.train import train_model
from modules.fuse_model import fuse_model
from custom_models.mobilenetv2_custom import mobilenet_v2_custom
from custom_models.mobilenetv3_custom import mobilenet_v3_custom
from custom_models.mobilenetv3_small_custom import mobilenet_v3_small_custom
from custom_models.timm_mobilnetv2_quantization import timm_mobilenet_v2_custom
from custom_models.timm_mobilnetv3_small_quantization import timm_mobilenet_v3_small_custom

from modules.fuse_model import fuse_model as generic_fuse
from custom_models.timm_fuse import load_fused_timm_model   

# ---------------------------------------------------------------------------
# Global quantization engine (overwritten in main from config)
# ---------------------------------------------------------------------------
QENGINE: str = "fbgemm"   # 기본값


def safe_fuse(model: nn.Module) -> None:
    """
    1) model.fuse_model() 메서드가 있으면 그것을 호출
    2) 없으면 generic_fuse(model) 실행
    """
    if hasattr(model, "fuse_model"):
        model.fuse_model()
    else:
        generic_fuse(model)


# ---------------------------------------------------------------------------
# Unified model‑factory
# ---------------------------------------------------------------------------
def create_model_by_name(
    model_name: str,
    num_classes: int,
    freeze: bool = False,
) -> nn.Module:


    # 1) timm_ prefix  →  timm 모델에 fuse 적용

    if model_name.startswith("timm_fuse_"):
        # strip leading tag  e.g. "timm_fuse_mobilenetv3_small_050"
        timm_name = model_name[len("timm_fuse_"):]      # "mobilenetv3_small_050"
        model = load_fused_timm_model(timm_name, pretrained=True,num_classes=num_classes)
        model.train()


    # 2) timm 모델의 가중치 torch vision에 load 하기.
    # 이름 매칭이 안되어서 순서대로 사이즈 맞으면 넣고 있는데, 30% 정도 밖에 load가 안된다.

    elif model_name.startswith("timm_mobilenetv2_"):
        # ex) "timm_mobilenetv2_050" → "050"
        width = re.search(r"_([0-9]{3})$", model_name).group(1)
        alpha = int(width) / 100.0                       # 50 → 0.50
        timm_variant = f"mobilenetv2_{width}"            # "mobilenetv2_050"

        model = timm_mobilenet_v2_custom(
            alpha=alpha,
            timm_pretrained=timm_variant,
            quantize=False,
        )

    elif model_name.startswith("timm_mobilenetv3_small_"):
        width = re.search(r"_([0-9]{3})$", model_name).group(1)
        alpha = int(width) / 100.0                       # 50 → 0.50

        model = timm_mobilenet_v3_small_custom(
            alpha=alpha,
            pretrained="timm",
        )


    # 3) torch vision 양자화 모델의 alpha 값 조절, 사전학습된 가중치는 존재하지 않음

    elif model_name.startswith("mobilenet_v2_custom"):
        alpha = float(model_name.split("_")[3]) / 100
        model = mobilenet_v2_custom(alpha=alpha, quantize=False, weights=None)

    elif model_name.startswith("mobilenet_v3_custom"):
        alpha = float(model_name.split("_")[3]) / 100
        model = mobilenet_v3_custom(alpha=alpha, quantize=False, weights=None)

    elif model_name.startswith("mobilenet_v3_small_custom"):
        alpha = float(model_name.split("_")[4]) / 100
        model = mobilenet_v3_small_custom(alpha=alpha, quantize=False, weights=None)


    # 3) torch vision 양자화 모델, 사전학습된 가중치가 존재. 가장 기본이 되는 모델

    elif model_name.startswith("mobilenet_v2"):
        model = torchvision.models.quantization.mobilenet_v2(pretrained=True)

    elif model_name.startswith("mobilenet_v3_large"):
        model = torchvision.models.quantization.mobilenet_v3_large(pretrained=True)
    else:
        raise ValueError(f"Unknown MODEL_NAME: {model_name}")

    # 분류기 변경
    model = replace_classifier(model, num_classes)

    # freeze
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model



def my_per_tensor_qat_qconfig() -> QConfig:
    """Return a per‑tensor QConfig for QAT."""

    act_observer = MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    )
    weight_observer = MovingAverageMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_affine
    )
    return QConfig(activation=act_observer, weight=weight_observer)


def evaluate_model_performance(model: nn.Module, dataloader, device):
    """Compute loss, accuracy, and mean single‑sample inference time."""

    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_loss, total_samples, correct = 0.0, 0, 0
    inference_times = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            start_time = time.time()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            inference_times.append(time.time() - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = correct / total_samples
    avg_loss = total_loss / total_samples
    mean_inference_time = np.mean(inference_times)

    return avg_loss, accuracy, mean_inference_time


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final classifier to match the number of classes."""

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential) and len(model.classifier) == 2:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_model_statistics(model: nn.Module):
    """Return parameter count and model size in MB (float32)."""

    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / (1024 ** 2)
    return param_count, model_size_mb


def apply_post_training_quantization(model: nn.Module, calibration_loader):
    """Apply PTQ using the given calibration loader."""

    model.eval()
    safe_fuse(model)

    # 전역 QENGINE 사용
    torch.backends.quantized.engine = QENGINE
    model.qconfig = get_default_qconfig(QENGINE)

    prepare(model, inplace=True)
    with torch.no_grad():
        for images, _, _ in calibration_loader:
            _ = model(images.to("cpu"))

    return convert(model, inplace=False)


# ---------------------------------------------------------------------------
# QAT Utilities
# ---------------------------------------------------------------------------

def convert_key(key: str) -> str | None:
    """Convert timm parameter keys to torchvision‑style keys (unused)."""

    # Mapping logic retained for potential future use
    if key.startswith("conv_stem"):
        return "features.0.0" + key[len("conv_stem"):]
    if key.startswith("bn1"):
        return "features.0.1" + key[len("bn1"):]
    if key.startswith("conv_head"):
        return "features.18.0" + key[len("conv_head"):]
    if key.startswith("bn2"):
        return "features.18.1" + key[len("bn2"):]
    if key.startswith("classifier"):
        return None

    match = re.match(r"blocks\.(\d)\.(\d)\.(.*)", key)
    if match:
        block, layer, rest = match.groups()
        tv_block = int(block) + 1
        mapping = {
            "conv_pw": "conv.0.0",
            "bn1": "conv.0.1",
            "conv_dw": "conv.1.0",
            "bn2": "conv.1.1",
            "conv_pwl": "conv.2",
            "bn3": "conv.3",
        }
        for k_timm, k_tv in mapping.items():
            if rest.startswith(k_timm):
                suffix = rest[len(k_timm):]
                return f"features.{tv_block}.conv.{layer}.{k_tv}{suffix}"
    return None


def apply_qat_training(
    model: nn.Module,
    trainloader,
    valloader,
    test_inference_loader,
    device,
    criterion,
    optimizer,
    scheduler=None,
    epochs: int = 5,
):
    """Perform QAT fine‑tuning and evaluate each epoch on INT8 model."""

    # Prepare model for QAT
    model.eval()
    model.fuse_model()
    model.train()

    # 전역 QENGINE 사용
    torch.backends.quantized.engine = QENGINE
    # model.qconfig = my_per_tensor_qat_qconfig()
    model.qconfig = get_default_qat_qconfig(QENGINE)
    prepare_qat(model, inplace=True)

    model.to(device)

    for epoch in range(epochs):
        # --------------------------- Training ---------------------------
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)

        # --------------------------- Validation -------------------------
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        if scheduler:
            scheduler.step()

        print(
            f"[QAT] Epoch [{epoch + 1}/{epochs}] | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # ---------------------- INT8 Evaluation ------------------------
        model_int8 = copy.deepcopy(model).to("cpu").eval()
        torch.backends.quantized.engine = QENGINE
        model_int8 = convert(model_int8, inplace=False)

        qat_loss, qat_acc, qat_time = evaluate_model_performance(
            model_int8, test_inference_loader, device="cpu"
        )

        print(
            f"[QAT Inference INT8] Loss={qat_loss:.4f}, Acc={qat_acc:.4f}, "
            f"Single‑sample Time={qat_time * 1000:.3f}ms\n"
        )

    return model


# ---------------------------------------------------------------------------
# Model‑Specific Helpers
# ---------------------------------------------------------------------------

def fuse_mobilenetv2_timm(model: nn.Module):
    """Fuse layers in a timm MobileNetV2 model for quantization."""

    # Stem
    fuse_modules(model, ["conv_stem", "bn1", "act1"], inplace=True)

    # Blocks
    for stage in model.blocks:
        for blk in stage:
            if isinstance(blk, DepthwiseSeparableConv):
                fuse_modules(blk, ["conv_dw", "bn1", "act1"], inplace=True)
                fuse_modules(blk, ["conv_pw", "bn2"], inplace=True)
            elif isinstance(blk, InvertedResidual):
                fuse_modules(blk, ["conv_pw", "bn1", "act1"], inplace=True)
                fuse_modules(blk, ["conv_dw", "bn2", "act2"], inplace=True)
                fuse_modules(blk, ["conv_pwl", "bn3"], inplace=True)

    # Head
    fuse_modules(model, ["conv_head", "bn2", "act2"], inplace=True)



# ---------------------------------------------------------------------------
# Main Training Script
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Quantization Training Pipeline")
    parser.add_argument("--model", type=str, help="모델 이름 (예: mobilenet_v2)")
    parser.add_argument("--augment", type=int, choices=[0,1,2,3], help="데이터 증강 Version (0, 1, 2, 3)", default=None)
    parser.add_argument("--epochs", type=int, help="학습 epoch 수", default=None)
    parser.add_argument("--batch-size", type=int, help="배치 크기", default=None)
    parser.add_argument("--patience", type=int, help="Early stopping patience", default=None)
    # parser.add_argument("--alpha", type=float, help="alpha 값", default=None)
    # parser.add_argument("--quant-mode", type=str, choices=["PTQ", "QAT"], help="양자화 모드", default=None)
    # parser.add_argument("--weights", type=str, help="사전 학습 가중치 경로", default=None)
    # parser.add_argument("--lr", type=float, help="학습률", default=None)

    return parser.parse_args()


def main():
    """Main entry point: PTQ + QAT training pipeline."""

    global QENGINE  # 전역 변수 수정

    args = parse_args()

    # --------------------------- Configuration ---------------------------
    with open("config-qat.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hparams = config["hyperparameters"]
    data_cfg = config["data"]
    train_cfg = config["training"]


    QENGINE = config.get("quantization", {}).get("qengine", QENGINE)
    torch.backends.quantized.engine = QENGINE
    print(f"[Config] Quantization Engine set to '{QENGINE}'")

    SEED = hparams["seed"]
    MODEL_NAME = hparams["model_name"]
    BATCH_SIZE = hparams["batch_size"]
    EPOCHS = hparams["epochs"]
    PATIENCE = hparams["patience"]
    MAX_CHECKPOINTS = hparams["max_checkpoints"]
    NUM_WORKERS = hparams["num_workers"]
    freeze_layers = train_cfg["freeze_layers"]
    use_augmentation = int(data_cfg.get("use_augmentation", 0))

    data_root = data_cfg["root"]
    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg["val_ratio"]

    project_name = train_cfg["project_name"]
    base_dir = train_cfg["checkpoint_base_dir"]

    base_lr = float(train_cfg.get("base_lr", 1e-4))
    head_lr = float(train_cfg.get("head_lr", 1e-3))
    use_scheduler = train_cfg.get("use_scheduler", False)
    T_max = train_cfg.get("T_max", EPOCHS)
    eta_min = float(train_cfg.get("eta_min", 1e-6))

    # --------------------------- Environment -----------------------------
    seed_everything(SEED)

    load_dotenv()

    if args.model:
        MODEL_NAME = args.model
    # if args.alpha is not None:
    #     alpha_value = args.alpha
    # if args.quant_mode:
    #     quant_mode = args.quant_mode  # PTQ 또는 QAT
    # else:
    #     quant_mode = None
    # if args.weights:
    #     weights_path = args.weights
    # else:
    #     weights_path = None
    # if args.lr is not None:
    #     base_lr = args.lr
    if args.augment is not None:
        use_augmentation = args.augment
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.patience is not None:
        PATIENCE = args.patience

    checkpoint_base_dir = os.path.join(base_dir, f"seed-{SEED}-version-{use_augmentation}")

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------- Data Loaders ----------------------------
    trainloader, valloader, testloader, classes = get_defect_data_loaders(
        data_root=data_root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_augmentation=use_augmentation,
    )

    # Inference loader (batch_size=1)
    test_inference_loader = torch.utils.data.DataLoader(
        testloader.dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------------------
    #  (A) PTQ Model (model1)
    # --------------------------------------------------------------------
    print(MODEL_NAME)

    # --------------------------- Model Init -----------------------------
    model1 = create_model_by_name(MODEL_NAME, num_classes=len(classes), freeze=freeze_layers)
    model1.to(device)

    # --------------------------- Optimizer ------------------------------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model1.parameters()), lr=base_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        if use_scheduler
        else None
    )

    # --------------------------- W&B Setup ------------------------------
    param_count, model_size_mb = get_model_statistics(model1)

    wandb.init(project=project_name, name=f"{MODEL_NAME}_PTQ")
    wandb.config.update(
        {
            "model_name": f"{MODEL_NAME}_PTQ",
            "batch_size": BATCH_SIZE,
            "use_augmentation": use_augmentation,
            "freeze_layers": freeze_layers,
            "seed": SEED,
            "qengine": QENGINE,
        }
    )
    wandb.log({"total_params": param_count, "model_size_MB": model_size_mb})

    # --------------------------- FP32 Training --------------------------
    _ = train_model(
        model=model1,
        criterion=criterion,
        optimizer=optimizer,
        trainloader=trainloader,
        valloader=valloader,
        layers=[],
        model_name=f"{MODEL_NAME}",
        num_epochs=EPOCHS,
        patience=PATIENCE,
        max_checkpoints=MAX_CHECKPOINTS,
        checkpoint_dir=checkpoint_base_dir,
        scheduler=scheduler,
    )

    # --------------------------- Load Best FP32 -------------------------
    best_model_path = os.path.join(checkpoint_base_dir, f"{MODEL_NAME}__best.pt")
    model1.load_state_dict(torch.load(best_model_path, map_location=device))
    model1.eval()

    # --------------------------- FP32 Test ------------------------------
    correct = total = total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, _ in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model1(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += criterion(outputs, labels).item() * labels.size(0)

    test_loss_fp32 = total_loss / total
    test_acc_fp32 = correct / total

    print(f"[FP32 - model1] Test Loss={test_loss_fp32:.4f}, Acc={test_acc_fp32:.4f}")
    wandb.log({"Test Loss (FP32)": test_loss_fp32, "Test Acc (FP32)": test_acc_fp32})

    # --------------------------- FP32 Inference -------------------------
    fp32_loss, fp32_acc, fp32_time = evaluate_model_performance(model1, test_inference_loader, device="cpu")
    print(
        f"[FP32 Inference - model1] Loss={fp32_loss:.4f}, Acc={fp32_acc:.4f}, "
        f"Single‑sample Time={fp32_time * 1000:.3f}ms"
    )
    wandb.log(
        {
            "FP32_Infer_Loss": fp32_loss,
            "FP32_Infer_Acc": fp32_acc,
            "FP32_Infer_Time(ms)": fp32_time * 1000,
        }
    )

    # --------------------------- PTQ -----------------------------------
    model1.to("cpu")
    ptq_model = apply_post_training_quantization(model1, valloader)

    ptq_loss, ptq_acc, ptq_time = evaluate_model_performance(ptq_model, test_inference_loader, device="cpu")
    print(
        f"[PTQ Inference] Loss={ptq_loss:.4f}, Acc={ptq_acc:.4f}, "
        f"Single‑sample Time={ptq_time * 1000:.3f}ms"
    )
    wandb.log(
        {
            "PTQ_Infer_Loss": ptq_loss,
            "PTQ_Infer_Acc": ptq_acc,
            "PTQ_Infer_Time(ms)": ptq_time * 1000,
        }
    )
    wandb.finish()

    # --------------------------------------------------------------------
    #  (B) QAT Model (model2)
    # --------------------------------------------------------------------
    model2 = create_model_by_name(MODEL_NAME, num_classes=len(classes), freeze=freeze_layers)
    model2.to(device)

    # --------------------------- W&B (QAT) ------------------------------
    wandb.init(project=project_name, name=f"{MODEL_NAME}_QAT")
    wandb.config.update(
        {
            "model_name": f"{MODEL_NAME}_QAT",
            "batch_size": BATCH_SIZE,
            "use_augmentation": use_augmentation,
            "freeze_layers": freeze_layers,
            "seed": SEED,
            "qengine": QENGINE,
        }
    )

    # Load best FP32 weights
    model2.load_state_dict(torch.load(best_model_path, map_location=device))

    # Quick FP32 evaluation
    model2.eval()
    correct = total = total_loss = 0.0
    with torch.no_grad():
        for inputs, labels, _ in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model2(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += criterion(outputs, labels).item() * labels.size(0)

    print(
        f"[FP32 - model2] Test Loss={total_loss / total:.4f}, Acc={correct / total:.4f}"
    )
    wandb.log(
        {
            "Test Loss (FP32)_model2": total_loss / total,
            "Test Acc (FP32)_model2": correct / total,
        }
    )

    # QAT fine‑tuning setup
    optimizer_qat = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=base_lr)
    scheduler_qat = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer_qat, T_max=5, eta_min=eta_min)
        if use_scheduler
        else None
    )

    # --------------------------- QAT Fine‑Tuning ------------------------
    model2_qat = apply_qat_training(
        model=model2,
        trainloader=trainloader,
        valloader=valloader,
        test_inference_loader=test_inference_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer_qat,
        scheduler=scheduler_qat,
        epochs=20,
    )

    # --------------------------- INT8 Conversion -----------------------
    model2_qat.eval().to("cpu")
    torch.backends.quantized.engine = QENGINE
    qat_int8_model = convert(model2_qat, inplace=False)

    qat_loss, qat_acc, qat_time = evaluate_model_performance(qat_int8_model, test_inference_loader, device="cpu")
    print(
        f"[QAT Inference] Loss={qat_loss:.4f}, Acc={qat_acc:.4f}, "
        f"Single‑sample Time={qat_time * 1000:.3f}ms"
    )
    wandb.log(
        {
            "QAT_Infer_Loss": qat_loss,
            "QAT_Infer_Acc": qat_acc,
            "QAT_Infer_Time(ms)": qat_time * 1000,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
