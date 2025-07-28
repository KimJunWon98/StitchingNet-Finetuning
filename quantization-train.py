import argparse
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import yaml
from dotenv import load_dotenv

from torch.ao.quantization import QConfig, prepare_qat, convert
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.quantization import (
    fuse_modules,
    get_default_qat_qconfig,
    get_default_qconfig,
    prepare,
)

from modules.data_loader import get_defect_data_loaders
from modules.seed import seed_everything
from modules.train import train_model
from modules.fuse_model import fuse_model as generic_fuse, fuse_model
from custom_models.mobilenetv2_custom import mobilenet_v2_custom
from custom_models.mobilenetv3_custom import mobilenet_v3_custom
from custom_models.mobilenetv3_small_custom import mobilenet_v3_small_custom


# Global quantization engine 
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


def create_model_by_name(
    model_name: str,
    num_classes: int,
    freeze: bool = False,
) -> nn.Module:
    # 모델 이름에 따라 커스텀 또는 torchvision 모델 생성
    # alpha 값 등 하이퍼파라미터를 모델명에서 추출하여 적용

    if model_name.startswith("mobilenet_v2_custom"):
        alpha = float(model_name.split("_")[3]) / 100
        model = mobilenet_v2_custom(alpha=alpha, quantize=False, weights=None)

    elif model_name.startswith("mobilenet_v3_custom"):
        alpha = float(model_name.split("_")[3]) / 100
        model = mobilenet_v3_custom(alpha=alpha, quantize=False, weights=None)

    elif model_name.startswith("mobilenet_v3_small_custom"):
        alpha = float(model_name.split("_")[4]) / 100
        model = mobilenet_v3_small_custom(alpha=alpha, quantize=False, weights=None)

    elif model_name.startswith("mobilenet_v2"):
        model = torchvision.models.quantization.mobilenet_v2(pretrained=True)

    elif model_name.startswith("mobilenet_v3_large"):
        model = torchvision.models.quantization.mobilenet_v3_large(pretrained=True)
    else:
        raise ValueError(f"Unknown MODEL_NAME: {model_name}")

    # 분류기 레이어를 클래스 수에 맞게 교체
    model = replace_classifier(model, num_classes)

    # freeze 옵션이 True면 파라미터 업데이트 비활성화
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model


def my_per_tensor_qat_qconfig() -> QConfig:
    """QAT를 위한 per-tensor QConfig 반환"""
    act_observer = MovingAverageMinMaxObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    )
    weight_observer = MovingAverageMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_affine
    )
    return QConfig(activation=act_observer, weight=weight_observer)


def evaluate_model_performance(model: nn.Module, dataloader, device):
    """모델의 loss, accuracy, 평균 추론 시간(ms) 계산"""
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
    """모델의 마지막 분류기 레이어를 num_classes에 맞게 교체"""
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential) and len(model.classifier) == 2:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_model_statistics(model: nn.Module):
    """모델의 파라미터 개수와 float32 기준 모델 크기(MB) 반환"""
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / (1024 ** 2)
    return param_count, model_size_mb


def apply_post_training_quantization(model: nn.Module, calibration_loader):
    """PTQ(사후 양자화) 적용 및 변환된 모델 반환"""
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
    log_callback=None,
):
    """QAT(Qualization Aware Training) 파인튜닝 및 valloader/testloader에 대한
       Fake-quant & INT8 평가를 에폭별로 수행"""
    # QAT 준비
    model.eval()
    model.fuse_model()
    model.train()

    torch.backends.quantized.engine = QENGINE
    model.qconfig = get_default_qat_qconfig(QENGINE)
    prepare_qat(model, inplace=True)
    model.to(device)

    for epoch in range(epochs):
        # 1) 학습 루프
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

        # 2) Fake-quant 모델 검증 (GPU) on valloader & testloader
        model.eval()
        with torch.no_grad():
            val_loss_fq, val_acc_fq, _ = evaluate_model_performance(
                model, valloader, device
            )
            test_loss_fq, test_acc_fq, _ = evaluate_model_performance(
                model, test_inference_loader, device
            )

        # 스케줄러 스텝 (validation 후)
        if scheduler:
            scheduler.step()

        print(
            f"[QAT FakeQuant] Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss/Acc: {val_loss_fq:.4f}/{val_acc_fq:.4f} | "
            f"Test Loss/Acc: {test_loss_fq:.4f}/{test_acc_fq:.4f}"
        )

        # 3) 실제 INT8 모델 변환 & 평가 (CPU)
        model_int8 = copy.deepcopy(model).cpu().eval()
        torch.backends.quantized.engine = QENGINE
        model_int8 = convert(model_int8, inplace=False) # CPU에서 INT8 변환 (GPU에서 변환하면 오류 발생)

        with torch.no_grad():
            val_loss_int8, val_acc_int8, val_time_int8 = evaluate_model_performance(
                model_int8, valloader, device="cpu"
            )
            test_loss_int8, test_acc_int8, test_time_int8 = evaluate_model_performance(
                model_int8, test_inference_loader, device="cpu"
            )

        print(
            f"[QAT INT8] Epoch [{epoch+1}/{epochs}] | "
            f"Val Loss/Acc: {val_loss_int8:.4f}/{val_acc_int8:.4f} | "
            f"Val Time: {val_time_int8*1000:.3f}ms | "
            f"Test Loss/Acc: {test_loss_int8:.4f}/{test_acc_int8:.4f} | "
            f"Test Time: {test_time_int8*1000:.3f}ms\n"
        )

        # 콜백으로 로그 전달
        if log_callback is not None:
            log_callback(
                epoch+1,
                val_loss_fq, val_acc_fq, test_loss_fq, test_acc_fq,
                val_loss_int8, val_acc_int8, val_time_int8,
                test_loss_int8, test_acc_int8, test_time_int8
            )

    return model



def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Quantization Training Pipeline")
    parser.add_argument("--model", type=str, help="모델 이름 (예: mobilenet_v2)")
    parser.add_argument("--augment", type=int, choices=[0,1,2,3], help="데이터 증강 Version (0, 1, 2, 3)", default=None)
    parser.add_argument("--epochs", type=int, help="학습 epoch 수", default=None)
    parser.add_argument("--batch-size", type=int, help="배치 크기", default=None)
    parser.add_argument("--patience", type=int, help="Early stopping patience", default=None)
    parser.add_argument("--config", type=str, help="설정 파일 경로", default=None)
    # parser.add_argument("--alpha", type=float, help="alpha 값", default=None)
    # parser.add_argument("--quant-mode", type=str, choices=["PTQ", "QAT"], help="양자화 모드", default=None)
    # parser.add_argument("--weights", type=str, help="사전 학습 가중치 경로", default=None)
    # parser.add_argument("--lr", type=float, help="학습률", default=None)

    return parser.parse_args()


def main():
    """PTQ + QAT 전체 학습 파이프라인 메인 함수"""

    global QENGINE  # 전역 변수 수정

    args = parse_args()

    # --------------------------- 설정 파일 로드 ---------------------------
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    hparams = config["hyperparameters"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # 양자화 엔진 설정
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

    # --------------------------- 환경 설정 -----------------------------
    seed_everything(SEED)

    load_dotenv()

    # 명령행 인자 적용
    if args.model:
        MODEL_NAME = args.model
    if args.augment is not None:
        use_augmentation = args.augment
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.patience is not None:
        PATIENCE = args.patience

    checkpoint_base_dir = os.path.join(base_dir, f"seed-{SEED}-version-{use_augmentation}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------- 데이터 로더 ----------------------------
    trainloader, valloader, testloader, classes = get_defect_data_loaders(
        data_root=data_root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_augmentation=use_augmentation,
    )

    # 추론용 데이터로더 (batch_size=1)
    test_inference_loader = torch.utils.data.DataLoader(
        testloader.dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------------------
    #  (A) PTQ 모델 (model1)
    # --------------------------------------------------------------------
    print(MODEL_NAME)

    # --------------------------- 모델 생성 -----------------------------
    model1 = create_model_by_name(MODEL_NAME, num_classes=len(classes), freeze=freeze_layers)
    model1.to(device)

    # --------------------------- 옵티마이저 ------------------------------
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model1.parameters()), lr=base_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        if use_scheduler
        else None
    )

    # --------------------------- CSV 로깅 준비 ------------------------------
    logs = []
    param_count, model_size_mb = get_model_statistics(model1)
    logs.append({
        "stage": "init",
        "model_name": f"{MODEL_NAME}_PTQ",
        "batch_size": BATCH_SIZE,
        "use_augmentation": use_augmentation,
        "freeze_layers": freeze_layers,
        "seed": SEED,
        "qengine": QENGINE,
        "total_params": param_count,
        "model_size_MB": model_size_mb
    })

    # --------------------------- FP32 학습 --------------------------
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

    # --------------------------- Best FP32 로드 -------------------------
    best_model_path = os.path.join(checkpoint_base_dir, f"{MODEL_NAME}__best.pt")
    model1.load_state_dict(torch.load(best_model_path, map_location=device))
    model1.eval()

    # --------------------------- FP32 테스트 ------------------------------
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
    logs.append({
        "stage": "FP32_test",
        "loss": test_loss_fp32,
        "acc": test_acc_fp32
    })

    # --------------------------- FP32 추론 -------------------------
    fp32_loss, fp32_acc, fp32_time = evaluate_model_performance(model1, test_inference_loader, device="cpu")
    print(
        f"[FP32 Inference - model1] Loss={fp32_loss:.4f}, Acc={fp32_acc:.4f}, "
        f"Single‑sample Time={fp32_time * 1000:.3f}ms"
    )
    logs.append({
        "stage": "FP32_infer",
        "loss": fp32_loss,
        "acc": fp32_acc,
        "time_ms": fp32_time * 1000
    })

    # --------------------------- PTQ -----------------------------------
    model1.to("cpu")
    ptq_model = apply_post_training_quantization(model1, valloader)

    ptq_loss, ptq_acc, ptq_time = evaluate_model_performance(ptq_model, test_inference_loader, device="cpu")
    print(
        f"[PTQ Inference] Loss={ptq_loss:.4f}, Acc={ptq_acc:.4f}, "
        f"Single‑sample Time={ptq_time * 1000:.3f}ms"
    )
    logs.append({
        "stage": "PTQ_infer",
        "loss": ptq_loss,
        "acc": ptq_acc,
        "time_ms": ptq_time * 1000
    })

    # --------------------------------------------------------------------
    #  (B) QAT 모델 (model2)
    # --------------------------------------------------------------------
    model2 = create_model_by_name(MODEL_NAME, num_classes=len(classes), freeze=freeze_layers)
    model2.to(device)



    # Best FP32 가중치 로드
    model2.load_state_dict(torch.load(best_model_path, map_location=device))

    # FP32 평가
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
    logs.append({
        "stage": "FP32_test_model2",
        "loss": total_loss / total,
        "acc": correct / total
    })

    # QAT 파인튜닝 설정
    optimizer_qat = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=base_lr)
    scheduler_qat = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer_qat, T_max=5, eta_min=eta_min)
        if use_scheduler
        else None
    )

    # --------------------------- QAT 파인튜닝 ------------------------
    qat_train_logs = []
    def qat_log_callback(epoch, fq_val_loss, fq_val_acc, fq_test_loss, fq_test_acc, int8_val_loss, int8_val_acc, int8_val_time, int8_test_loss, int8_test_acc, int8_test_time):
        qat_train_logs.append({
            "stage": "QAT_epoch",
            "epoch": epoch,
            "fq_val_loss": fq_val_loss,
            "fq_val_acc": fq_val_acc,
            "fq_test_loss": fq_test_loss,
            "fq_test_acc": fq_test_acc,
            "int8_val_loss": int8_val_loss,
            "int8_val_acc": int8_val_acc,
            "int8_val_time_ms": int8_val_time * 1000,
            "int8_test_loss": int8_test_loss,
            "int8_test_acc": int8_test_acc,
            "int8_test_time_ms": int8_test_time * 1000
        })

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
        log_callback=qat_log_callback
    )

    # --------------------------- INT8 변환 -----------------------
    model2_qat.eval().to("cpu")
    torch.backends.quantized.engine = QENGINE
    qat_int8_model = convert(model2_qat, inplace=False)

    qat_loss, qat_acc, qat_time = evaluate_model_performance(qat_int8_model, test_inference_loader, device="cpu")
    print(
        f"[QAT Inference] Loss={qat_loss:.4f}, Acc={qat_acc:.4f}, "
        f"Single‑sample Time={qat_time * 1000:.3f}ms"
    )
    logs.append({
        "stage": "QAT_infer",
        "loss": qat_loss,
        "acc": qat_acc,
        "time_ms": qat_time * 1000
    })

    # QAT 학습 로그 합치기
    logs.extend(qat_train_logs)

    # CSV로 저장
    df = pd.DataFrame(logs)
    df.to_csv(f"{MODEL_NAME}_quantization_log.csv", index=False)


if __name__ == "__main__":
    main()
