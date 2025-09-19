import argparse
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import timm
import wandb
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
    
    # timm 모델
    if model_name.startswith("timm_"):
        timm_model_name = model_name.replace("timm_", "")
        try:
            model = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=num_classes,
            )
        except Exception as e:
            raise ValueError(
                f"Model '{model_name}' not found in timm or failed to load: {e}"
            )
    
    
    # 모델 이름에 따라 커스텀 또는 torchvision 모델 생성
    # alpha 값 등 하이퍼파라미터를 모델명에서 추출하여 적용
    else:
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
    testloader,
    test_inference_loader,
    device,
    criterion,
    optimizer,
    scheduler=None,
    epochs: int = 5,
    checkpoint_base_dir: str = "./checkpoints",
    model_name: str = "model",
):
    """
    QAT(Quantization Aware Training) 파인튜닝 수행 함수
    - 매 epoch마다 FakeQuant(GPU) 및 INT8(CPU) 평가
    - Val Loss가 개선될 때마다 모델 state_dict를 저장
    저장 경로: {checkpoint_base_dir}/qat/{model_name}_epoch{n}_best.pt
    """
    # ── 1. QAT 전용 체크포인트 디렉토리 생성 ──
    qat_dir = os.path.join(checkpoint_base_dir, "qat")
    os.makedirs(qat_dir, exist_ok=True)


    # ── 1.1) Pre-QAT PTQ 변환 & CPU 평가 ──
    ptq_model = copy.deepcopy(model).cpu().eval()
    torch.backends.quantized.engine = QENGINE
    ptq_model = apply_post_training_quantization(ptq_model, trainloader)

    with torch.no_grad():
        val_loss_ptq, val_acc_ptq, val_time_ptq = evaluate_model_performance(
            ptq_model, valloader, device="cpu"
        )
        test_loss_ptq, test_acc_ptq, test_time_ptq = evaluate_model_performance(
            ptq_model, test_inference_loader, device="cpu"
        )

    print(
        f"[Pre-QAT PTQ] Val Loss/Acc: {val_loss_ptq:.4f}/{val_acc_ptq:.4f} | "
        f"Val Time: {val_time_ptq*1000:.3f}ms | "
        f"Test Loss/Acc: {test_loss_ptq:.4f}/{test_acc_ptq:.4f} | "
        f"Test Time: {test_time_ptq*1000:.3f}ms\n"
    )


    # ── 2. 모델 QAT 준비 단계 ──
    model.eval()              # 평가 모드로 전환 (fuse 전 상태)
    model.fuse_model()        # 모델 내 fuse 가능한 레이어 결합
    model.train()             # 다시 학습 모드로 전환
    torch.backends.quantized.engine = QENGINE  # 양자화 엔진 설정
    model.qconfig = get_default_qat_qconfig(QENGINE)  # 기본 QAT QConfig 적용
    prepare_qat(model, inplace=True)  # QAT 준비: FakeQuant 모듈 삽입
    model.to(device)          # 지정된 디바이스로 이동

    # Val Loss 최소값 초기화
    best_val_loss = float('inf')

    # ── 3. Epoch 루프 ──
    for epoch in range(1, epochs + 1):
        # 3.1) 학습 단계
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

        # 3.2) FakeQuant 평가 (GPU)
        model.eval()
        with torch.no_grad():
            val_loss_fq, val_acc_fq, _ = evaluate_model_performance(
                model, valloader, device
            )
            # test_loss_fq, test_acc_fq, _ = evaluate_model_performance(
            #     model, test_inference_loader, device
            # )
            test_loss_fq, test_acc_fq, _ = evaluate_model_performance(
                model, testloader, device
            )

        # # 3.3) FakeQuant Val Loss 개선 시 체크포인트 저장
        # if val_loss_fq < best_val_loss:
        #     best_val_loss = val_loss_fq
        #     save_path = os.path.join(
        #         qat_dir,
        #         f"{model_name}_epoch{epoch}_best.pt"
        #     )
        #     torch.save(model.state_dict(), save_path)
        #     print(f">>> [QAT Checkpoint] Epoch {epoch}, Val Loss={best_val_loss:.4f} → saved: {save_path}")


        # 3.3) 체크포인트 모두 저장
        save_path = os.path.join(
            qat_dir,
            f"{model_name}_epoch{epoch}_best.pt"
        )
        torch.save(model.state_dict(), save_path)
        print(f">>> [QAT Checkpoint] Epoch {epoch}, Val Loss={val_loss_fq:.4f} → saved: {save_path}")

        # 3.4) 스케줄러 스텝 (validation 후)
        if scheduler:
            scheduler.step()

        # 3.5) FakeQuant 결과 출력
        print(
            f"[QAT FakeQuant] Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss/Acc: {val_loss_fq:.4f}/{val_acc_fq:.4f} | "
            f"Test Loss/Acc: {test_loss_fq:.4f}/{test_acc_fq:.4f}"
        )

        # 3.6) INT8 변환 후 CPU에서 평가
        model_int8 = copy.deepcopy(model).cpu().eval()
        torch.backends.quantized.engine = QENGINE

        # 1) PTQ용 설정: qconfig 지정
        model_int8.qconfig = get_default_qconfig(QENGINE)

        # 2) Observer 삽입
        prepare(model_int8, inplace=True)

        # 3) 캘리브레이션: valloader 한 바퀴
        with torch.no_grad():
            for inputs, _, _ in trainloader:   
                model_int8(inputs)

        # 4) 실제 INT8 변환
        model_int8 = convert(model_int8, inplace=False)

        with torch.no_grad():
            val_loss_int8, val_acc_int8, val_time_int8 = evaluate_model_performance(
                model_int8, valloader, device="cpu"
            )
            test_loss_int8, test_acc_int8, test_time_int8 = evaluate_model_performance(
                model_int8, test_inference_loader, device="cpu"
            )

        # 3.7) INT8 결과 출력
        print(
            f"[QAT INT8] Epoch [{epoch}/{epochs}] | "
            f"Val Loss/Acc: {val_loss_int8:.4f}/{val_acc_int8:.4f} | "
            f"Val Time: {val_time_int8*1000:.3f}ms | "
            f"Test Loss/Acc: {test_loss_int8:.4f}/{test_acc_int8:.4f} | "
            f"Test Time: {test_time_int8*1000:.3f}ms\n"
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
    config_path = args.config if args.config else "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --------------------------- Hyperparameters ---------------------------
    hparams        = config["hyperparameters"]
    data_cfg       = config["data"]
    train_cfg      = config["training"]

    # --------------------------- Quantization -----------------------------
    QENGINE = config.get("quantization", {}).get("qengine", QENGINE)
    torch.backends.quantized.engine = QENGINE
    print(f"[Config] Quantization Engine set to '{QENGINE}'")

    # --------------------------- 기본 설정 -----------------------------
    SEED            = hparams["seed"]
    MODEL_NAME      = hparams["model_name"]
    BATCH_SIZE      = hparams["batch_size"]
    EPOCHS          = hparams["epochs"]
    PATIENCE        = hparams["patience"]
    MAX_CHECKPOINTS = hparams["max_checkpoints"]
    NUM_WORKERS     = hparams["num_workers"]
    freeze_layers   = train_cfg["freeze_layers"]
    use_augmentation= int(data_cfg.get("use_augmentation", 0))

    dataset_name    = dataset_name = data_cfg.get("dataset_name", "StitchingNet")
    img_size_map   = data_cfg.get("img_size_map", {})
    img_size       = tuple(img_size_map.get(dataset_name, [224, 224]))  # 기본값 224,224
    data_root       = data_cfg["root"]
    train_ratio     = data_cfg["train_ratio"]
    val_ratio       = data_cfg["val_ratio"]

    TRAINING_FLAG = train_cfg.get("training_flag", True)
    project_name    = train_cfg["project_name"]
    base_dir        = train_cfg["checkpoint_base_dir"]

    base_lr         = float(train_cfg.get("base_lr", 1e-4))
    head_lr         = float(train_cfg.get("head_lr", 1e-3))
    use_scheduler   = train_cfg.get("use_scheduler", False)
    T_max           = train_cfg.get("T_max", EPOCHS)
    eta_min         = float(train_cfg.get("eta_min", 1e-6))

    # --------------------------- QAT 설정 -----------------------------
    USE_QAT             = config.get("qat", {}).get("use_qat", False)
    BEST_MODEL_METRIC   = config.get("qat", {}).get("best_model_metric", "val_loss_fq")
    QAT_EPOCHS          = int(config.get("qat", {}).get("qat_epochs", EPOCHS))
    QAT_LR              = float(config.get("qat", {}).get("qat_lr", base_lr))
    QAT_HEAD_LR         = config.get("qat", {}).get("qat_head_lr", head_lr)
    QAT_USE_SCHEDULER   = config.get("qat", {}).get("qat_use_scheduler", use_scheduler)
    QAT_T_MAX           = int(config.get("qat", {}).get("qat_T_max", QAT_EPOCHS))
    QAT_ETA_MIN         = float(config.get("qat", {}).get("qat_eta_min", eta_min))
    FP32_CHECKPOINT     = config.get("qat", {}).get("fp32_checkpoint", None)


    # --------------------------- KD 설정 -----------------------------
    KD_CFG = config.get("kd", {})
    USE_KD              = bool(KD_CFG.get("use_kd", False))
    TEACHER_MODEL_NAME  = KD_CFG.get("teacher_model_name", None)  # 예: "timm_efficientnet_b3"
    TEACHER_CHECKPOINT  = KD_CFG.get("teacher_checkpoint", None)  # (옵션) teacher 가중치 경로
    KD_TEMPERATURE      = float(KD_CFG.get("temperature", 4.0))
    KD_ALPHA            = float(KD_CFG.get("alpha", 0.5))
    TEACHER_FREEZE      = bool(KD_CFG.get("freeze_teacher", True))

    print(f"[KD] use_kd={USE_KD}, teacher={TEACHER_MODEL_NAME}, T={KD_TEMPERATURE}, alpha={KD_ALPHA}")

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

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------- 데이터 로더 ----------------------------
    trainloader, valloader, testloader, classes = get_defect_data_loaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_augmentation=use_augmentation,
        split_output_dir=checkpoint_base_dir + "/split_data_info", # 데이터 분할 결과 저장 디렉토리
        img_size=img_size,
    )

    # 추론용 데이터로더 (batch_size=1)
    test_inference_loader = torch.utils.data.DataLoader(
        testloader.dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------------------
    #  (A) PTQ 모델 (model1)
    # --------------------------------------------------------------------


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

    if TRAINING_FLAG:
        print(MODEL_NAME)

        # --------------------------- W&B 설정 ------------------------------
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

        # --------------------------- FP32 학습 --------------------------
        if USE_KD:
            if not TEACHER_MODEL_NAME:
                raise ValueError("KD가 활성화되었지만 'kd.teacher_model_name'이 설정되지 않았습니다.")
            

            print("KD 학습!")
            print("Teacher Model :", TEACHER_MODEL_NAME)
            print("Student Model :", MODEL_NAME)

            # 1) Teacher 생성
            teacher = create_model_by_name(
                TEACHER_MODEL_NAME, num_classes=len(classes), freeze=True
            )
            if TEACHER_CHECKPOINT:
                teacher.load_state_dict(torch.load(TEACHER_CHECKPOINT, map_location="cpu"))

            # 2) Student(KD) 학습
            _ = train_model(
                model=model1,
                criterion=criterion,     # 내부에서 KD 손실로 대체되어 사용됨(검증은 CE)
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
                device=device,
                # --- KD 옵션 ---
                teacher=teacher,
                kd_temperature=KD_TEMPERATURE,
                kd_alpha=KD_ALPHA,
                freeze_teacher=TEACHER_FREEZE,
            )
        else:
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
                device=device,
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
        wandb.log({"Test Loss (FP32)": test_loss_fp32, "Test Acc (FP32)": test_acc_fp32})

        # --------------------------- FP32 추론 -------------------------
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
        ptq_model = apply_post_training_quantization(model1, trainloader)

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
    #  (B) QAT 모델 (model2)
    # --------------------------------------------------------------------
    if not USE_QAT:
        print("QAT is disabled. Exiting after PTQ.")
        return
    
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

    # Best FP32 가중치 로드
    if TRAINING_FLAG:
        # 일반 학습 모드: 바로 앞에서 저장한 best_model_path의 FP32 가중치 사용
        model2.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        # 학습 모드가 아니면, config에서 지정한 FP32 체크포인트 경로(FP32_CHECKPOINT) 사용
        if FP32_CHECKPOINT:
            model2.load_state_dict(torch.load(FP32_CHECKPOINT, map_location=device))
        else:
            # FP32 체크포인트 경로가 없으면 에러 발생
            raise ValueError("FP32 checkpoint is required for QAT training.")

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
    wandb.log(
        {
            "Test Loss (FP32)_model2": total_loss / total,
            "Test Acc (FP32)_model2": correct / total,
        }
    )

    # QAT 파인튜닝 설정
    optimizer_qat = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=QAT_LR)
    scheduler_qat = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer_qat, T_max=QAT_T_MAX, eta_min=QAT_ETA_MIN)
        if use_scheduler
        else None
    )

    # --------------------------- QAT 파인튜닝 ------------------------
    model2_qat = apply_qat_training(
        model=model2,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        test_inference_loader=test_inference_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer_qat,
        scheduler=scheduler_qat,
        epochs=QAT_EPOCHS,
        checkpoint_base_dir=checkpoint_base_dir,
        model_name=f"{MODEL_NAME}_QAT",
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
