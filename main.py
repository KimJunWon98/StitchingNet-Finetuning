import yaml
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import os
from dotenv import load_dotenv
import datetime
from zoneinfo import ZoneInfo

# fvcore를 이용해 FLOPs 계산
from fvcore.nn import FlopCountAnalysis

# 모듈 임포트
from modules.seed import seed_everything
from modules.data_loader import get_defect_data_loaders
from modules.dataset import count_labels
from modules.utils import load_model
from modules.train import train_model
from modules.evaluate import test_best_model

def replace_classifier(model, num_classes):
    """
    모델 구조에 따라 최종 분류 레이어를 교체하는 헬퍼 함수.
    다양한 모델(ResNet, MobileNet, 등)에 대응.
    """
    if hasattr(model, 'fc'):  # 예: ResNet, RegNet 등
        if isinstance(model.fc, nn.Linear):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            return model

    if hasattr(model, 'classifier'):  # 예: MobileNet, EfficientNet 등
        if isinstance(model.classifier, nn.Linear):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            return model
        if isinstance(model.classifier, nn.Sequential):
            last_idx = len(model.classifier) - 1
            if isinstance(model.classifier[last_idx], nn.Linear):
                num_ftrs = model.classifier[last_idx].in_features
                model.classifier[last_idx] = nn.Linear(num_ftrs, num_classes)
                return model

    return model

def get_model_statistics(model, input_size=(1, 3, 224, 224)):
    """
    모델의 파라미터 수, 용량(추정), FLOPs를 계산하는 헬퍼 함수.
    - model: 이미 GPU/CPU device에 .to(device) 된 상태의 모델
    - input_size: (batch_size, channel, height, width)
    """
    # 1) 파라미터 수 & 모델 용량 계산
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / (1024**2)  # float32 기준 (4 bytes)

    # 2) FLOPs 계산을 위해 모델을 평가 모드로 전환
    prev_mode = model.training  # 현재 training 여부 저장
    model.eval()
    dummy_input = torch.randn(*input_size, device=next(model.parameters()).device)
    flops_analysis = FlopCountAnalysis(model, dummy_input)
    flops = flops_analysis.total()
    
    # 원래 training 모드였다면 복원
    if prev_mode:
        model.train()

    return param_count, model_size_mb, flops


def main():
    # 1) YAML config 파일 읽기
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    hparams = config["hyperparameters"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    SEED            = hparams["seed"]
    MODEL_LIST      = hparams["model_list"]
    BATCH_SIZE      = hparams["batch_size"]
    EPOCHS          = hparams["epochs"]
    PATIENCE        = hparams["patience"]
    MAX_CHECKPOINTS = hparams["max_checkpoints"]
    NUM_WORKERS     = hparams["num_workers"]
    LAYERS          = hparams["layers"]
    freeze_layers   = train_cfg["freeze_layers"]

    # 추가로, use_augmentation 가져오기
    use_augmentation = data_cfg.get("use_augmentation", False)

    data_root   = data_cfg["root"]
    train_ratio = data_cfg["train_ratio"]
    val_ratio   = data_cfg["val_ratio"]

    project_name        = train_cfg["project_name"]
    use_dataparallel    = train_cfg["use_dataparallel"]
    checkpoint_base_dir = train_cfg["checkpoint_base_dir"]

    # (기타 hyperparam)
    base_lr   = float(train_cfg.get("base_lr", 1e-4))
    head_lr   = float(train_cfg.get("head_lr", 1e-3))
    use_scheduler = train_cfg.get("use_scheduler", False)
    T_max     = train_cfg.get("T_max", EPOCHS)
    eta_min   = float(train_cfg.get("eta_min", 1e-6))

    # 2) 시드 고정
    seed_everything(SEED)

    # 3) wandb 로그인
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # 4) 데이터 로더 준비
    trainloader, valloader, testloader, classes = get_defect_data_loaders(
        data_root=data_root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_augmentation=use_augmentation  # ← 증강 여부
    )

    print("[Dataset Info]")
    print(f"Classes: {classes}")
    print("\nTraining set:")
    count_labels(trainloader, classes)
    print(len(trainloader))

    print("\nValidation set:")
    count_labels(valloader, classes)
    print(len(valloader))

    print("\nTest set:")
    count_labels(testloader, classes)
    print(len(testloader))

    # 4) 모델 학습 루프
    for model_name in MODEL_LIST:
        today_str = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M")
        print(f"\nStarting model: {model_name}")

        model_subdir = f"{today_str}_{model_name}"
        model_checkpoint_dir = os.path.join(checkpoint_base_dir, model_subdir)

        for layer_list in LAYERS:
            layers_str = '_'.join(layer_list)
            wandb_name = f"{model_name}_{layers_str}"
            print(f"Layers (freeze pattern): {layer_list}, freeze_layers={freeze_layers}")

            wandb.init(project=project_name, name=wandb_name)
            wandb.config.update({
                "model_name": model_name,
                "batch_size": BATCH_SIZE,
                "use_augmentation" : use_augmentation,
                "freeze_layers": freeze_layers,
                "layer_pattern": layer_list,
                "seed": SEED
            })

            # 1) 모델 로드
            model = load_model(model_name, pretrained=True)
            # 1-1) 최종 분류 레이어 교체
            model = replace_classifier(model, num_classes=len(classes))

            # (A) Freeze 여부
            if freeze_layers:
                for param in model.parameters():
                    param.requires_grad = False
                for name, param in model.named_parameters():
                    if any(layer_name in name for layer_name in layer_list):
                        param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = True

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # DataParallel
            if torch.cuda.device_count() > 1 and use_dataparallel:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
                model = nn.DataParallel(model)

            model.to(device)

            # (B) Optimizer 설정
            classifier_keywords = ["fc", "classifier", "head"]
            param_groups = []

            if isinstance(model, nn.DataParallel):
                named_params = model.module.named_parameters()
            else:
                named_params = model.named_parameters()

            for name, param in named_params:
                if param.requires_grad:
                    # 최종 레이어인 경우 head_lr, 아니면 base_lr
                    if any(ck in name for ck in classifier_keywords):
                        param_groups.append({"params": param, "lr": head_lr})
                    else:
                        param_groups.append({"params": param, "lr": base_lr})

            optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            criterion = nn.CrossEntropyLoss()

            # (C) 스케줄러 설정
            scheduler = None
            if use_scheduler:
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            # ─────────────────────────────────────────────────────────
            # (D) 모델 통계 계산 (파라미터 수, 모델 용량, FLOPs)
            #     - 반드시 모델이 to(device)된 뒤, 학습 전(초기 상태) 시점에 계산
            param_count, model_size_mb, flops = get_model_statistics(
                model,
                input_size=(1, 3, 224, 224)  # 실제 이미지 크기에 맞게 조정
            )
            # wandb에 기록
            wandb.log({
                "total_params": param_count,
                "model_size_MB": model_size_mb,
                "total_FLOPs": flops
            })
            # ─────────────────────────────────────────────────────────

            # (E) train_model
            results = train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                trainloader=trainloader,
                valloader=valloader,
                layers=layer_list,
                model_name=model_name,
                num_epochs=EPOCHS,
                patience=PATIENCE,
                max_checkpoints=MAX_CHECKPOINTS,
                checkpoint_dir=model_checkpoint_dir,
                scheduler=scheduler
            )

            # (F) 체크포인트 평가
            best_model_path = os.path.join(
                model_checkpoint_dir,
                f"{model_name}_{layers_str}_best.pt"
            )
            print(f"\nBest Model Path: {best_model_path}")
            test_loss, test_acc, average_time_per_batch = test_best_model(model, testloader, device, best_model_path)

            print("[Results]")
            print(results)

            # 추가 wandb 로그
            wandb.log({
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "Model Name": model_name,
                "Batch Size": BATCH_SIZE,
                "inference_time_per_batch": average_time_per_batch
            })

            wandb.finish()

if __name__ == "__main__":
    main()


# import yaml
# import wandb
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# from dotenv import load_dotenv
# import datetime
# from zoneinfo import ZoneInfo

# # 모듈 임포트
# from modules.seed import seed_everything
# from modules.dataset import get_defect_data_loaders, count_labels
# from modules.utils import load_model
# from modules.train import train_model
# from modules.evaluate import test_best_model


# def replace_classifier(model, num_classes):
#     """
#     모델 구조에 따라 최종 분류 레이어를 교체하는 헬퍼 함수.
#     다양한 모델(ResNet, MobileNet, 등)에 대응.
#     """

#     # 1) ResNet 계열: model.fc 속성 존재
#     if hasattr(model, 'fc'):
#         # 예: ResNet, RegNet, etc.
#         if isinstance(model.fc, nn.Linear):
#             # 단순 Linear
#             num_ftrs = model.fc.in_features
#             model.fc = nn.Linear(num_ftrs, num_classes)
#             return model
#         # 혹은 Sequential 등 다양한 형태 가능하므로 필요 시 추가 로직

#     # 2) MobileNet / EfficientNet 계열: 보통 model.classifier 속성
#     if hasattr(model, 'classifier'):
#         # (A) 바로 nn.Linear인 경우
#         if isinstance(model.classifier, nn.Linear):
#             num_ftrs = model.classifier.in_features
#             model.classifier = nn.Linear(num_ftrs, num_classes)
#             return model

#         # (B) Sequential 형태: 마지막 레이어가 nn.Linear인지 확인
#         if isinstance(model.classifier, nn.Sequential):
#             last_idx = len(model.classifier) - 1
#             if isinstance(model.classifier[last_idx], nn.Linear):
#                 num_ftrs = model.classifier[last_idx].in_features
#                 model.classifier[last_idx] = nn.Linear(num_ftrs, num_classes)
#                 return model

#     # 3) 다른 모델의 경우 사용자 정의 처리
#     # 예: Inception, ViT, etc.
#     # 필요 시 모델 구조 확인 후 코드 추가
#     return model


# def main():
#     # 1) YAML config 파일 읽기
#     with open("config.yaml", "r") as f:
#         config = yaml.safe_load(f)

#     # 2) 하이퍼파라미터, 데이터, 학습설정 불러오기
#     hparams = config["hyperparameters"]
#     data_cfg = config["data"]
#     train_cfg = config["training"]

#     SEED = hparams["seed"]
#     MODEL_LIST     = hparams["model_list"]
#     BATCH_SIZE     = hparams["batch_size"]
#     EPOCHS         = hparams["epochs"]
#     PATIENCE       = hparams["patience"]
#     MAX_CHECKPOINTS = hparams["max_checkpoints"]
#     NUM_WORKERS    = hparams["num_workers"]
#     LAYERS         = hparams["layers"]
#     freeze_layers  = train_cfg["freeze_layers"]

#     data_root   = data_cfg["root"]
#     train_ratio = data_cfg["train_ratio"]
#     val_ratio   = data_cfg["val_ratio"]

#     project_name        = train_cfg["project_name"]
#     use_dataparallel    = train_cfg["use_dataparallel"]
#     checkpoint_base_dir = train_cfg["checkpoint_base_dir"]

#     # 추가: 학습률, 스케줄러 파라미터
#     base_lr = float(train_cfg.get("base_lr", 1e-4))
#     head_lr = float(train_cfg.get("head_lr", 1e-3))
#     use_scheduler = train_cfg.get("use_scheduler", False)
#     T_max = train_cfg.get("T_max", EPOCHS)
#     eta_min = float(train_cfg.get("eta_min", 1e-6))

#     # 3) 시드 고정
#     seed_everything(SEED)

#     # 4) wandb 로그인 (.env에서 API Key 로드)
#     load_dotenv()
#     wandb_api_key = os.getenv("WANDB_API_KEY")
#     wandb.login(key=wandb_api_key)

#     # 5) 데이터 준비
#     trainloader, valloader, testloader, classes = get_defect_data_loaders(
#         data_root=data_root,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         train_ratio=train_ratio,
#         val_ratio=val_ratio
#     )

#     print("[Dataset Info]")
#     print(f"Classes: {classes}")
#     print("\nTraining set:")
#     count_labels(trainloader, classes)
#     print("\nValidation set:")
#     count_labels(valloader, classes)
#     print("\nTest set:")
#     count_labels(testloader, classes)


#     # 6) 모델 학습 루프
#     for model_name in MODEL_LIST:

#         # 한국 시간으로 날짜_시분
#         today_str = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M")
        
#         print(f"\nStarting model: {model_name}")

#         model_subdir = f"{today_str}_{model_name}"
#         model_checkpoint_dir = os.path.join(checkpoint_base_dir, model_subdir)

#         for layer_list in LAYERS:
#             layers_str = '_'.join(layer_list)
#             wandb_name = f"{model_name}_{layers_str}"
#             print(f"Layers (freeze pattern): {layer_list}, freeze_layers={freeze_layers}")

#             wandb.init(project=project_name, name=wandb_name)

#             # (옵션) wandb config에 기본 정보 기록
#             wandb.config.update({
#                 "model_name": model_name,
#                 "batch_size": BATCH_SIZE,
#                 "freeze_layers": freeze_layers,
#                 "layer_pattern": layer_list,
#                 "seed":SEED
#             })

#             # 1) 모델 로드
#             model = load_model(model_name, pretrained=True)
            
#             # 1-1) 최종 분류 레이어 교체
#             model = replace_classifier(model, num_classes=len(classes))

#             # (A) Freeze 여부
#             if freeze_layers:
#                 # 부분 동결
#                 for param in model.parameters():
#                     param.requires_grad = False

#                 # layer_list에 포함된 문자열이 name에 있으면 unfreeze
#                 for name, param in model.named_parameters():
#                     if any(layer_name in name for layer_name in layer_list):
#                         param.requires_grad = True
#             else:
#                 # 전부 학습
#                 for param in model.parameters():
#                     param.requires_grad = True

#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#             # DataParallel
#             if torch.cuda.device_count() > 1 and use_dataparallel:
#                 print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
#                 model = nn.DataParallel(model)

#             model.to(device)

#             # (B) Optimizer 설정
#             classifier_keywords = ["fc", "classifier", "head"]
#             param_groups = []

#             if isinstance(model, nn.DataParallel):
#                 named_params = model.module.named_parameters()
#             else:
#                 named_params = model.named_parameters()

#             for name, param in named_params:
#                 if param.requires_grad:
#                     # 최종 레이어일 가능성이 있는 키워드가 있으면 head_lr, 아니면 base_lr
#                     if any(ck in name for ck in classifier_keywords):
#                         param_groups.append({"params": param, "lr": head_lr})
#                     else:
#                         param_groups.append({"params": param, "lr": base_lr})

#             optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
#             criterion = nn.CrossEntropyLoss()

#             # (C) 스케줄러 설정
#             scheduler = None
#             if use_scheduler:
#                 from torch.optim.lr_scheduler import CosineAnnealingLR
#                 scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

#             # (D) train_model
#             results = train_model(
#                 model=model,
#                 criterion=criterion,
#                 optimizer=optimizer,
#                 trainloader=trainloader,
#                 valloader=valloader,
#                 layers=layer_list,
#                 model_name=model_name,
#                 num_epochs=EPOCHS,
#                 patience=PATIENCE,
#                 max_checkpoints=MAX_CHECKPOINTS,
#                 checkpoint_dir=model_checkpoint_dir,
#                 scheduler=scheduler
#             )

#             # (E) 체크포인트 평가
#             best_model_path = os.path.join(
#                 model_checkpoint_dir,
#                 f"{model_name}_{layers_str}_best.pt"
#             )
#             print(f"\nBest Model Path: {best_model_path}")

#             # test_best_model이 (test_loss, test_acc)를 리턴한다고 가정
#             test_loss, test_acc, average_time_per_batch = test_best_model(model, testloader, device, best_model_path)

#             print("[Results]")
#             print(results)

#             # (F) 추가 로그 기록
#             # 모델 이름, 배치 크기, test_loss, test_acc 등을 wandb에 로깅
#             wandb.log({
#                 "Test Loss": test_loss,
#                 "Test Accuracy": test_acc,
#                 "Model Name": model_name,
#                 "Batch Size": BATCH_SIZE,
#                 "inference_time_per_batch" : average_time_per_batch
#             })

#             # wandb 세션 종료
#             wandb.finish()


# if __name__ == "__main__":
#     main()
