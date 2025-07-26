"""
data_loader.py

StitchingNet 데이터셋을 위한 DataLoader 생성 유틸리티.
- 데이터셋 경로에서 샘플 및 클래스 목록 생성
- 계층적/라벨 균형 분할로 train/val/test 세트 생성
- 다양한 데이터 증강(transform) 버전 지원
- PyTorch DataLoader 객체 반환

사용 예시:
    trainloader, valloader, testloader, classes = get_defect_data_loaders(...)
"""

from torch.utils.data import DataLoader
from .dataset import (
    make_all_samples, stratified_split,
    DefectDataset, get_train_transform_v1, get_train_transform_v2, get_train_transform_v3, get_val_test_transform
)

def get_defect_data_loaders(
    data_root,
    batch_size=64,
    num_workers=0,
    train_ratio=0.7,
    val_ratio=0.15,
    use_augmentation=0
):

    """
    데이터셋을 로드하고 DataLoader를 반환합니다.
    Args:
        data_root: 데이터셋 경로
        batch_size: 배치 크기
        num_workers: DataLoader worker 수
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        use_augmentation: 증강 버전 (0~3)
        shuffle_train: 학습 데이터 셔플 여부
        drop_last: 마지막 배치 버릴지 여부
    Returns:
        trainloader, valloader, testloader, classes
    """

    # 1) 전체 samples와 classes 생성
    samples, classes = make_all_samples(data_root)

    # 2) stratified_split으로 train/val/test 분할
    train_samples, val_samples, test_samples = stratified_split(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    # 3) transform 정의
    transform_map = {
        1: (get_train_transform_v1, "version 1"),
        2: (get_train_transform_v2, "version 2"),
        3: (get_train_transform_v3, "version 3"),
        0: (get_val_test_transform, "version 0"),
    }

    transform_func, version_name = transform_map.get(use_augmentation, transform_map[0])
    print(version_name)

    train_transform = transform_func()
    val_test_transform = get_val_test_transform()

    # 4) Dataset 생성
    train_dataset = DefectDataset(train_samples, classes, transform=train_transform) 
    val_dataset   = DefectDataset(val_samples,   classes, transform=val_test_transform)
    test_dataset  = DefectDataset(test_samples,  classes, transform=val_test_transform)

    # 5) DataLoader 생성
    # BatchNorm 레이어가 있는 모델을 학습하는 경우 오류가 발생 가능. 배치에 1장의 이미지만 있으면.
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last = True) 
    valloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader, classes
