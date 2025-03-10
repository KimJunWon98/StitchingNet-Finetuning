from torch.utils.data import DataLoader
from .dataset import (
    make_all_samples, stratified_split,
    DefectDataset, get_train_transform, get_val_test_transform
)

def get_defect_data_loaders(
    data_root,
    batch_size=64,
    num_workers=0,
    train_ratio=0.7,
    val_ratio=0.15,
    use_augmentation=False
):
    """
    - use_augmentation=True면, 훈련용에만 수평/수직 뒤집기 등 증강이 들어감.
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
    if use_augmentation:
        train_transform = get_train_transform()
    else:
        # 증강 없이 기본 변환만
        train_transform = get_val_test_transform()

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
