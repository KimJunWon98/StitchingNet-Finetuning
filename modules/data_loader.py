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
    StitchingNet_make_all_samples, StitchingNetVer2_make_all_samples, stratified_split,
    DefectDataset, get_train_transform_v1, get_train_transform_v2, get_train_transform_v3, get_val_test_transform
)

import json
import os

def get_defect_data_loaders(
    dataset_name,
    data_root,
    batch_size=64,
    num_workers=0,
    train_ratio=0.7,
    val_ratio=0.15,
    use_augmentation=0,
    split_output_dir: str = None,
    json_file: str = None # 입력 JSON 파일 경로 (선택적, 데이터셋이 이미 분할된 경우 사용)
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
    if dataset_name == "StitchingNet":
        samples, classes = StitchingNet_make_all_samples(data_root)
    elif dataset_name == "StitchingNetVer2":
        samples, classes = StitchingNetVer2_make_all_samples(data_root)

    # 2) stratified_split으로 train/val/test 분할
    train_samples, val_samples, test_samples = stratified_split(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    # 3) split 정보 저장: split_output_dir이 지정되면 dirname을 만들고 자동 파일명 생성
    if split_output_dir:
        os.makedirs(split_output_dir, exist_ok=True)
        filename = "dataset_split.json"
        output_path = os.path.join(split_output_dir, filename)

        def _to_rel(samples):
            return [
                (os.path.relpath(path, data_root), lbl)
                for path, lbl in samples
            ]

        dataset_info = {
            "train": _to_rel(train_samples),
            "val":   _to_rel(val_samples),
            "test":  _to_rel(test_samples)
        }
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(dataset_info, fp, ensure_ascii=False, indent=4)
        print(f"[INFO] Saved split info to {output_path}")


    # 3.5) input_json_file이 지정되면 해당 파일에서 샘플 로드
    if json_file:
        data = json.load(open(json_file, "r", encoding="utf-8"))
        train_samples = [
            (os.path.join(data_root, rel), int(lbl))
            for rel, lbl in data["train"]
        ]
        val_samples = [
            (os.path.join(data_root, rel), int(lbl))
            for rel, lbl in data["val"]
        ]
        test_samples = [
            (os.path.join(data_root, rel), int(lbl))
            for rel, lbl in data["test"]
        ]

    print(f"[INFO] Loaded from JSON: {json_file}")

    def _print_head(name, samples, n=5):
        print(f"  {name} (first {n}):")
        for path, lbl in samples[:n]:
            print(f"    - {path}  (label={lbl})")

    _print_head("train_samples", train_samples)
    _print_head("val_samples",   val_samples)
    _print_head("test_samples",  test_samples)

    # 4) transform 정의
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

    # 5) Dataset 생성
    train_dataset = DefectDataset(train_samples, classes, transform=train_transform) 
    val_dataset   = DefectDataset(val_samples,   classes, transform=val_test_transform)
    test_dataset  = DefectDataset(test_samples,  classes, transform=val_test_transform)

    # 6) DataLoader 생성
    # BatchNorm 레이어가 있는 모델을 학습하는 경우 오류가 발생 가능. 배치에 1장의 이미지만 있으면.
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last = True) 
    valloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader, classes
