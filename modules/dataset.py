# dataset.py
import os
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


IMG_SIZE = (224, 224)
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# 이미지 파일 확장자 체크 함수
def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])

# Albumentations 변환을 PyTorch Dataset에서 사용하기 위한 래퍼 클래스
class AlbumentationsTransform:
    """
    Albumentations 변환을 감싼 Wrapper.
    __call__(img) 형태로 호출하면, PIL 이미지 혹은 numpy array를 받아
    transform(image=...)로 처리하고, 최종 torch.Tensor를 반환.
    """

    def __init__(self, a_transform):
        """
        a_transform: Albumentations.Compose(...) 등 Albumentations 변환 객체
        """
        self.a_transform = a_transform

    def __call__(self, img):
        # img가 PIL 객체면 np.array()로 변환
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Albumentations 변환 적용
        augmented = self.a_transform(image=img)
        # 최종 결과물은 augmented["image"]에 들어 있음 (A.ToTensorV2()가 포함된 경우 torch.Tensor 형태)
        return augmented["image"]


# 증강 옵션을 조합하여 AlbumentationsTransform 객체를 생성
def make_transform(
    geo: bool = True,
    color: bool = False,
    blur_noise: bool = False,
    normalize: bool = True,
) -> AlbumentationsTransform:
    ops = [A.Resize(*IMG_SIZE)]
    if geo:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=15,
                p=0.5,
                interpolation=cv2.INTER_LANCZOS4,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
        ]
    if color:
        ops += [
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.5,
            ),
        ]
    if blur_noise:
        ops += [
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3,5), p=1.0),
            ], p=0.35),
            A.GaussNoise(
                std_range=(0.01, 0.05),
                mean_range=(0.0, 0.0),
                per_channel=True,
                noise_scale_factor=0.1,
                p=0.35,
            ),
        ]
    if normalize:
        ops.append(A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    ops.append(ToTensorV2())
    return AlbumentationsTransform(A.Compose(ops, p=1.0))


# --- 버전별 Transform 정의 ---------------------
def get_train_transform_v1():
    return make_transform(geo=True, color=False, blur_noise=False, normalize=True)


def get_train_transform_v2():
    return make_transform(geo=True, color=True, blur_noise=False, normalize=True)


def get_train_transform_v3():
    return make_transform(geo=True, color=True, blur_noise=True, normalize=True)


def get_val_test_transform():
    return make_transform(geo=False, color=False, blur_noise=False, normalize=True)


def get_train_transform_v1_no_norm():
    return make_transform(geo=True, color=False, blur_noise=False, normalize=False)


def get_train_transform_v2_no_norm():
    return make_transform(geo=True, color=True, blur_noise=False, normalize=False)


def get_train_transform_v3_no_norm():
    return make_transform(geo=True, color=True, blur_noise=True, normalize=False)


# Dataset 클래스 정의
class DefectDataset(Dataset):
    """
    특정 (image_path, label) 리스트와 transform을 받아, __getitem__에서 이미지를 로드·변환해 반환
    """
    def __init__(self, samples, classes, transform=None):
        """
        samples: [(img_path, label_idx), ...] 형태
        classes: 전체 클래스 이름 리스트(인덱스 순서대로 정렬된 상태)
        transform: torchvision.transforms
        """
        self.samples = samples
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path


# StitchingNet version2
def StitchingNetVer2_make_all_samples(root: str):
    """
    root ─┐
          ├─ Bobbin thread pulling up/
          ├─ Broken stitch/
          ├─ …
          └─ Twisted/
               ├─ img_001.jpg
               └─ …

    (img_path, label_idx) 리스트와
    classes, class_to_idx 딕셔너리를 반환
    """

    # 1) 클래스(폴더) 이름 — 고정 12종
    classes = [
        "Bobbin thread pulling up",
        "Broken stitch",
        "Crooked seam",
        "Needle mark",
        "Overlapped stitch",
        "Pleated fabric",
        "Puckering",
        "Ready_Normal",
        "Skipped stitch",
        "Stain_and_Damage",
        "Thread sagging",
        "Twisted",
    ]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # 2) 모든 이미지 탐색
    samples = []
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"폴더가 없습니다: {cls_dir}")

        # 하위 폴더가 더 있을 수도 있으니 os.walk 사용
        for dirpath, _, filenames in os.walk(cls_dir):
            for fname in filenames:
                if is_image_file(fname):
                    img_path = os.path.join(dirpath, fname)
                    samples.append((img_path, class_to_idx[cls]))

    if len(samples) == 0:
        raise RuntimeError(f"No images found under {root}")

    return samples, classes
    # return samples, classes, class_to_idx

def StitchingNet_make_all_samples(root):
    """
    data_root 디렉토리 구조를 훑어서 (img_path, defect_label_idx) 형태의 전체 samples와
    classes(결함종류 목록), class_to_idx를 생성해 반환
    """
    samples = []
    defect_set = set()

    # 1) root 내부의 모든 fabric 폴더 → defect 폴더 → 이미지
    for fabric in os.listdir(root):
        fabric_path = os.path.join(root, fabric)
        if os.path.isdir(fabric_path):
            for defect in os.listdir(fabric_path):
                defect_path = os.path.join(fabric_path, defect)
                if os.path.isdir(defect_path):
                    defect_set.add(defect)
                    for fname in os.listdir(defect_path):
                        if is_image_file(fname):
                            img_path = os.path.join(defect_path, fname)
                            samples.append((img_path, defect))

    # 2) 클래스 목록, 라벨 인덱스 매핑
    classes = sorted(list(defect_set))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # 3) samples에 라벨 인덱스 적용
    #    [(img_path, label_idx), ...]
    samples = [(path, class_to_idx[label]) for (path, label) in samples]

    return samples, classes




# stratified_split 함수
def stratified_split(samples, train_ratio=0.7, val_ratio=0.15):
    """
    전체 samples 리스트를 대상으로 클래스 라벨 분포를 유지한 상태로(train/val/test) 분할
    반환: train_samples, val_samples, test_samples
    """
    total = len(samples)
    indices = list(range(total))
    labels = [s[1] for s in samples]  # label_idx 리스트

    # (train+val) vs test
    train_val_ratio = train_ratio + val_ratio
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=1 - train_val_ratio,
        stratify=labels
    )

    # train vs val
    val_ratio_in_train_val = val_ratio / train_val_ratio
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio_in_train_val,
        stratify=train_val_labels
    )

    train_samples = [samples[i] for i in train_indices]
    val_samples   = [samples[i] for i in val_indices]
    test_samples  = [samples[i] for i in test_indices]
    return train_samples, val_samples, test_samples


def count_labels(loader, classes):
    """
    주어진 DataLoader에 대해 각 클래스(결함 종류)별 이미지 개수를 출력합니다.
    """
    all_labels = []
    for images, labels, paths in loader:
        all_labels.extend(labels.numpy())
    label_count = Counter(all_labels)
    sorted_labels = sorted(label_count.items(), key=lambda x: classes[x[0]])
    for label, count in sorted_labels:
        print(f"Class {classes[label]}: {count} images")

def show_augmented_samples(dataset, n_samples=5):
    """
    (증강된) Dataset에서 n_samples개 이미지를 뽑아 시각화
    """
    import matplotlib.pyplot as plt

    for i in range(n_samples):
        img, label, path = dataset[i]
        # img: (C, H, W) 텐서 → (H, W, C)로 permute
        img_np = img.permute(1, 2, 0).cpu().numpy()
        # Normalize를 했으니 clip으로 간단히 시연
        img_np = img_np.clip(0, 1)

        plt.figure(figsize=(3,3))
        plt.imshow(img_np)
        plt.title(f"Label: {dataset.classes[label]}")  
        plt.axis('off')
        plt.show()