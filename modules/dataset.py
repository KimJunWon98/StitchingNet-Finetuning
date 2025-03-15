# dataset.py
import os
import random
import torch
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2



def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])

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


# Albumentations 증강 정의

# 1) 기하학적 변형만
def get_train_transform_v1():

    return AlbumentationsTransform(A.Compose([
        # 기하학적 변형형
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=15, # ±15도 범위 회전
            p=0.5,
            interpolation=cv2.INTER_LANCZOS4,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        # Normalize & ToTensor
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], p=1.0))

# 2) 기하학적 변형 + 색상 변형
def get_train_transform_v2():

    return AlbumentationsTransform(A.Compose([
        # 기하학적 변형형
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=15, # ±15도 범위 회전
            p=0.5,
            interpolation=cv2.INTER_LANCZOS4,
            border_mode=cv2.BORDER_REFLECT_101
        ),

        # 색상 변형
        # 1) Hue Saturation Value: 이미지의 색조, 채도, 명도 변경
        A.HueSaturationValue(
            hue_shift_limit=20,      # hue 변화 범위: -20 ~ +20
            sat_shift_limit=30,      # saturation 변화 범위: -30 ~ +30
            val_shift_limit=20,      # value 변화 범위: -20 ~ +20
            p=0.5                    # 50% 확률로 적용
        ),

        # 2) Color Jitter: 밝기, 대비, 채도, 색조를 무작위로 조정
        A.ColorJitter(
            brightness=0.2,          # 밝기를 ±20% 조정
            contrast=0.2,            # 대비를 ±20% 조정
            saturation=0.2,          # 채도를 ±20% 조정
            hue=0.1,                 # 색조를 ±10% 조정
            p=0.5                    # 50% 확률로 적용
        ),
        
        # 3) RGB Shift: R, G, B 채널 각각을 무작위로 이동
        A.RGBShift(
            r_shift_limit=20,        # R 채널 변화 범위: -20 ~ +20
            g_shift_limit=20,        # G 채널 변화 범위: -20 ~ +20
            b_shift_limit=20,        # B 채널 변화 범위: -20 ~ +20
            p=0.5                    # 50% 확률로 적용
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], p=1.0))

# 3) 기하학적 변형 + 색상 변형 + 블러 or 노이즈즈
def get_train_transform_v3():

    return AlbumentationsTransform(A.Compose([
        # 기하학적 변형형
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=15, # ±15도 범위 회전
            p=0.5,
            interpolation=cv2.INTER_LANCZOS4,
            border_mode=cv2.BORDER_REFLECT_101
        ),

        # 색상 변형
        # 1) Hue Saturation Value: 이미지의 색조, 채도, 명도 변경
        A.HueSaturationValue(
            hue_shift_limit=20,      # hue 변화 범위: -20 ~ +20
            sat_shift_limit=30,      # saturation 변화 범위: -30 ~ +30
            val_shift_limit=20,      # value 변화 범위: -20 ~ +20
            p=0.5                    # 50% 확률로 적용
        ),

        # 2) Color Jitter: 밝기, 대비, 채도, 색조를 무작위로 조정
        A.ColorJitter(
            brightness=0.2,          # 밝기를 ±20% 조정
            contrast=0.2,            # 대비를 ±20% 조정
            saturation=0.2,          # 채도를 ±20% 조정
            hue=0.1,                 # 색조를 ±10% 조정
            p=0.5                    # 50% 확률로 적용
        ),
        
        # 3) RGB Shift: R, G, B 채널 각각을 무작위로 이동
        A.RGBShift(
            r_shift_limit=20,        # R 채널 변화 범위: -20 ~ +20
            g_shift_limit=20,        # G 채널 변화 범위: -20 ~ +20
            b_shift_limit=20,        # B 채널 변화 범위: -20 ~ +20
            p=0.5                    # 50% 확률로 적용
        ),
        

        # blur or noise
        # MotionBlur, GaussianBlur,  GaussNoise
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3,5), p=1.0),
        ], p=0.25),

        A.GaussNoise(
            std_range=(0.01, 0.1),
            mean_range=(0.0, 0.0),
            per_channel=True,
            noise_scale_factor=1,
            p=0.5
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], p=1.0))

# 검증/테스트용 변환 (증강 없음)
def get_val_test_transform():  # torch.Tensor 반환. (C, H, W)
    return AlbumentationsTransform(A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], p=1.0))

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


def make_all_samples(root):
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



# 시각화용 normalize 적용 X 
# ===========================================
# 1) 기하학적 변형만 (No Norm 버전)
# ===========================================
def get_train_transform_v1_no_norm():
    return AlbumentationsTransform(A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=15,  # ±15도 범위 회전
            p=0.5,
            interpolation=cv2.INTER_LANCZOS4,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        # A.Normalize(...) 제거
        ToTensorV2()
    ], p=1.0))

# ===========================================
# 2) 기하학적 변형 + 색상 변형 (No Norm 버전)
# ===========================================
def get_train_transform_v2_no_norm():
    return AlbumentationsTransform(A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=15,
            p=0.5,
            interpolation=cv2.INTER_LANCZOS4,
            border_mode=cv2.BORDER_REFLECT_101
        ),

        # Hue Saturation Value
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        # Color Jitter
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        # RGB Shift
        A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20,
            p=0.5
        ),
        # A.Normalize(...) 제거
        ToTensorV2()
    ], p=1.0))

# ===========================================
# 3) 기하학적 변형 + 색상 변형 + 블러 or 노이즈 (No Norm 버전)
# ===========================================
def get_train_transform_v3_no_norm():
    return AlbumentationsTransform(A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=15,
            p=0.5,
            interpolation=cv2.INTER_LANCZOS4,
            border_mode=cv2.BORDER_REFLECT_101
        ),

        # Hue Saturation Value
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        # Color Jitter
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        # RGB Shift
        A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20,
            p=0.5
        ),

        # blur or noise
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3,5), p=1.0),
        ], p=0.25),

        A.GaussNoise(
            std_range=(0.01, 0.1),
            mean_range=(0.0, 0.0),
            per_channel=True,
            noise_scale_factor=1,
            p=0.5
        ),

        # A.Normalize(...) 제거
        ToTensorV2()
    ], p=1.0))