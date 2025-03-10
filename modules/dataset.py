import os
import random
import torch
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])

class AddGaussianNoise(object):
    """가우시안 노이즈 추가 (p 확률로)"""
    def __init__(self, mean=0., std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0., 1.)
        return tensor

def get_train_transform():
    """훈련 데이터에 사용될 데이터 증강 (수평/수직 뒤집기, 회전, 밝기·명암, 노이즈 등)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.02, p=0.5),
        # transforms.Normalize((0.569, 0.441, 0.439), (0.149, 0.177, 0.173))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_test_transform():
    """검증/테스트 데이터에 사용될 기본 변환 (증강 없음)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.569, 0.441, 0.439), (0.149, 0.177, 0.173))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
