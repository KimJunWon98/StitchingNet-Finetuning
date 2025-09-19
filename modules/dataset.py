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
from typing import List, Tuple, Dict

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
    img_size: tuple = (224, 224), 
) -> AlbumentationsTransform:
    ops = [A.Resize(*img_size)]
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
def get_train_transform_v1(img_size):
    return make_transform(geo=True, color=False, blur_noise=False, normalize=True, img_size=img_size)


def get_train_transform_v2(img_size):
    return make_transform(geo=True, color=True, blur_noise=False, normalize=True, img_size=img_size)


def get_train_transform_v3(img_size):
    return make_transform(geo=True, color=True, blur_noise=True, normalize=True, img_size=img_size)


def get_val_test_transform(img_size):
    return make_transform(geo=False, color=False, blur_noise=False, normalize=True, img_size=img_size)


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


# def ImageNet_1k_make_all_samples(root, split="train"):
#     """
#     ILSVRC2012 ImageNet 데이터셋에서 (img_path, label_idx) 리스트와
#     classes(클래스 synset 목록), class_to_idx 매핑을 생성

#     Args:
#         root (str): ImageNet 루트 경로 (예: ".../ImageNet-1k")
#         split (str): "train" 또는 "val"

#     Returns:
#         samples (list of (img_path, label_idx))
#         classes (list of str): synset ID 리스트 (예: ["n01440764", ...])
#         class_to_idx (dict): {synset: index}
#     """
#     split_dir = os.path.join(root, split)
#     samples = []
#     synset_set = set()

#     if split == "train":
#         # train: synset 폴더 구조
#         for synset in os.listdir(split_dir):
#             synset_path = os.path.join(split_dir, synset)
#             if os.path.isdir(synset_path):
#                 synset_set.add(synset)
#                 for fname in os.listdir(synset_path):
#                     if is_image_file(fname):
#                         img_path = os.path.join(synset_path, fname)
#                         samples.append((img_path, synset))
#     elif split == "val":
#         # val: 이미지가 한 폴더에 모여 있고, 라벨은 devkit ground truth에서 매핑해야 함
#         gt_file = os.path.join(root, "ILSVRC2012_devkit_t12", "data", "ILSVRC2012_validation_ground_truth.txt")
#         synset_words_file = os.path.join(root, "ILSVRC2012_devkit_t12", "data", "meta.mat")

#         # ground truth는 synset ID가 아니라 정수 index라서 meta.mat 파싱 필요
#         # 여기서는 placeholder: devkit 파싱 함수 별도 필요
#         raise NotImplementedError("val split은 devkit(meta.mat) 파싱 필요")

#     # 클래스 목록과 매핑
#     classes = sorted(list(synset_set))
#     class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

#     # samples에 index 적용
#     samples = [(path, class_to_idx[label]) for (path, label) in samples]

    
#     return samples, classes
#     # return samples, classes, class_to_idx


def ImageNet_1k_make_all_samples(root: str, split: str = "train") -> Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
    """
    ILSVRC2012(ImageNet-1K)에서 (img_path, label_idx) 리스트와 classes, class_to_idx를 반환합니다.
      - train: root/train/<WNID>/*.JPEG
      - val  : root/val/*.JPEG  (+ devkit의 meta.mat, validation_ground_truth.txt로 라벨 매핑)
      - test : root/test/*.JPEG (라벨 비공개 → label_idx = -1)

    Returns:
        samples: List[(img_path, label_idx)]
        classes: List[str]  (알파벳 정렬된 WNID 리스트; test는 빈 리스트)
        class_to_idx: Dict[WNID, idx]  (test는 빈 dict)
    """
    # ---- 내부 유틸 ----
    def _is_img(fn: str) -> bool:
        fn = fn.lower()
        return fn.endswith(".jpg") or fn.endswith(".jpeg") or fn.endswith(".png")

    root = os.path.abspath(root)
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"{split_dir} 가 존재하지 않습니다.")

    samples: List[Tuple[str, int]] = []
    classes: List[str] = []
    class_to_idx: Dict[str, int] = {}

    if split == "train":
        synset_set = set()
        for synset in os.listdir(split_dir):
            synset_path = os.path.join(split_dir, synset)
            if os.path.isdir(synset_path):
                synset_set.add(synset)
                for fname in os.listdir(synset_path):
                    if _is_img(fname):
                        samples.append((os.path.join(synset_path, fname), synset))
        classes = sorted(synset_set)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        samples = [(p, class_to_idx[syn]) for (p, syn) in samples]
        return samples, classes
        # return samples, classes, class_to_idx

    elif split == "val":
        # 1) devkit 폴더 탐색
        devkit_dir = None
        for cand in ("ILSVRC2012_devkit_t12", "devkit"):
            p = os.path.join(root, cand)
            if os.path.isdir(p):
                devkit_dir = p
                break
        if devkit_dir is None:
            raise FileNotFoundError(
                f"Devkit directory not found under {root}. "
                "Expected 'ILSVRC2012_devkit_t12/' (untar the devkit tar.gz first)."
            )

        # 2) meta.mat 로드 → ILSVRC2012_ID(1..1000) → WNID 매핑, classes
        try:
            from scipy.io import loadmat
        except ImportError as e:
            raise ImportError("val 처리를 위해 scipy가 필요합니다. `pip install scipy`") from e

        meta_path = os.path.join(devkit_dir, "data", "meta.mat")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta.mat not found: {meta_path}")

        mat = loadmat(meta_path, squeeze_me=True, struct_as_record=False)
        synsets = mat["synsets"]

        id_to_wnid: Dict[int, str] = {}
        wnids: List[str] = []
        for s in synsets:
            ilsvrc_id = int(s.ILSVRC2012_ID)
            num_children = int(s.num_children)
            if num_children == 0 and 1 <= ilsvrc_id <= 1000:
                wnid = str(s.WNID)
                id_to_wnid[ilsvrc_id] = wnid
                wnids.append(wnid)

        classes = sorted(wnids)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # 3) validation GT 읽기 (파일명 알파벳 오름차순과 1:1 대응)
        gt_path = os.path.join(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"validation ground truth not found: {gt_path}")

        with open(gt_path, "r") as f:
            gt_ids = [int(line.strip()) for line in f if line.strip()]

        val_files = sorted([fn for fn in os.listdir(split_dir) if _is_img(fn)])
        if len(val_files) != len(gt_ids):
            raise RuntimeError(f"val 이미지 수({len(val_files)})와 GT 라인 수({len(gt_ids)})가 다릅니다.")

        for fname, ilsvrc_id in zip(val_files, gt_ids):
            wnid = id_to_wnid.get(ilsvrc_id)
            if wnid is None:
                # 방어적 처리: 드물지만 매핑 실패 시 스킵
                continue
            samples.append((os.path.join(split_dir, fname), class_to_idx[wnid]))

        return samples, classes
        # return samples, classes, class_to_idx

    elif split == "test":
        for fname in os.listdir(split_dir):
            if _is_img(fname):
                samples.append((os.path.join(split_dir, fname), -1))
        return samples, [], {}

    else:
        raise ValueError("split must be one of {'train','val','test'}")


def ImageNet_tiny_make_all_samples(root: str, split: str = "train") -> Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
    """
    Tiny-ImageNet-200에서 (img_path, label_idx), classes(WNID 리스트), class_to_idx를 반환합니다.
      - train: root/train/<WNID>/images/*.JPEG
      - val  : root/val/images/*.JPEG  + val_annotations.txt로 라벨 매핑
      - test : root/test/images/*.JPEG (라벨 비공개 → label_idx = -1)

    Args:
        root: tiny-imagenet-200 디렉토리 경로 (예: ".../ImageNet-tiny/tiny-imagenet-200")
        split: "train" | "val" | "test"

    Returns:
        samples: List[(img_path, label_idx)]
        classes: List[str]  (WNID 리스트; 가능한 경우 wnids.txt 순서를 유지)
        class_to_idx: Dict[WNID, idx]
    """

    def _is_img(fn: str) -> bool:
        f = fn.lower()
        return f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png")

    root = os.path.abspath(root)
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of {'train','val','test'}")

    # ---- 클래스 목록 구성: wnids.txt가 있으면 그 순서를 사용 (권장) ----
    wnids_path = os.path.join(root, "wnids.txt")
    classes: List[str] = []
    if os.path.isfile(wnids_path):
        with open(wnids_path, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        # fallback: train 디렉토리의 하위 폴더를 스캔해 알파벳 정렬
        train_dir = os.path.join(root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Neither {wnids_path} nor train/ found under {root}")
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    class_to_idx: Dict[str, int] = {wnid: i for i, wnid in enumerate(classes)}

    samples: List[Tuple[str, int]] = []

    if split == "train":
        train_dir = os.path.join(root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"{train_dir} not found")

        # 각 WNID 폴더 아래 images/의 모든 이미지를 수집
        for wnid in classes:
            wnid_dir = os.path.join(train_dir, wnid)
            img_dir = os.path.join(wnid_dir, "images")
            if not os.path.isdir(img_dir):
                # 일부 배포본은 바로 wnid 폴더 안에 이미지가 있을 수도 있으니 fallback
                img_dir = wnid_dir
            if not os.path.isdir(img_dir):
                # 해당 wnid가 실제 train에 없을 수도 있으므로 스킵
                continue

            for fname in os.listdir(img_dir):
                if _is_img(fname):
                    img_path = os.path.join(img_dir, fname)
                    samples.append((img_path, class_to_idx[wnid]))

        return samples, classes
        # return samples, classes, class_to_idx

    elif split == "val":
        val_dir = os.path.join(root, "val")
        img_dir = os.path.join(val_dir, "images")
        ann_path = os.path.join(val_dir, "val_annotations.txt")
        if not os.path.isdir(val_dir) or not os.path.isdir(img_dir):
            raise FileNotFoundError(f"{img_dir} not found")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"{ann_path} not found")

        # val_annotations.txt 포맷: "filename<TAB>wnid<TAB>xmin<TAB>ymin<TAB>xmax<TAB>ymax"
        # 라벨만 필요하므로 filename -> wnid 매핑만 사용
        fname_to_wnid: Dict[str, str] = {}
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    filename, wnid = parts[0], parts[1]
                    fname_to_wnid[filename] = wnid

        # images/ 안의 파일을 사전순으로 읽고 라벨 매핑
        img_files = sorted([fn for fn in os.listdir(img_dir) if _is_img(fn)])
        for fname in img_files:
            wnid = fname_to_wnid.get(fname)
            if wnid is None:
                # 매핑 없으면 스킵 (비정상 배포 방어)
                continue
            if wnid not in class_to_idx:
                # wnids.txt에 없는 wnid가 온다면 방어적으로 추가할 수도 있지만, 보통 발생하지 않음
                continue
            img_path = os.path.join(img_dir, fname)
            samples.append((img_path, class_to_idx[wnid]))

        return samples, classes
        # return samples, classes, class_to_idx

    else:  # split == "test"
        test_dir = os.path.join(root, "test")
        img_dir = os.path.join(test_dir, "images")
        if not os.path.isdir(img_dir):
            # 일부 배포본은 test/ 바로 아래에 images 없이 파일이 있을 수 있음
            img_dir = test_dir
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"{os.path.join(test_dir, 'images')} not found")

        for fname in sorted(os.listdir(img_dir)):
            if _is_img(fname):
                img_path = os.path.join(img_dir, fname)
                samples.append((img_path, -1))  # test 라벨 비공개
        
        return samples, classes
        # return samples, classes, class_to_idx

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