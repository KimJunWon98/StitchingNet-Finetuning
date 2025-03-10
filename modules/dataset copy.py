import os
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])

class DefectDataset(Dataset):
    """
    DefectDataset은 data_root 내의 모든 원단 폴더를 순회하여,
    각 원단 폴더 내의 결함(정상 포함) 폴더의 이미지를 로드합니다.
    이때, 라벨은 결함 종류(두 번째 수준 폴더 이름)만을 사용하며, 
    원단 종류는 무시합니다.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.569, 0.441, 0.439), (0.149, 0.177, 0.173))
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.samples = []  # (image_path, defect_label) 튜플 목록
        defect_set = set()

        # 각 원단(예: "A. Cotton-Poly", "B. Linen-Poly", ...) 폴더 내의 결함 폴더 순회
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
                                self.samples.append((img_path, defect))
        
        # 결함 종류를 알파벳순으로 정렬하여 클래스 목록 생성
        self.classes = sorted(list(defect_set))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # 각 샘플의 라벨을 인덱스로 변환
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path

# def get_defect_data_loaders(
#     data_root,
#     batch_size=64,
#     num_workers=0,
#     train_ratio=0.7,
#     val_ratio=0.15,
#     transform=None
# ):
#     """
#     data_root 경로에서 DefectDataset을 이용해 데이터를 로드하고,
#     전체 데이터셋을 train, validation, test로 분할하여 DataLoader를 반환합니다.
#     """
#     dataset = DefectDataset(root=data_root, transform=transform)
#     total = len(dataset)
#     n_train = int(total * train_ratio)
#     n_val = int(total * val_ratio)
#     n_test = total - n_train - n_val

#     train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test])
#     trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return trainloader, valloader, testloader, dataset.classes


def get_defect_data_loaders(
    data_root,
    batch_size=64,
    num_workers=0,
    train_ratio=0.7,
    val_ratio=0.15,
    transform=None
):
    dataset = DefectDataset(root=data_root, transform=transform)
    
    # 1) 전체 샘플의 인덱스와 라벨을 추출 (DefectDataset.samples에 (path, label) 형태로 저장됨)
    indices = list(range(len(dataset)))
    labels = [s[1] for s in dataset.samples]  # 각 sample의 label
    
    # 2) 우선 (train+val) vs test로 1차 분할
    #    예: train_ratio=0.7, val_ratio=0.15 => train+val=0.85, test=0.15
    train_val_ratio = train_ratio + val_ratio
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=1 - train_val_ratio,
        stratify=labels  # stratify 지정
    )
    
    # 3) train+val 세트에서 다시 train / val를 2차 분할
    #    (train+val) 전체를 1로 보면, val 비율 = val_ratio / (train_ratio + val_ratio)
    val_ratio_in_train_val = val_ratio / train_val_ratio

    # train_val_indices에 해당하는 라벨
    train_val_labels = [labels[i] for i in train_val_indices]
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio_in_train_val,
        stratify=train_val_labels  # stratify 지정
    )
    
    # 4) Subset을 사용해 Dataset 분할
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)
    test_dataset  = Subset(dataset, test_indices)
    
    # 5) DataLoader 생성
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader, dataset.classes

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
