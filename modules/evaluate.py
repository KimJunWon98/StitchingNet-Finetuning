import torch
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def test_model(model, dataloader, device):
    """학습 후의 모델을 테스트 세트로 간단 평가"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, digits=4)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

def test_best_model(model, dataloader, device, checkpoint_path):
    """
    저장된 체크포인트(최적 모델) 로드 후 평가
    misclassified samples를 (checkpoint_path의 상위 폴더)/val_results 에 CSV로 저장
    """
    # 1) 체크포인트 로드
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    misclassified_samples = []
    processing_times = []

    # 2) 배치별 추론
    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            start_time = time.time()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            end_time = time.time()

            processing_times.append(end_time - start_time)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # 오분류 샘플(정답 != 예측)만 기록
            mis_indices = (predicted != labels).nonzero().flatten()
            for idx in mis_indices:
                misclassified_samples.append((paths[idx], labels[idx].item(), predicted[idx].item()))

    # 3) 정확도, 혼동행렬, 리포트
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, digits=4)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

    # 4) misclassified samples를 상위 폴더/val_results에 저장
    parent_folder = os.path.dirname(checkpoint_path)        # 체크포인트 상위 폴더
    val_results_folder = os.path.join(parent_folder, "val_results")
    os.makedirs(val_results_folder, exist_ok=True)

    # CSV 형식으로 저장
    results_file = os.path.join(val_results_folder, "misclassified_samples.csv")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("path,true_label,pred_label\n")
        for path, true_label, pred_label in misclassified_samples:
            # CSV 형태: 한 줄에 path, true_label, pred_label
            f.write(f"{path},{true_label},{pred_label}\n")

    print(f"\nMisclassified samples saved to: {results_file}")

    # 5) 평균 배치 처리 시간
    average_time_per_batch = np.mean(processing_times)
    # 주의: 이 값은 "배치 전체를 처리하는 데 걸린 평균 시간(초)"
    # 배치 내 이미지 수를 고려하지 않았으므로 "배치당 시간"임
    print(f"Average processing time per batch: {average_time_per_batch:.4f} seconds")
