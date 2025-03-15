import os
import torch
import json
import matplotlib.pyplot as plt
from collections import deque

class EarlyStopping:
    # val_loss가 최소값을 갱신할 때만 best 모델로 저장

    def __init__(self, patience=5, verbose=False, delta=0, layers='', model_name=''):
        self.patience = patience # 개선 없는 epoch가 'patience'에 도달 시 조기 종료
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.layers = layers
        self.model_name = model_name # 체크포인트 파일명에 사용할 모델 이름 (ex resnet50)

    def __call__(self, val_loss, model, checkpoint_dir=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, checkpoint_dir):  # val_loss가 개선될 때만 호출되어 Best checkpoint를 저장
       

        if checkpoint_dir is None:
            checkpoint_dir = "./"
        checkpoint_filename = f"{self.model_name}_{self.layers}_best.pt"
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_filename)
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                  f"Saving best model as {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)
        self.val_loss_min = val_loss


def train_model(
    model, 
    criterion, 
    optimizer, 
    trainloader, 
    valloader, 
    layers, 
    model_name,
    num_epochs=25, 
    patience=5,  # EarlyStopping patience
    max_checkpoints=None,  # 최근 체크포인트 보관 개수 (없으면 patience로 설정)
    checkpoint_dir="./checkpoints",
    scheduler=None
):

    import wandb
    from collections import deque
    from modules.train import EarlyStopping
    import datetime

    # max_checkpoints가 None이면 patience를 사용
    if max_checkpoints is None:
        max_checkpoints = patience

    # EarlyStopping 세팅
    early_stopping = EarlyStopping(
        patience=patience, 
        verbose=True, 
        layers='_'.join(layers), 
        model_name=model_name
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 훈련/검증 결과 저장용
    layer_key = '_'.join(layers)
    results = {
        layer_key: {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    }
    best_val_accuracy = 0.0

    # 체크포인트 저장 디렉토리
    os.makedirs(checkpoint_dir, exist_ok=True)

    # val_results 폴더 생성 (evaluation.py에서 쓰는 폴더와 동일 경로)
    val_results_folder = os.path.join(checkpoint_dir, "val_results")
    os.makedirs(val_results_folder, exist_ok=True)

    # 최근 체크포인트 경로들 관리 (최신 max_checkpoints개)
    recent_ckpts = deque([], maxlen=max_checkpoints)

    for epoch in range(num_epochs):

        # 1) Train 
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels, paths in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_accuracy = 100.0 * train_correct / train_total
        results[layer_key]['train_loss'].append(train_loss)
        results[layer_key]['train_accuracy'].append(train_accuracy)


        # 2) Validation

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels, paths in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(valloader)
        val_accuracy = 100.0 * val_correct / val_total
        results[layer_key]['val_loss'].append(val_loss)
        results[layer_key]['val_accuracy'].append(val_accuracy)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # wandb 로그
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        # scheduler step (epoch 단위)
        if scheduler is not None:
            scheduler.step()


        # 3) Save checkpoint (최근 max_checkpoints개)

        epoch_ckpt_filename = f"{model_name}_{layer_key}_epoch{epoch+1}.pt"
        epoch_ckpt_path = os.path.join(checkpoint_dir, epoch_ckpt_filename)
        torch.save(model.state_dict(), epoch_ckpt_path)

        #  Deque에 추가 → 초과 시 가장 오래된 ckpt 제거
        if len(recent_ckpts) == max_checkpoints:
            # 가장 오래된 ckpt 파일 삭제
            old_ckpt = recent_ckpts.popleft()
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
        recent_ckpts.append(epoch_ckpt_path)


        # 4) Early Stopping (best val_loss 기준)
        early_stopping(val_loss, model, checkpoint_dir=checkpoint_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 최종 결과
    print(f"Finished Training with Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Recent checkpoints: {[p for p in recent_ckpts]}")


    # # 5) 그래프 저장
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.plot(results[layer_key]['train_loss'], label='Train Loss')
    # plt.plot(results[layer_key]['val_loss'], label='Val Loss')
    # plt.title('Loss over epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(results[layer_key]['train_accuracy'], label='Train Accuracy')
    # plt.plot(results[layer_key]['val_accuracy'], label='Val Accuracy')
    # plt.title('Accuracy over epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()

    # # 그래프를 val_results 폴더에 저장
    # plot_path = os.path.join(val_results_folder, f"{model_name}_{layer_key}_training_curve.png")
    # plt.savefig(plot_path)
    # print(f"Training curve plot saved to: {plot_path}")

    # plt.close()  # plt.show() 대신 이미지 파일만 저장


    # 6) JSON 저장
    json_path = os.path.join(val_results_folder, "training_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Training results JSON saved to: {json_path}")

    return results
