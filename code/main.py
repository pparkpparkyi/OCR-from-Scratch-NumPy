# main.py
import numpy as np
import matplotlib.pyplot as plt
from model import OCRModel
from dataset import OCRDataset
from loss import CTCLoss  # <-- [수정] CTCLoss 임포트
import os

# ----------------------------------------------------
# 1. 옵티마이저 (SGD 및 Adam)
# ----------------------------------------------------
class SGD:
    """확률적 경사하강법 옵티마이저"""
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

class Adam:
    """Adam 옵티마이저 (SGD보다 빠름)"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 1차 모멘트
        self.v = None  # 2차 모멘트
        self.t = 0     # 타임스텝

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)

# ----------------------------------------------------
# 2. 학습 함수 (train_model) 수정
# ----------------------------------------------------
def train_model(model, dataset, epochs=20, batch_size=32, lr=0.001):
    """모델 학습"""
    optimizer = Adam(lr=lr)  # <-- [수정] Adam 옵티마이저 사용
    
    train_losses = []
    # train_accs = []  <-- [삭제] Accuracy 계산 제거
    
    print("Starting training...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        # epoch_acc = 0  <-- [삭제]
        num_batches = len(dataset) // batch_size
        
        for batch in range(num_batches):
            # 배치 데이터 가져오기
            # t_batch는 이제 NumPy 배열이 아닌, 파이썬 리스트입니다.
            x_batch, t_batch = dataset.get_batch(batch_size)
            
            # 순전파 및 손실 계산 (CTCLoss 호출)
            loss = model.loss(x_batch, t_batch)
            
            # 역전파 (CTCLoss의 backward 호출)
            model.backward()
            
            # 파라미터 업데이트
            params, grads = model.get_params_and_grads()
            optimizer.update(params, grads)
            
            epoch_loss += loss
            
            # [삭제] --- Accuracy 계산 로직 ---
            # pred = model.predict(x_batch)
            # acc = np.mean(pred == t_batch) # (이 코드는 이제 작동하지 않음)
            # epoch_acc += acc
            # -------------------------------
            
            if batch % 10 == 0:
                # [수정] Acc 출력 제거
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch}/{num_batches}, Loss: {loss:.4f}")
        
        # 에포크 평균
        avg_loss = epoch_loss / num_batches
        # avg_acc = epoch_acc / num_batches  <-- [삭제]
        
        train_losses.append(avg_loss)
        # train_accs.append(avg_acc)  <-- [삭제]
        
        # [수정] Acc 출력 제거
        print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}\n")
    
    return train_losses  # <-- [수정] losses만 반환

# ----------------------------------------------------
# 3. 시각화 함수 (plot_results) 수정
# ----------------------------------------------------
def plot_results(train_losses): # <-- [수정] train_accs 받지 않음
    """학습 결과 시각화"""
    
    # [수정] 1개의 그래프만 생성
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    # Loss 그래프
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()
    ax1.grid(True)
    
    # [삭제] --- Accuracy 그래프 ---
    # ax2.plot(train_accs, 'r-', label='Training Accuracy')
    # ...
    # ----------------------------
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    print("Results saved to training_results.png")

# ----------------------------------------------------
# 4. 메인 함수 (main) 수정
# ----------------------------------------------------
def main():
    """메인 실행 함수"""
    # 하이퍼파라미터
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001 # (Adam은 SGD 0.01보다 작은 LR을 씀)
    MAX_SAMPLES = 10000   # (테스트용, 실제론 1000 이상 권장)
    
    # [수정] 전처리된 데이터 폴더 경로
    TRAIN_DATA_DIR = "./data/train"
    DUMMY_NUM_CLASSES = 100 # (더미 데이터용)
    
    print("=" * 50)
    print("OCR 딥러닝 프로젝트 - NumPy 구현 (2단계: CRNN-CTC)")
    print("=" * 50)
    
    # [수정] --- 데이터셋 로드 및 Vocab 설정 ---
    print("\n[1] Loading dataset...")
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"Warning: {TRAIN_DATA_DIR} not found. Creating dummy data...")
        dataset = create_dummy_dataset(MAX_SAMPLES, DUMMY_NUM_CLASSES)
        num_classes = DUMMY_NUM_CLASSES
    else:
        # 1. Dataset 로드 (이때 vocab.json이 로드/생성됨)
        dataset = OCRDataset(TRAIN_DATA_DIR, max_samples=MAX_SAMPLES)
        # 2. Dataset에서 실제 클래스 개수 가져오기
        num_classes = dataset.get_vocab_size()
    
    print(f"\n[2] Creating model... (NumClasses = {num_classes})")
    # 3. Model에 실제 클래스 개수 주입
    model = OCRModel(num_classes=num_classes)
    
    # 4. (⭐️ 중요) Model의 손실 함수를 CTCLoss로 교체
    model.loss_layer = CTCLoss(blank_label=0) # dataset.py와 약속한대로 0번
    # ----------------------------------------------------
    
    # [수정] 학습
    print("\n[3] Training model...")
    train_losses = train_model( # <-- accs 반환 안 받음
        model, dataset, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        lr=LEARNING_RATE
    )
    
    # [수정] 결과 시각화
    print("\n[4] Plotting results...")
    plot_results(train_losses) # <-- accs 전달 안 함
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    # print(f"Final Accuracy: {train_accs[-1]:.4f}") # <-- [삭제]
    print("=" * 50)


# ----------------------------------------------------
# 5. 더미 데이터 생성 함수 (create_dummy_dataset) 수정
# ----------------------------------------------------
def create_dummy_dataset(num_samples, num_classes):
    """데모용 더미 데이터셋 (2단계 호환)"""
    class DummyDataset:
        def __init__(self, num_samples, num_classes):
            self.num_samples = num_samples
            self.num_classes = num_classes
            
        def __len__(self):
            return self.num_samples
        
        def get_vocab_size(self):
            """모델 초기화를 위한 가짜 vocab size"""
            return self.num_classes
        
        def get_batch(self, batch_size):
            images = np.random.randn(batch_size, 1, 32, 32) * 0.1
            
            # [수정] NumPy 배열이 아닌, 리스트의 리스트 반환
            labels_list = []
            for _ in range(batch_size):
                # (1~num_classes-1 사이의 값, 0은 blank)
                l = np.random.randint(1, 6) # 1~5 길이의 랜덤 시퀀스
                labels = np.random.randint(1, self.num_classes, l).tolist()
                labels_list.append(labels)
                
            return images, labels_list
    
    return DummyDataset(num_samples)


if __name__ == "__main__":
    main()