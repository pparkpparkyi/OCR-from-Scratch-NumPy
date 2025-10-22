# main.py
import numpy as np
import matplotlib.pyplot as plt
from model import OCRModel
from dataset import OCRDataset
from loss import CTCLoss
import os

# ----------------------------------------------------
# 1. 옵티마이저 (Adam)
# ----------------------------------------------------
class Adam:
    """Adam 옵티마이저"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

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
# 2. 학습 및 검증 함수 (train_model) 수정
# ----------------------------------------------------
def train_model(model, train_dataset, valid_dataset, epochs=20, batch_size=32, lr=0.001):
    """모델 학습 및 검증"""
    optimizer = Adam(lr=lr)
    
    train_losses = []
    valid_losses = [] # <-- [추가] 검증 손실 기록용
    
    print("Starting training and validation...")
    
    for epoch in range(epochs):
        # --- 1. Training Phase ---
        model.is_training = True # (혹시 모를 Dropout 등을 위해)
        epoch_train_loss = 0
        num_train_batches = len(train_dataset) // batch_size
        
        for batch in range(num_train_batches):
            x_batch, t_batch = train_dataset.get_batch(batch_size)
            
            loss = model.loss(x_batch, t_batch)
            model.backward()
            
            params, grads = model.get_params_and_grads()
            optimizer.update(params, grads)
            
            epoch_train_loss += loss
            
            if batch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} [Train] Batch {batch}/{num_train_batches}, Loss: {loss:.4f}")
        
        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # --- 2. Validation Phase ---
        model.is_training = False # 평가 모드
        epoch_valid_loss = 0
        num_valid_batches = len(valid_dataset) // batch_size
        
        print(f"\n--- Running Validation for Epoch {epoch+1} ---")
        for batch in range(num_valid_batches):
            x_batch, t_batch = valid_dataset.get_batch(batch_size)
            
            # [⭐️ 중요] forward()만 실행하여 손실 계산
            # backward()와 optimizer.update()는 절대 호출하지 않음!
            loss = model.loss(x_batch, t_batch)
            epoch_valid_loss += loss

        avg_valid_loss = epoch_valid_loss / num_valid_batches
        valid_losses.append(avg_valid_loss)
        
        # --- Epoch 결과 요약 ---
        print(f"Epoch {epoch+1} Summary - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}\n")
    
    return train_losses, valid_losses

# ----------------------------------------------------
# 3. 시각화 함수 (plot_results) 수정
# ----------------------------------------------------
def plot_results(train_losses, valid_losses):
    """학습 및 검증 결과 시각화"""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.plot(valid_losses, 'r-', label='Validation Loss') # <-- [추가]
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss over Epochs')
    ax1.legend()
    ax1.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_and_validation_results.png', dpi=300)
    print(f"Results saved to {os.path.abspath('training_and_validation_results.png')}")

# ----------------------------------------------------
# 4. 메인 함수 (main) 수정
# ----------------------------------------------------
def main():
    """메인 실행 함수"""
    # 하이퍼파라미터
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MAX_SAMPLES_TRAIN = 10000
    MAX_SAMPLES_VALID = 2000
    
    TRAIN_DATA_DIR = "data/train"
    VALID_DATA_DIR = "data/valid" # <-- [추가]
    
    print("=" * 50)
    print("OCR 딥러닝 프로젝트 - NumPy 구현 (CRNN-CTC with Validation)")
    print("=" * 50)
    
    # --- [수정] 데이터셋 로드 (Train/Valid 분리) ---
    print("\n[1] Loading datasets...")
    
    # Train Dataset
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"Fatal: Training directory '{TRAIN_DATA_DIR}' not found. Run preprocess.py first!")
        return
    train_dataset = OCRDataset(TRAIN_DATA_DIR, max_samples=MAX_SAMPLES_TRAIN)
    num_classes = train_dataset.get_vocab_size()

    # Valid Dataset
    if not os.path.exists(VALID_DATA_DIR):
        print(f"Fatal: Validation directory '{VALID_DATA_DIR}' not found. Run preprocess.py first!")
        return
    # [⭐️ 중요] 검증 데이터는 학습 데이터와 '동일한' 문자 사전을 사용해야 함
    valid_dataset = OCRDataset(VALID_DATA_DIR, max_samples=MAX_SAMPLES_VALID)
    valid_dataset.char_to_idx = train_dataset.char_to_idx
    valid_dataset.idx_to_char = train_dataset.idx_to_char
    
    print(f"\n[2] Creating model... (NumClasses = {num_classes})")
    model = OCRModel(num_classes=num_classes)
    model.loss_layer = CTCLoss(blank_label=0)
    
    print("\n[3] Training model...")
    train_losses, valid_losses = train_model(
        model, train_dataset, valid_dataset,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        lr=LEARNING_RATE
    )
    
    print("\n[4] Plotting results...")
    plot_results(train_losses, valid_losses)
    
    print("\n" + "=" * 50)
    print("Training and validation completed!")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Valid Loss: {valid_losses[-1]:.4f}")
    print("=" * 50)

# (더미 데이터 함수는 실제 데이터를 사용하므로 생략 가능)

if __name__ == "__main__":
    main()
