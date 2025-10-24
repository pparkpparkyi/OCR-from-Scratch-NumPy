# main.py
import numpy as np
import matplotlib.pyplot as plt
from model import OCRModel
from dataset import OCRDataset
import os

class SGD:
    """확률적 경사하강법 옵티마이저"""
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad


def train_model(model, dataset, epochs=20, batch_size=32, lr=0.01):
    """모델 학습"""
    optimizer = SGD(lr=lr)
    
    train_losses = []
    train_accs = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        num_batches = len(dataset) // batch_size
        
        for batch in range(num_batches):
            # 배치 데이터 가져오기
            x_batch, t_batch = dataset.get_batch(batch_size)
            
            # 순전파 및 손실 계산
            loss = model.loss(x_batch, t_batch)
            
            # 역전파
            model.backward()
            
            # 파라미터 업데이트
            params, grads = model.get_params_and_grads()
            optimizer.update(params, grads)
            
            # 정확도 계산
            pred = model.predict(x_batch)
            acc = np.mean(pred == t_batch)
            
            epoch_loss += loss
            epoch_acc += acc
            
            if batch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch}/{num_batches}, Loss: {loss:.4f}, Acc: {acc:.4f}")
        
        # 에포크 평균
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        
        print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}\n")
    
    return train_losses, train_accs


def plot_results(train_losses, train_accs):
    """학습 결과 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 그래프
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy 그래프
    ax2.plot(train_accs, 'r-', label='Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    print("Results saved to training_results.png")


def main():
    """메인 실행 함수"""
    # 하이퍼파라미터
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    MAX_SAMPLES = 1000
    
    # 데이터 경로 (실제 경로로 수정 필요)
    TRAIN_DATA_DIR = "./data/1.Training"
    
    print("=" * 50)
    print("OCR 딥러닝 프로젝트 - NumPy 구현")
    print("=" * 50)
    
    # 데이터셋 로드
    print("\n[1] Loading dataset...")
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"Warning: {TRAIN_DATA_DIR} not found. Creating dummy data for demonstration...")
        # 데모용 더미 데이터 생성
        os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
        dataset = create_dummy_dataset(MAX_SAMPLES)
    else:
        dataset = OCRDataset(TRAIN_DATA_DIR, max_samples=MAX_SAMPLES)
    
    # 모델 생성
    print("\n[2] Creating model...")
    model = OCRModel(num_classes=100)  # 단순화된 클래스 수
    
    # 학습
    print("\n[3] Training model...")
    train_losses, train_accs = train_model(
        model, dataset, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        lr=LEARNING_RATE
    )
    
    # 결과 시각화
    print("\n[4] Plotting results...")
    plot_results(train_losses, train_accs)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Final Accuracy: {train_accs[-1]:.4f}")
    print("=" * 50)


def create_dummy_dataset(num_samples):
    """데모용 더미 데이터셋 생성"""
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
        
        def get_batch(self, batch_size):
            images = np.random.randn(batch_size, 1, 32, 32) * 0.1
            labels = np.random.randint(0, 100, batch_size)
            return images, labels
    
    return DummyDataset(num_samples)


if __name__ == "__main__":
    main()
