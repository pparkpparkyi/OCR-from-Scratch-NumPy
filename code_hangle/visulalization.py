# visualize.py
# (학습은 하지 않고, 모델 예측 시각화만 수행)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from model import OCRModel
from dataset import OCRDataset
from loss import CTCLoss # (모델이 참조하므로 임포트 필요)
import os

# ----------------------------------------------------
# 1. 헬퍼 함수 (main.py에서 복사)
# ----------------------------------------------------
def find_korean_font():
    """시스템에 설치된 한글 폰트를 찾아서 반환합니다."""
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font_file in font_files:
        if 'malgun' in font_file.lower(): return font_file
    for font_file in font_files:
        if 'apple' in font_file.lower() and 'gothic' in font_file.lower(): return font_file
    for font_file in font_files:
        if 'nanum' in font_file.lower() and 'gothic' in font_file.lower(): return font_file
    return font_files[0] if font_files else None

def ctc_decode(indices, idx_to_char):
    """
    단순한 'Best Path' CTC 디코더.
    [0, 0, 10, 10, 5, 0, 1] -> "대구"
    """
    text = ""
    last_char_idx = -1
    for idx in indices:
        if idx == 0: # 0번은 'blank'
            last_char_idx = -1
            continue
        if idx == last_char_idx: # 중복 제거
            continue
        
        text += idx_to_char.get(idx, '?') # 사전에 없는 글자면 '?'
        last_char_idx = idx
    return text

def show_predictions(model, valid_dataset, num_samples=5):
    """(이미지 2) 실제 예측 결과 시각화"""
    print(f"\n[1] Generating {num_samples} prediction samples...")
    print("    (Note: This model is UNTRAINED, predictions will be random.)")
    
    # 1. 한글 폰트 설정
    font_path = find_korean_font()
    if font_path:
        font_prop = fm.FontProperties(fname=font_path, size=12)
        print(f"Using Korean font: {os.path.basename(font_path)}")
    else:
        font_prop = None
        print("Warning: Korean font not found. Labels may appear broken.")

    # 2. 샘플 데이터 가져오기
    x_batch, t_batch = valid_dataset.get_batch(num_samples)
    
    # 3. 모델 예측 실행
    # model.predict()는 (N, 8) 형태의 argmax 인덱스를 반환
    pred_indices = model.predict(x_batch) 
    
    # 4. matplotlib으로 시각화
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 4))
    if num_samples == 1: axes = [axes] # 샘플이 1개일 때도 리스트로 만듦

    for i in range(num_samples):
        img = x_batch[i].reshape(32, 32)
        
        # CTC 디코딩
        true_text = ctc_decode(t_batch[i], valid_dataset.idx_to_char)
        pred_text = ctc_decode(pred_indices[i], valid_dataset.idx_to_char)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"True: {true_text}\nPred: {pred_text}", fontproperties=font_prop)

    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300)
    print(f"Prediction results saved to {os.path.abspath('prediction_results.png')}")
    plt.close(fig)

# ----------------------------------------------------
# 2. 메인 실행 함수
# ----------------------------------------------------
def main():
    """메인 실행 함수"""
    MAX_SAMPLES_TRAIN = 10000 # (사전 로드용)
    MAX_SAMPLES_VALID = 2000 # (샘플 추출용)
    
    TRAIN_DATA_DIR = "data/train"
    VALID_DATA_DIR = "data/valid" 
    
    print("=" * 50)
    print("OCR Prediction Visualizer (using UNTRAINED model)")
    print("=" * 50)
    
    # --- 데이터셋 로드 (사전(vocab)이 필요함) ---
    print("\n[A] Loading datasets (to get vocabulary)...")
    
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"Fatal: Training directory '{TRAIN_DATA_DIR}' not found.")
        return
    train_dataset = OCRDataset(TRAIN_DATA_DIR, max_samples=MAX_SAMPLES_TRAIN)
    num_classes = train_dataset.get_vocab_size()

    if not os.path.exists(VALID_DATA_DIR):
        print(f"Fatal: Validation directory '{VALID_DATA_DIR}' not found.")
        return
    valid_dataset = OCRDataset(VALID_DATA_DIR, max_samples=MAX_SAMPLES_VALID)
    valid_dataset.char_to_idx = train_dataset.char_to_idx
    valid_dataset.idx_to_char = train_dataset.idx_to_char
    
    print(f"\n[B] Creating UNTRAINED model... (NumClasses = {num_classes})")
    model = OCRModel(num_classes=num_classes)
    
    # [C] 시각화 실행
    show_predictions(model, valid_dataset, num_samples=5)
    
    print("\n" + "=" * 50)
    print("Visualization completed!")
    print("=" * 50)

def evaluate_model_metrics(model, dataset, vocab, num_samples=100):
    """정량 평가 지표 계산"""
    results = {
        'exact_match': 0,
        'total_chars_correct': 0,
        'total_chars': 0,
        'length_distribution': {}
    }
    
    for i in range(min(num_samples, len(dataset.samples))):
        img, label = dataset.get_sample(i)
        
        # 예측
        img_batch = img.reshape(1, 1, 32, 32)
        pred_logits = model.forward(img_batch)
        pred_indices = np.argmax(pred_logits[0], axis=-1)
        
        # 디코딩
        pred_text = ctc_decode(pred_indices, vocab['idx_to_char'])
        true_text = ''.join([vocab['idx_to_char'][idx] for idx in label])
        
        # 메트릭 업데이트
        if pred_text == true_text:
            results['exact_match'] += 1
        
        # CER 계산
        min_len = min(len(pred_text), len(true_text))
        results['total_chars_correct'] += sum(
            p == t for p, t in zip(pred_text[:min_len], true_text[:min_len])
        )
        results['total_chars'] += len(true_text)
        
        # 길이별 분포
        length_key = f"{len(true_text)}자"
        if length_key not in results['length_distribution']:
            results['length_distribution'][length_key] = {'correct': 0, 'total': 0}
        results['length_distribution'][length_key]['total'] += 1
        if pred_text == true_text:
            results['length_distribution'][length_key]['correct'] += 1
    
    # 최종 계산
    accuracy = results['exact_match'] / num_samples * 100
    cer = (1 - results['total_chars_correct'] / results['total_chars']) * 100
    
    return {
        'Exact Match Accuracy (%)': round(accuracy, 2),
        'Character Error Rate (%)': round(cer, 2),
        'Length Distribution': results['length_distribution']
    }

if __name__ == "__main__":
    main()
