# 📂 code — Implementation Details for OCR-from-Scratch-NumPy

이 폴더는 **딥러닝 프레임워크 없이 NumPy만으로 구현한 OCR 모델**의  
전체 학습 파이프라인을 구성하는 소스 코드 모음입니다.

---

## 🧱 전체 구성

| 파일명 | 설명 |
|--------|------|
| `preprocess.py` | AI Hub 손글씨 OCR 원본 ZIP 데이터를 BBox 단위로 크롭하여 `(32×32)` 크기의 학습용 이미지·라벨 세트로 변환합니다. |
| `dataset.py` | 전처리된 이미지를 불러오고, `.txt` 라벨을 **정수 시퀀스**로 변환합니다. (CTCLoss용 blank token 포함) |
| `layer.py` | Conv2D, ReLU, MaxPooling, Linear 등 **기초 신경망 레이어**를 NumPy로 직접 구현합니다. |
| `rnn.py` | Simple RNN (Recurrent Neural Network) 레이어와 BPTT(Backpropagation Through Time)를 직접 구현합니다. |
| `model.py` | CNN + RNN + Linear 구조로 구성된 **CRNN (Convolutional Recurrent Neural Network)** 모델 정의. |
| `loss.py` | **CTCLoss(Connectionist Temporal Classification)** 를 NumPy 기반 동적 프로그래밍(DP)으로 구현합니다. |
| `main.py` | 모델 학습 및 평가 스크립트. Optimizer, 손실 계산, 역전파, 파라미터 업데이트 등을 수행합니다. |
| `img_diagram.py` | 모델 구조(입력 → CNN → RNN → Linear → CTC) 시각화용 다이어그램 생성 코드입니다. |
| `verify_preprocess.py` | `preprocess.py` 결과 이미지·라벨 쌍의 정상 여부를 검사하는 유효성 검증 스크립트입니다. |

---

## ⚙️ 실행 순서

1️⃣ **데이터 전처리**
```bash
python preprocess.py
```

→ `data/train/images/` & `data/train/labels/` 생성

2️⃣ **데이터셋 로딩**

```python
from dataset import OCRDataset
dataset = OCRDataset('./data/train')
x_batch, y_batch = dataset.get_batch(32)
```

3️⃣ **모델 학습**

```bash
python main.py
```

→ CNN + RNN + CTC 기반 모델 학습 시작
→ Epoch별 loss 출력 및 수렴 확인

4️⃣ **결과 확인**

* 학습 로그 출력
* CTCLoss 정렬 확인 (blank 제거)
* 모델 예측 → 디코딩 후 “문자열” 출력

---

## 🧩 주요 구현 특징

* **모든 Forward/Backward 연산 NumPy로 직접 구현**
* **im2col / col2im**을 활용한 합성곱 최적화
* **CTCLoss**로 가변 길이 시퀀스 정렬 문제 해결
* **Bridge Layer (Reshape)** 로 CNN의 2D 출력을 RNN 입력 시퀀스로 변환
* **모듈화 설계:** 모든 구성요소가 독립적으로 import 가능

---

## 📈 출력 예시

| 입력 이미지             | 예측 시퀀스            | 디코딩 결과 |
| ------------------ | ----------------- | ------ |
| handwritten_01.png | `[ㄷ, ㅐ, ㄱ, ㅜ, _]` | “대구”   |
| handwritten_02.png | `[ㅅ, ㅜ, _]`       | “수”    |

---

## 📚 참고

* **데이터:** AI Hub 「대용량 손글씨 OCR 데이터셋」
* **구현 논문:**

  * Graves et al., *Connectionist Temporal Classification*, ICML 2006
  * Shi et al., *Scene Text Recognition with CRNN*, TPAMI 2017

---

📍**참고:**
이 `code/` 폴더는 독립적으로 import 가능하며,
`main.py`에서 모든 모듈을 통합하여 학습 파이프라인을 실행합니다.
