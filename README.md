# OCR-from-Scratch-NumPy
**NumPy만을 사용하여 구현한 딥러닝 기반 한글 손글씨 OCR (CRNN + CTC)**  

---

## 📘 프로젝트 개요
이 프로젝트는 **TensorFlow/PyTorch 등의 프레임워크 없이**,  
**NumPy만으로 딥러닝 OCR 모델을 직접 구현**한 학습형 프로젝트입니다.  

모델은 **CRNN (Convolutional Recurrent Neural Network)** 구조를 기반으로 하며,  
CNN으로 시각적 특징을 추출하고, RNN으로 시퀀스 문맥을 해석하며,  
**CTCLoss(Connectionist Temporal Classification)** 로 문자 정렬 문제를 해결합니다.

---

## 🧩 핵심 목표
- ✅ 딥러닝 기본 연산 (Conv2D, MaxPool, RNN, Linear) 직접 구현  
- ✅ Forward / Backward Propagation 전 과정 NumPy로 작성  
- ✅ CTCLoss 직접 구현을 통한 Alignment-Free 학습 이해  
- ✅ AI Hub 손글씨 데이터 전처리 및 시퀀스 학습

---

## 📂 프로젝트 구조

```

OCR-from-Scratch-NumPy/
│
├── code/
│   ├── dataset.py              # 데이터 로딩 및 시퀀스 라벨 처리
│   ├── layer.py                # Conv2D, ReLU, Pooling, Linear 레이어
│   ├── rnn.py                  # RNN Layer (BPTT 포함)
│   ├── model.py                # CNN + RNN + CTC 전체 구조
│   ├── loss.py                 # CTCLoss 구현
│   ├── preprocess.py           # AI Hub 데이터 전처리 (bbox crop)
│   ├── main.py                 # 학습 및 평가 스크립트
│   ├── img_diagram.py          # 모델 구조 다이어그램
│   └── verify_preprocess.py    # 전처리 결과 검증
│
├── 손글씨OCR_데이터설명서.pdf
├── 손글씨OCR_데이터 구축 가이드라인.pdf
└── README.md

```

---

## 🧠 모델 개요

| 구성 | 역할 | 출력 형태 |
|------|------|------------|
| **CNN** | 이미지에서 시각적 특징 추출 | (N, 64, 8, 8) |
| **Reshape** | 2D → 1D 시퀀스 변환 | (N, 8, 512) |
| **RNN** | 순차적 문맥 이해 | (N, 8, 256) |
| **Linear** | 각 타임스텝 문자 확률 계산 | (N, 8, NumClasses) |
| **CTCLoss** | 정렬 문제 해결 | Loss 값 계산 |

> CTCLoss는 blank 토큰(`'_'`)을 사용하여  
> 모델 출력과 실제 문자 시퀀스를 동적으로 정렬합니다.

---

## 🧮 데이터 전처리 (preprocess.py)

AI Hub 손글씨 OCR 데이터 중 `"수집 매체: 종이, 내용: 자유필사"` 샘플만 사용.  

1. JSON 라벨의 BBox 좌표를 기준으로 글자 단위 crop  
2. Grayscale 변환 및 (32×32) 리사이즈  
3. 이미지(`data/train/images/`)와 라벨(`data/train/labels/`)로 저장  

```
data/train/
├── images/
│    ├── sample_000_bbox_000.png
│    ├── sample_000_bbox_001.png
│    └── ...
└── labels/
├── sample_000_bbox_000.txt
├── sample_000_bbox_001.txt
└── ...
```

---

## ⚙️ 실행 방법

```
# 1️⃣ 데이터 전처리
python preprocess.py

# 2️⃣ 학습 시작
python main.py

# 3️⃣ 결과 확인
# - 학습 로그 출력
# - Loss 수렴 그래프 확인
```

> ⚠️ 주의: GPU 및 딥러닝 프레임워크 사용 금지.
> 모든 연산은 NumPy로만 수행됩니다.

---

## 🧾 주요 특징

* CNN–RNN–CTC 전체 구조를 **NumPy로 직접 구현**
* CTCLoss의 Forward / Backward 연산 직접 구현
* im2col, col2im 기반의 효율적 Conv2D 연산
* 학습/전처리/시각화 모듈 완전 분리 설계
* AI Hub 한글 손글씨 OCR 데이터셋 완전 호환

---

## 📈 학습 결과 (예시)

| 항목     | 결과                                       |
| ------ | ---------------------------------------- |
| 손실값    | 초기 2.3 → 0.4까지 감소                        |
| 시퀀스 출력 | `[‘ㄷ’, ‘ㅐ’, ‘ㄱ’, ‘ㅜ’]` → Collapse 후 "대구" |
| 정렬 성능  | 길이가 다른 시퀀스에서도 안정적 수렴                     |

---

## 🧩 참고자료

* [AI Hub 대용량 손글씨 OCR 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=605)
* Graves A. et al., *Connectionist Temporal Classification*, ICML 2006
* Shi B. et al., *An End-to-End Trainable Neural Network for Scene Text Recognition*, TPAMI 2017

---

## 👨‍💻 Author

**박서진 (Seojin Park)**
숭실대학교 산업정보시스템공학과 · 컴퓨터학부
📧 E-mail: [parkkpparkyi@gmail.com](mailto:parkkpparkyi@gmail.com)

---

⭐ **학습 목적:** “딥러닝을 프레임워크 없이 완전히 이해하고 구현하기.”

> → “코드로 수식을 구현한 진짜 End-to-End OCR.”
