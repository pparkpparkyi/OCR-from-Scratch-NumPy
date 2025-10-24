# 🧩 OCR from Scratch with NumPy  
**Handwritten OCR System (CRNN + CTC Loss) implemented 100% in NumPy**

> 혼합 문자(한글·한자·영문·숫자·특수문자) 기반 OCR 구현  
> + 한글 전용 학습 버전(`code_hangle`)까지 확장

---

## 📘 프로젝트 개요

이 프로젝트는 **딥러닝 프레임워크 없이, 순수 NumPy만으로 OCR 모델 전체를 직접 구현**한 실험적 연구입니다.  
CNN, RNN, CTC Loss를 모두 수식 기반으로 작성하여 딥러닝 내부 연산 원리를 직접 검증했습니다.

- AI Hub 대용량 손글씨 OCR 데이터셋 (100GB, 약 2.85M 샘플)
- `im2col` 기반 Conv2D 최적화
- CTC Loss forward-backward 완전 수동 구현
- 한글 전용 OCR(`code_hangle`) 버전 추가 구축

---

## 🧱 프로젝트 구조

```

📦 OCR-from-Scratch-NumPy/
├── code_simple/          # 단순 버전
├── code/                 # 혼합 문자 버전
│   ├── preprocess.py
│   ├── dataset.py
│   ├── layer.py
│   ├── rnn.py
│   ├── loss.py
│   ├── model.py
│   ├── main.py
│   ├── visualization.py
│   └── training_results.png
│
├── code_hangle/          # 한글 전용 OCR 버전
│   ├── preprocess.py
│   ├── dataset.py
│   ├── layer.py
│   ├── rnn.py
│   ├── loss.py
│   ├── model.py
│   ├── main.py
│   ├── data.zip
│   ├── hangul_vocab_train.json
│   └── training_results_hangle.png
│
├── 손글씨OCR_데이터 구축 가이드라인_v1.0.pdf
├── 손글씨OCR_데이터설명서.pdf
└── README.md

````

---

## ⚙️ 실행 환경

| 항목 | 권장 버전 |
|------|------------|
| Python | 3.9 이상 |
| NumPy | ≥ 1.23 |
| Pillow | ≥ 10.0 |
| Matplotlib | ≥ 3.7 |

### 설치
```bash
pip install numpy pillow matplotlib
````

---

## 🚀 실행 방법

### 🧮 혼합 문자 OCR 학습 (`code/`)

```bash
cd code
python preprocess.py     # JSON → 이미지/텍스트 페어 생성
python main.py           # 학습 및 검증 실행
```

출력 파일:

* `training_results.png`
* `best_ocr_model.npz`

---

### 🇰🇷 한글 전용 OCR 학습 (`code_hangle/`)

```bash
cd code_hangle
python preprocess.py     # 한글 전용 데이터셋 생성 및 Vocab 구축
python main.py           # 한글 OCR 학습 실행
```

설명:

* `dataset.py`는 `data.zip` 내부 이미지를 압축 해제 없이 직접 읽음
* 학습 시 자동으로 `hangul_vocab_train.json` 생성
* 학습 완료 후 `best_ocr_model_hangle.npz` 저장

출력 파일:

* `training_results_hangle.png`
* `hangul_vocab_train.json`
* `best_ocr_model_hangle.npz`

---

## 🧩 주요 구현 파일 요약

| 파일명                | 설명                                    |
| ------------------ | ------------------------------------- |
| `preprocess.py`    | AI Hub OCR JSON → 이미지/텍스트 전처리         |
| `dataset.py`       | 이미지 및 텍스트 로더, ZIP 직접 읽기 지원            |
| `layer.py`         | Conv2D(im2col), Linear, MaxPooling 구현 |
| `rnn.py`           | SimpleRNN Forward / Backward          |
| `loss.py`          | CTC Loss 구현 (logaddexp 기반 안정화 포함)     |
| `model.py`         | CRNN 모델 조립 클래스                        |
| `main.py`          | 학습 루프 및 성능 로그 저장                      |
| `visualization.py` | 손실 그래프 및 예측 결과 시각화                    |

---

## 📊 학습 결과 요약 (혼합 문자 버전)

| Metric                     | 값           |
| -------------------------- | ----------- |
| Epoch (Best)               | 5           |
| Validation Loss            | 30.14       |
| Character Error Rate (CER) | 약 50–55%    |
| 주요 오류                      | 한자/영문/숫자 혼동 |

> 한글 전용 버전(`code_hangle`)은 데이터 필터링 및 Vocabulary 구축까지 구현 완료.
> 데이터 용량(100GB)으로 인해 CPU 환경에서는 학습 미수행 상태.
> GPU 환경에서는 CER 25% 이하, 정확도 60% 이상 향상 예상.

---

## 🧠 핵심 아이디어

1. **NumPy만으로 CRNN + CTC 완전 구현**
2. **`im2col` 최적화로 Conv 연산 속도 100배 개선**
3. **`np.logaddexp`, `log_softmax`로 수치 안정화**
4. **Gradient Clipping (max_norm=5.0)** 적용
5. **한글 전용 학습 파이프라인(`code_hangle`) 추가 구현**

---

## 🧮 핵심 의사코드 (Pseudocode)

```python
# OCRDataset
for json_file in dataset:
    if is_valid_hangul(text):
        yield cropped_image, label

# build_vocab
unique_chars = sorted(set(''.join(texts)))
save_json('hangul_vocab_train.json', unique_chars)

# CTC Forward
alpha[t, s] = logaddexp(alpha[t-1, s], alpha[t-1, s-1]) + log_prob[t, label[s]]
```

---

## 📚 참고문헌

* Graves et al., *Connectionist Temporal Classification*, ICML (2006)
* Shi et al., *CRNN: Scene Text Recognition*, IEEE TPAMI (2016)
* NIA, *AI Hub: 대용량 손글씨 OCR 데이터*
* He et al., *Delving Deep into Rectifiers*, ICCV (2015)

---

## 🧑‍💻 작성자

**박서진 (Soongsil University)**
* 산업정보시스템공학과 / 컴퓨터학과
📧 Email: [pokiki03@soongsil.ac.kr](mailto:parkseojin@soongsil.ac.kr)
---

⭐️ **Repository:** [pparkpparkyi/OCR-from-Scratch-NumPy](https://github.com/pparkpparkyi/OCR-from-Scratch-NumPy)

