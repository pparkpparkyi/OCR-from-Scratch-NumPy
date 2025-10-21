## 1\. 파일 구조 및 역할

총 5개의 Python 파일(`.py`)과 1개의 실행 파일(`main.py`)로 구성하는 것을 권장합니다.

```
/DL_OCR_Project
|
+-- layer.py             # 기본 레이어 (Conv, ReLU, Pool, Linear) 정의
+-- rnn.py               # RNN/LSTM 시퀀스 레이어 정의
+-- loss.py              # CTC Loss 함수 정의
+-- model.py             # 전체 CNN-RNN 모델 구조 정의 및 조합
+-- dataset.py           # 데이터 로딩 및 전처리 (이미지 로드, 라벨 인코딩)
+-- main.py              # 학습(Training) 및 평가(Evaluation) 실행
```

-----

## 2\. 파일별 코드 설명 (Classes & Methods)

### 1\. `dataset.py`: 데이터 로딩 및 전처리

| 클래스/함수 | 설명 | 주요 Numpy 작업 |
| :--- | :--- | :--- |
| `CharTokenizer` | 텍스트 라벨을 정수 시퀀스로 변환 | 어휘 집합(Vocabulary) 구축 및 텍스트 $\leftrightarrow$ 인덱스 매핑 테이블 생성. CTC용 **Blank 토큰** 추가. |
| `OCRDataset` | 이미지와 라벨을 로드하고 전처리 | 1. **PIL/OpenCV**를 이용한 이미지 로드 및 **Numpy 배열** 변환. 2. 이미지 **Grayscale 변환**, **정규화**($0 \sim 1$). 3. 이미지 **크롭/리사이즈/패딩**으로 통일. 4. `CharTokenizer`로 텍스트 라벨 인코딩 및 패딩. |
| `DataLoader` | 배치(Batch) 단위로 데이터를 제공 | `OCRDataset` 객체를 받아 데이터를 미니배치 단위로 묶어 제공하며, 필요시 데이터 셔플(Shuffle) 기능 구현. |

### 2\. `layer.py`: 핵심 딥러닝 레이어 (Numpy 전용)

| 클래스 | 메서드 | 설명 |
| :--- | :--- | :--- |
| `Layer` (Base) | `forward(X)`, `backward(dL_dA)` | 모든 레이어의 기본 틀 (상속용). |
| `Conv2D` | `forward(X)`, `backward(dL_dZ)` | **합성곱** 연산 구현. `im2col` 또는 직접 루프를 사용한 효율적인 순전파/역전파, 가중치/편향 업데이트용 기울기 계산. |
| `ReLU` | `forward(Z)`, `backward(dL_dA)` | $A = \max(0, Z)$ 순전파 및 $Z>0$ 일 때만 기울기 전달하는 역전파 구현. |
| `MaxPool` | `forward(X)`, `backward(dL_dZ)` | 풀링 영역 내 **최댓값** 추출 및 순전파 시 최댓값 위치를 저장하여 역전파 때 사용. |
| `Linear` (Dense) | `forward(X)`, `backward(dL_dZ)` | $Z = XW + b$ 행렬 곱셈 순전파 및 $W, b, X$에 대한 기울기 계산. |
| `Softmax` | `forward(Z)`, `backward(dL_dA)` | 최종 출력에서 각 클래스(문자) 확률 계산. (보통 Cross-Entropy Loss와 결합되지만, 여기서는 CTC와 함께 사용). |

### 3\. `rnn.py`: 시퀀스 처리 레이어 (Numpy 전용)

| 클래스 | 메서드 | 설명 |
| :--- | :--- | :--- |
| `LSTMCell` | `forward(X_t, h_t_prev, c_t_prev)` | 하나의 시퀀스 스텝($t$)에서 **LSTM의 게이트 및 상태 업데이트** 로직 구현. (입력, 망각, 출력, 셀 게이트) |
| `LSTM` | `forward(X)`, `backward(dL_dA)` | 전체 시퀀스에 대해 `LSTMCell`을 시간 축으로 반복 적용(`for` 루프). \*\*BPTT (Backpropagation Through Time)\*\*를 위한 역전파 구현. |

### 4\. `loss.py`: 손실 함수 (Numpy 전용)

| 클래스/함수 | 메서드 | 설명 |
| :--- | :--- | :--- |
| `CTCLoss` | `forward(Y, T)`, `backward()` | **CTC (Connectionist Temporal Classification) Loss** 계산. |
| | | 1. **Forward-Backward** 알고리즘을 이용한 $\log(\text{likelihood})$ (손실값) 계산. |
| | | 2. **알파-베타 경로**를 이용해 출력($Y$)에 대한 최종 기울기 $\partial L/\partial Y$ 계산. (Numpy 구현의 핵심 난이도) |

### 5\. `model.py`: 모델 구조 결합

| 클래스 | 메서드 | 설명 |
| :--- | :--- | :--- |
| `OCRModel` | `__init__`, `forward(X)`, `update_params()` | `layer.py`의 `Conv2D`, `ReLU`, `MaxPool`과 `rnn.py`의 `LSTM`, `Linear` 레이어를 **순서대로 조합**하여 전체 CNN-RNN 아키텍처 정의. |

### 6\. `main.py`: 실행 파일

| 함수 | 설명 |
| :--- | :--- |
| `train()` | **데이터 로드, 모델 초기화, 옵티마이저 정의.** 학습 루프(`for epoch`): 순전파, `CTCLoss` 계산, 역전파, 파라미터 업데이트(`Optimizer`). |
| `evaluate()` | `2.Validation` 데이터셋을 사용하여 최종 성능 측정. **CTC 디코딩** (예: Best Path Decodng)을 구현하여 예측된 텍스트와 실제 라벨 비교. |

**주의:** Numpy로 구현할 때는 `main.py`에 별도의 **`Optimizer` 클래스** (예: `SGD` 또는 `Adam`)를 정의하여 `model.py`에서 계산된 기울기를 바탕으로 가중치를 업데이트하는 로직을 구현해야 합니다.