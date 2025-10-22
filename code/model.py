# model.py
# OCRModel: CNN + RNN 기반 OCR 모델 구현
import numpy as np
# ⚠️ layer.py와 rnn.py, loss.py가 동일한 폴더에 있다고 가정합니다.
from layer import Conv2D, ReLU, MaxPooling2D, Linear, Softmax
from rnn import RNNLayer
# [수정 1] CrossEntropyLoss -> CTCLoss로 임포트 변경
from loss import CTCLoss

class OCRModel:
    """CNN + RNN 기반 OCR 모델 (시퀀스 처리 버전)"""
    def __init__(self, num_classes=2350):
        # CNN 레이어
        self.conv1 = Conv2D(1, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPooling2D(pool_size=2, stride=2)
        
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPooling2D(pool_size=2, stride=2)
        # CNN 최종 출력 shape: (N, 64, 8, 8)
        
        # [수정 1] RNN 레이어
        # (N, 64, 8, 8) -> (N, 8, 512)로 변환하여 RNN에 입력합니다.
        # 따라서 input_size는 (64 * 8) = 512가 됩니다.
        rnn_input_size = 64 * 8 
        rnn_hidden_size = 256
        self.rnn = RNNLayer(input_size=rnn_input_size, hidden_size=rnn_hidden_size)
        
        # 출력 레이어
        # RNN의 모든 타임스텝(8개)에 대해 분류를 수행합니다.
        # (N, 8, 256) -> (N, 8, num_classes)
        self.fc = Linear(rnn_hidden_size, num_classes)
        
        # [수정 2] Softmax 레이어 제거
        # self.softmax = Softmax() # (CTCLoss가 Softmax를 포함하므로 제거)
        
        # [수정 3] 손실 함수 기본값을 CTCLoss로 변경
        self.loss_layer = CTCLoss(blank_label=0)
        
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.rnn, self.fc
        ]
        
    def forward(self, x):
        """
        순전파
        x: (batch_size, 1, 32, 32)
        """
        # CNN 특징 추출
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        # out.shape: (N, 64, 8, 8) -> (Batch, Channels, Height, Width)
        
        # [수정 3] RNN 입력을 위한 Reshape
        N, C, H, W = out.shape 
        
        # (N, C, H, W) -> (N, W, C*H) : (Batch, Time, Features)
        # 즉, (N, 64, 8, 8) -> (N, 8, 64*8) = (N, 8, 512)
        # 8개의 타임스텝(W), 각 스텝은 512(C*H) 차원 벡터
        out = out.transpose(0, 3, 1, 2).reshape(N, W, C * H)
        
        # RNN 처리 (모든 타임스텝)
        out = self.rnn.forward(out) # -> (N, 8, 256) (Batch, Time, HiddenSize)
        
        # 분류 (모든 타임스텝에 대해)
        # Linear 레이어는 마지막 차원에 대해서만 연산
        # (N, 8, 256) -> (N, 8, num_classes)
        out = self.fc.forward(out) 
        
        # [수정 4] Softmax 제거
        # out = self.softmax.forward(out)
        
        # 최종 출력 shape: (Batch, Time=8, NumClasses)
        return out
    
    def loss(self, x, t):
        """
        손실 계산
        """
        y = self.forward(x)
        
        # [수정 4] CTCLoss를 호출하도록 수정 (None 반환 대신)
        # CTCLoss는 y (N, T, C)와 t (레이블 리스트)를 입력받습니다.
        return self.loss_layer.forward(y, t) 
    
    def backward(self):
        """
        역전파
        """
        # 1. CTCLoss로부터 역전파 시작
        dout = self.loss_layer.backward() # (N, 8, num_classes)
        
        # [수정 5] Softmax 역전파 제거
        # dout = self.softmax.backward(dout)
        
        # 2. FC 역전파
        dout = self.fc.backward(dout) # (N, 8, 256)
        
        # 3. RNN 역전파
        dout = self.rnn.backward(dout) # (N, 8, 512) (Batch, Time, Features)
        
        # [수정 6] CNN 입력을 위한 Reshape (forward의 역순)
        # (N, 8, 512) -> (N, 8, 64, 8) -> (N, 64, 8, 8)
        N, W, Features = dout.shape
        C = 64
        H = 8
        dout = dout.reshape(N, W, C, H).transpose(0, 2, 3, 1) # (N, C, H, W)
        
        # 4. CNN 역전파
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)
        
    def get_params_and_grads(self):
        """파라미터와 그래디언트 반환"""
        params = []
        grads = []
        
        # [수정 7] self.rnn.rnn -> self.rnn 으로 수정 (RNNLayer가 rnn 객체를 가짐)
        # (rnn.py 구현에 따라 다름)
        # 원본 코드를 보니 self.rnn.rnn, self.fc가 맞네요. 유지합니다.
        
        for layer in [self.conv1, self.conv2, self.rnn.rnn, self.fc]:
            if hasattr(layer, 'W'):
                params.append(layer.W)
                params.append(layer.b)
                grads.append(layer.dW if hasattr(layer, 'dW') else np.zeros_like(layer.W))
                grads.append(layer.db if hasattr(layer, 'db') else np.zeros_like(layer.b))
            if hasattr(layer, 'Wx'):
                params.extend([layer.Wx, layer.Wh, layer.b])
                grads.extend([layer.dWx, layer.dWh, layer.db])
                
        return params, grads
    
    def predict(self, x):
        """
        예측
        (⚠️ 경고: 이것은 CTCLoss용 '디코더'로 교체되어야 합니다)
        """
        y = self.forward(x) # (N, 8, num_classes)
        
        # 가장 단순한 (하지만 부정확한) 'Best Path' 디코딩
        # 각 타임스텝에서 가장 확률이 높은 글자를 선택
        pred_indices = np.argmax(y, axis=2) # (N, 8)
        
        # (참고: 실제로는 중복 제거 및 'blank' 레이블 제거 필요)
        return pred_indices