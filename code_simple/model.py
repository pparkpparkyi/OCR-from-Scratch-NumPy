# model.py
# OCRModel: CNN + RNN 기반 OCR 모델 구현
import numpy as np
from layer import Conv2D, ReLU, MaxPooling2D, Linear, Softmax
from rnn import RNNLayer
from loss import CrossEntropyLoss

class OCRModel:
    """CNN + RNN 기반 OCR 모델"""
    def __init__(self, num_classes=2350):  # 한글 자소 + 특수문자
        # CNN 레이어
        self.conv1 = Conv2D(1, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPooling2D(pool_size=2, stride=2)
        
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPooling2D(pool_size=2, stride=2)
        
        # RNN 레이어 (시퀀스 길이 8 가정)
        self.rnn = RNNLayer(input_size=64*8*8, hidden_size=256)
        
        # 출력 레이어
        self.fc = Linear(256, num_classes)
        self.softmax = Softmax()
        
        # 손실 함수
        self.loss_layer = CrossEntropyLoss()
        
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
        
        # RNN 입력을 위한 reshape
        batch_size = out.shape[0]
        out = out.reshape(batch_size, 1, -1)  # (batch, 1, features)
        
        # RNN 처리
        out = self.rnn.forward(out)
        out = out[:, -1, :]  # 마지막 타임스텝만 사용
        
        # 분류
        out = self.fc.forward(out)
        out = self.softmax.forward(out)
        
        return out
    
    def loss(self, x, t):
        """손실 계산"""
        y = self.forward(x)
        return self.loss_layer.forward(y, t)
    
    def backward(self):
        """역전파"""
        dout = self.loss_layer.backward()
        dout = self.softmax.backward(dout)
        dout = self.fc.backward(dout)
        
        # RNN 역전파
        dout = dout.reshape(dout.shape[0], 1, -1)
        dout = self.rnn.backward(dout)
        
        # CNN 역전파
        batch_size = dout.shape[0]
        dout = dout.reshape(batch_size, 64, 8, 8)
        
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
        """예측"""
        y = self.forward(x)
        return np.argmax(y, axis=1)
