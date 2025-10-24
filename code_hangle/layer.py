# layer.py
# Convolutional Layer Implementation
# im2col 기법 사용
import numpy as np
class Conv2D: # 이미지 입력을 "필터"로 훑는 연산. 효율을 위해 IM2COL 기법 사용 행렬곱 계산
    """2D Convolutional Layer using im2col method."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1): #왜 이렇게 정했는지
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        #He 초기화
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0                                                                                                / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        
        self.dW = None
        self.db = None
        self.x = None
        self.col = None
        self.col_W = None
    
    def forward(self, x):
        """Forward: im2cold convolution operation."""
        self.x = x
        N, C, H, W = x.shape # batch size, channels, height, width
        FN,C,FH,FW = self.W.shape # filter number(#output channels), channels, filter height, filter width
        """입력: (N=1, C=3, H=5, W=5)
            ↓   필터 16개 적용
            가중치: (FN=16, C=3, FH=3, FW=3)
            ↓
            출력: (N=1, FN=16, out_H, out_W)
        """
        
        # 출력 크기 계산
        out_h = ( H +2*self.padding - FH ) // self.stride + 1
        out_w = ( W +2*self.padding - FW ) // self.stride + 1
        
        # im2col 변환
        col = self.im2col(x, FH, FW, self.stride, self.padding) # (N*out_h*out_w, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T # (C*FH*FW, FN)
        
        out = np.dot(col, col_W) + self.b # (N*out_h*out_w, FN)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # (N, FN, out_h, out_w)
        
        self.col = col
        self.col_W = col_W
        
        return out
    
    def backward(self, dout):
        """Backward: Gradient calculation."""
        FN, C, FH, FW = self.W.shape
        
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) # (N*out_h*out_w, FN)
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout) # (C*FH*FW, FN)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        
        dcol = np.dot(dout, self.col_W.T) # (N*out_h*out_w, C*FH*FW)
        dx = self.col2im(dcol, self.x.shape, FH, FW, self.stride, self.padding)
        
        return dx
    
    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        """이미지를 2차원 배열로 변환"""
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h) // stride + 1
        out_w = (W + 2*pad - filter_w) // stride + 1
        
        img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
        
        col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
        return col
    
    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        """2차원 배열을 이미지로 변환"""
        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h) // stride + 1
        out_w = (W + 2*pad - filter_w) // stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)
        
        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        
        return img[:, :, pad:H + pad, pad:W + pad]
        
    
class ReLU:
    """ReLU Activation Function"""
    #Relu 클래스는 mask라는 이스턴스 변수를 가집니다. mask는 True/False로 구성된 넘파이 배열로, 순전파의 입력인 x의 원소값이 0이하인 인덱스는 True, 그 왜에는 False.
    # 역전파 때는 순전파 때 만들어둔 mask를 써서 mask가 True인 곳에 상류에서 전파된 dout을 0으로 설정함.
    def __init__(self):
        self.mask = None
    
    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
class MaxPooling2D:
    """최대 풀링 레이어"""
    def __init__(self, pool_size=2, stride = 2):
        self.pool_h = pool_size
        self.pool_w = pool_size
        self.stride = stride
        self.x = None
        self.arg_max = None
    def forward(self, x):
        N, C, H, W = x.shape
        out_h= int(1 + (H - self.pool_h) / self.stride)
        out_w= int(1 + (W - self.pool_w) / self.stride)
        col = self.im2col(x, self.pool_h, self.pool_w, self.stride, 0)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        self.x = x
        self.arg_max = arg_max
        
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = self.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, 0)
        
        return dx
    
    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h) // stride + 1
        out_w = (W + 2*pad - filter_w) // stride + 1
        
        img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
        
        col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
        return col
    
    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h) // stride + 1
        out_w = (W + 2*pad - filter_w) // stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)
        
        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        
        return img[:, :, pad:H + pad, pad:W + pad]

class Linear: # y = xW + b
    """ Fully Connected Layer """
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    # layer.py의 Linear 클래스에 붙여넣을 수정된 backward 함수

    def backward(self, dout):
        """
        [수정됨] 3D 텐서 (N, T, C) 입력을 처리하도록 수정
        """
        # dout shape: (N, T, out_features) - 예: (32, 8, 3812)
        # self.x shape: (N, T, in_features) - 예: (32, 8, 256)
        # self.W shape: (in_features, out_features) - 예: (256, 3812)

        # 1. dx 계산 (dLoss/dx)
        # (N, T, out) @ (out, in) = (N, T, in)
        # 이 부분은 3D에서도 np.dot이 올바르게 작동합니다.
        dx = np.dot(dout, self.W.T)

        # 2. dW 계산 (dLoss/dW)
        # (N, T, in) -> (N*T, in)
        x_reshaped = self.x.reshape(-1, self.W.shape[0]) 
        # (N, T, out) -> (N*T, out)
        dout_reshaped = dout.reshape(-1, self.W.shape[1]) 
        
        # (in, N*T) @ (N*T, out) = (in, out)
        self.dW = np.dot(x_reshaped.T, dout_reshaped)

        # 3. db 계산 (dLoss/db)
        # (N, T, out) -> (out,)
        # Batch(0)와 Time(1) 축 모두에 대해 합산해야 합니다.
        self.db = np.sum(dout, axis=(0, 1))
        
        return dx

class Softmax: # 분류 문제에서 마지막에 확률로 바꾸기 위한 함수
    """ Softmax Activation Function"""
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        x = x - np.max(x, axis=1, keepdims=True)  # Overflow 방지
        # 출력값을 확률로 바꾸는 단계
        exp_x = np.exp(x)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out
    
    def backward(self, dout):
        return dout  # Softmax와 Cross-Entropy Loss를 함께 사용할 때는 별도의 backward 구현이 필요 없습니다.