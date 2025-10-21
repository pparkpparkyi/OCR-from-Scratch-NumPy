#rnn.py
#시퀀스 데이터 처리를 위한 RNN구조
# RNN은 멀티 모달 처리
# 과거의 정보를 기억하고 활용
import numpy as np

class SimpleRNN:
    """단순 RNN 레이어 - 시퀀스 데이터 처리에 사용."""
    def __init__(self, input_size, hidden_size):
        #입력 크기, 은닉 상태 크기 설정
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #가중치 초기화
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01  # 입력에서 은닉 상태로 가는 가중치
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01 # 이전 은닉 상태에서 현재 은닉 상태로 가는 가중치
        self.b = np.zeros((1, hidden_size))                        # 편향
        
        #은닉 상태 초기화
        #self.h_prev = np.zeros((1, hidden_size))
        
        self.dWx = None
        self.dWh = None
        self.db = None
        
        self.cache = None
        
    def forward(self, x, h_prev):
        # 한 시점의 기본 동작. tanH 계산만
        """순전파 계산
        x: 입력 데이터 (배치 크기, 입력 크기)
        h_prev: 이전 은닉 상태 (배치 크기, 은닉 상태 크기)
        """
        #현재 은닉 상태 계산
        h_next = np.tanh(np.dot(x, self.Wx) + np.dot(h_prev, self.Wh) + self.b)
        
        #캐시에 값 저장 (역전파에 사용)
        self.cache = (x, h_prev, h_next)
        
        return h_next
    
    def backward(self, dh_next):
        # 한 시점의 오차전파. tanh 미분 포함
        """역전파"""
        x, h_prev, h_next = self.cache
        
        #tanh 미분
        dt = dh_next * (1 - h_next ** 2)
        
        #가중치 기울기 계산
        self.db = np.sum(dt,axis =0)
        self.dWx = np.dot(x.T, dt)
        self.dWh = np.dot(h_prev.T, dt)
        
        #이전 은닉 상태와 입력에 대한 기울기 계산
        dx = np.dot(dt, self.Wx.T)
        dh_prev = np.dot(dt, self.Wh.T)

        return dx, dh_prev

class RNNLayer:
    """다중 타임스텝 RNN 레이어"""
    def __init__(self, input_size, hidden_size):
        self.rnn = SimpleRNN(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.h = None
        self.dh = None
        
    def forward(self, xs):
        """
        xs: 입력 데이터 (배치 크기, 타임스텝 수, 입력 크기)
        return : 은닉 상태 시퀀스 (배치 크기, 타임스텝 수, 은닉 상태 크기)
        """
        batch_size, time_steps, _ = xs.shape
        hs = np.zeros((batch_size, time_steps, self.hidden_size))
        
        h = np.zeros((batch_size, self.hidden_size)) #초기 은닉 상태
        
        for t in range(time_steps):
            h = self.rnn.forward(xs[:, t, :], h)
            hs[:, t, :] = h
        
        self.h = hs
        return hs
    
    def backward(self, dhs):
        """
        dhs: 은닉 상태 시퀀스에 대한 기울기 (배치 크기, 타임스텝 수, 은닉 상태 크기)
        return : 입력 데이터에 대한 기울기 (배치 크기, 타임스텝 수, 입력 크기)
        """
        batch_size, time_steps, _ = dhs.shape
        dxs = np.zeros((batch_size, time_steps, self.rnn.input_size))
        
        dh = np.zeros((batch_size, self.hidden_size))
        
        for t in reversed(range(time_steps)):
            dh = dhs[:, t, :] + dh
            dx, dh = self.rnn.backward(dh)
            dxs[:, t, :] = dx

        self.dh = dh
        return dxs