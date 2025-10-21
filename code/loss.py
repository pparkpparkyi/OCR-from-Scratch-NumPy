# loss.py
import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.y = None
        self.t = None
    
    def forward(self, y, t):
        """Compute the cross-entropy loss.
        y: Predicted probabilities (batch_size, num_classes)
        t: True labels (batch_size, num_classes) - one-hot encoded
        """
        self.y = y
        self.t = t
        
        batch_size = y.shape[0]
        
        
        #수치 안정성을 위한 처리
        y_shifted = y - np.max(y, axis=1, keepdims=True)
        log_y = y_shifted - np.log(np.sum(np.exp(y_shifted), axis=1, keepdims=True))

        loss = -np.sum(log_y[np.arange(batch_size), t]) / batch_size
        return loss
    
    def backward(self):
        """역전파"""
        
        batch_size = self.y.shape[0]
        
        #Soft max와 교차 엔트로피 손실의 결합으로 인한 그래디언트 계산
        y_shifted = self.y -np.max(self.y, axis=1, keepdims=True)
        softmax = np.exp(y_shifted) / np.sum(np.exp(y_shifted), axis=1, keepdims=True)
        
        dx = softmax.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx / batch_size
        
        return dx

class SimplifiedCTCLoss:
    """A simplified version of CTC Loss for demonstration purposes."""
    # Note: This is a highly simplified version and does not cover all edge cases.
    def __init__ (self, blank_label):
        self.blank_label = blank_label
        self.ce_loss = CrossEntropyLoss()
    
    def forward(self, y_seq, t_seq):
        """Compute a simplified CTC loss.
        y_seq: Predicted probabilities (time_steps, num_classes)
        t_seq: True labels (list of integers)
        """
        # For simplicity, we will just compute cross-entropy loss at each time step
        batch_seize, seq_len, num_classes = y_seq.shape
        #각 타임스템의 손실 평균
        total_loss = 0
        for t in range(min(seq_len, t_seq.shape[1])):
            loss = self.ce_loss.forward(y_seq[:, t, :], t_seq[:, t])
            total_loss += loss
        return total_loss / min(seq_len, t_seq.shape[1])
    
    def backward(self):
        """역전파"""
        return self.ce_loss.backward()