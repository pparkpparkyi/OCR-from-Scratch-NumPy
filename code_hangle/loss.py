# loss.py (CTCLoss 구현)
import numpy as np

def log_softmax(x):
    """
    수치적으로 안정적인 LogSoftmax 계산 (Overflow 방지)
    x shape: (N, T, C)
    """
    max_x = np.max(x, axis=2, keepdims=True)
    e_x = np.exp(x - max_x)
    sum_e_x = np.sum(e_x, axis=2, keepdims=True)
    epsilon = 1e-12 
    return (x - max_x) - np.log(sum_e_x + epsilon)


class CTCLoss:
    """Connectionist Temporal Classification (CTC) Loss"""
    def __init__(self, blank_label=0):
        self.blank_label = blank_label
        self.log_probs = None 
        self.labels_list = None 
        self.input_lengths = None 
        self.label_lengths = None 
        self.batch_size = None
        
    def forward(self, y_pred, labels_list):
        self.batch_size, T, C = y_pred.shape
        self.labels_list = labels_list
        self.input_lengths = np.full(self.batch_size, T, dtype=np.int32)
        self.label_lengths = np.array([len(l) for l in labels_list], dtype=np.int32)

        self.log_probs = log_softmax(y_pred)
        
        total_loss = 0
        valid_samples = 0 # (NEW) inf가 아닌 샘플 수
        
        for b in range(self.batch_size):
            log_prob = self.log_probs[b] 
            labels = self.labels_list[b]   
            T = self.input_lengths[b]
            L = self.label_lengths[b]

            L_prime = 2 * L + 1
            labels_with_blanks = np.full(L_prime, self.blank_label, dtype=np.int32)
            labels_with_blanks[1::2] = labels
            
            alpha = np.full((T, L_prime), -np.inf) 

            alpha[0, 0] = log_prob[0, self.blank_label] 
            if L > 0:
                alpha[0, 1] = log_prob[0, labels[0]] 
                
            for t in range(1, T):
                for s in range(L_prime):
                    current_label = labels_with_blanks[s]
                    p1 = alpha[t-1, s]
                    p2 = -np.inf
                    if s > 0: p2 = alpha[t-1, s-1]
                    p3 = -np.inf
                    if s > 1 and current_label != self.blank_label and current_label != labels_with_blanks[s-2]:
                        p3 = alpha[t-1, s-2]
                    alpha[t, s] = np.logaddexp(np.logaddexp(p1, p2), p3) + log_prob[t, current_label]

            if L == 0: 
                loss_b = -alpha[T-1, 0]
            else:
                l1 = alpha[T-1, L_prime - 1] 
                l2 = alpha[T-1, L_prime - 2] 
                loss_b = -np.logaddexp(l1, l2) 
                
            # [ ★★★★★ 수정된 부분 ★★★★★ ]
            # (inf/nan이 아닌 유효한 샘플만 loss에 더함)
            if not np.isnan(loss_b) and not np.isinf(loss_b):
                total_loss += loss_b
                valid_samples += 1
            
        # (NEW) 유효한 샘플 수로만 나눔
        if valid_samples == 0:
            return 0.0 # (매우 드문 경우: 모든 배치가 inf)
        return total_loss / valid_samples

    def backward(self):
        grad = np.zeros_like(self.log_probs)
        
        for b in range(self.batch_size):
            log_prob = self.log_probs[b] 
            labels = self.labels_list[b]   
            T = self.input_lengths[b]
            L = self.label_lengths[b]
            
            L_prime = 2 * L + 1
            labels_with_blanks = np.full(L_prime, self.blank_label, dtype=np.int32)
            labels_with_blanks[1::2] = labels

            # 1. (Forward) Alpha 테이블 재계산
            alpha = np.full((T, L_prime), -np.inf)
            alpha[0, 0] = log_prob[0, self.blank_label]
            if L > 0: alpha[0, 1] = log_prob[0, labels[0]]
            
            for t in range(1, T):
                for s in range(L_prime):
                    current_label = labels_with_blanks[s]
                    p1 = alpha[t-1, s]
                    p2 = -np.inf
                    if s > 0: p2 = alpha[t-1, s-1]
                    p3 = -np.inf
                    if s > 1 and current_label != self.blank_label and current_label != labels_with_blanks[s-2]:
                        p3 = alpha[t-1, s-2]
                    alpha[t, s] = np.logaddexp(np.logaddexp(p1, p2), p3) + log_prob[t, current_label]

            # [ ★★★★★ 수정된 부분 ★★★★★ ]
            # Z를 먼저 계산하고, Z가 -inf이면 (loss가 inf였음)
            # 해당 샘플의 그래디언트 계산을 건너뜀 (grad[b]는 0으로 남음)
            if L == 0:
                Z = alpha[T-1, 0]
            else:
                Z = np.logaddexp(alpha[T-1, L_prime - 1], alpha[T-1, L_prime - 2])
            
            if np.isneginf(Z):
                continue # (그래디언트 계산 건너뛰기)
                
            # 2. (Backward) Beta 테이블 초기화
            beta = np.full((T, L_prime), -np.inf)
            if L == 0: 
                beta[T-1, 0] = 0.0
            else:
                beta[T-1, L_prime - 1] = 0.0 
                beta[T-1, L_prime - 2] = 0.0 
                
            # 3. Beta 테이블 채우기
            for t in range(T - 2, -1, -1): 
                for s in range(L_prime):
                    current_label = labels_with_blanks[s]
                    p1 = beta[t+1, s]
                    p2 = -np.inf
                    if s < L_prime - 1: p2 = beta[t+1, s+1]
                    p3 = -np.inf
                    if s < L_prime - 2 and labels_with_blanks[s+2] != self.blank_label and labels_with_blanks[s+2] != current_label:
                        p3 = beta[t+1, s+2]
                    beta[t, s] = np.logaddexp(np.logaddexp(p1, p2), p3) + log_prob[t, current_label]

            # 4. 그래디언트 계산
            log_alpha_beta = alpha + beta
            grad_b = np.full_like(log_prob, -np.inf) 
            
            for t in range(T):
                for s in range(L_prime):
                    current_label = labels_with_blanks[s]
                    grad_b[t, current_label] = np.logaddexp(grad_b[t, current_label], log_alpha_beta[t, s])
            
            # 5. 최종 그래디언트 계산 (들여쓰기 수정됨)
            P_path_given_y = np.exp(grad_b - Z)
            P_path_given_y[np.isnan(P_path_given_y)] = 0 
            
            grad[b] = np.exp(self.log_probs[b]) - P_path_given_y

        # 6. 배치 전체에 대해 평균
        return grad / self.batch_size