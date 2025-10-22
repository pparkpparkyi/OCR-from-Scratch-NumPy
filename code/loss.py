# loss.py (CTCLoss 구현)
import numpy as np

# ----------------------------------------------------
# 1. (삭제) CrossEntropyLoss 클래스
# ----------------------------------------------------
# class CrossEntropyLoss:
#     ... (이전 코드 삭제) ...
# ----------------------------------------------------


def log_softmax(x):
    """
    수치적으로 안정적인 LogSoftmax 계산 (Overflow 방지)
    x shape: (N, T, C)
    """
    # C 축에 대해 최대값 빼기
    max_x = np.max(x, axis=2, keepdims=True)
    e_x = np.exp(x - max_x)
    sum_e_x = np.sum(e_x, axis=2, keepdims=True)
    
    # LogSoftmax 공식: log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))
    return (x - max_x) - np.log(sum_e_x)


class CTCLoss:
    """Connectionist Temporal Classification (CTC) Loss"""
    def __init__(self, blank_label=0):
        """
        blank_label: dataset.py에서 0번으로 정의한 '_' 토큰
        """
        self.blank_label = blank_label
        self.log_probs = None # log_softmax(y_pred) 저장용
        self.labels_list = None # 정답 시퀀스 리스트 저장용
        self.input_lengths = None # T (시간 길이)
        self.label_lengths = None # L (레이블 길이)
        self.batch_size = None
        
    def forward(self, y_pred, labels_list):
        """
        y_pred (Logits): (N, T, C) - 모델의 순수 출력
        labels_list (List): [[10, 5], [3], ...] - 길이가 다른 정답 리스트
        """
        self.batch_size, T, C = y_pred.shape
        self.labels_list = labels_list
        self.input_lengths = np.full(self.batch_size, T, dtype=np.int32)
        self.label_lengths = np.array([len(l) for l in labels_list], dtype=np.int32)

        # 1. LogSoftmax 계산 (CTCLoss의 첫 단계)
        self.log_probs = log_softmax(y_pred)
        
        total_loss = 0
        
        # --- 배치(N)의 각 샘플에 대해 개별적으로 DP 계산 ---
        for b in range(self.batch_size):
            log_prob = self.log_probs[b] # (T, C)
            labels = self.labels_list[b]   # (L,)
            T = self.input_lengths[b]
            L = self.label_lengths[b]

            # 2. 정답 레이블을 CTCLoss용으로 변환
            # e.g., "대구" [10, 5] -> [_, 10, _, 5, _] (Blank 삽입)
            # L' = 2*L + 1
            L_prime = 2 * L + 1
            labels_with_blanks = np.full(L_prime, self.blank_label, dtype=np.int32)
            labels_with_blanks[1::2] = labels
            
            # 3. DP 테이블 (Alpha) 초기화
            # alpha[t, s] = t 시점, s 위치에 있을 로그 확률
            alpha = np.full((T, L_prime), -np.inf) # 로그 확률이므로 -inf로 초기화

            # 4. DP 시작점
            alpha[0, 0] = log_prob[0, self.blank_label] # (t=0, s=0) -> Blank
            if L > 0:
                alpha[0, 1] = log_prob[0, labels[0]] # (t=0, s=1) -> '대'
                
            # 5. DP 테이블 채우기 (Forward Pass)
            for t in range(1, T):
                for s in range(L_prime):
                    current_label = labels_with_blanks[s]
                    
                    # 5-1. (경로 1) 이전 시점(t-1)의 같은 위치(s)에서 온 경우
                    p1 = alpha[t-1, s]
                    
                    # 5-2. (경로 2) 이전 시점(t-1)의 이전 위치(s-1)에서 온 경우
                    p2 = -np.inf
                    if s > 0:
                        p2 = alpha[t-1, s-1]
                        
                    # 5-3. (경로 3) 이전 시점(t-1)의 (s-2) 위치에서 온 경우
                    # (s가 blank가 아니고, s와 s-2가 다를 때만 허용. e.g., L_L)
                    p3 = -np.inf
                    if s > 1 and current_label != self.blank_label and current_label != labels_with_blanks[s-2]:
                        p3 = alpha[t-1, s-2]
                        
                    # 3개 경로의 로그 확률을 더함 (np.logaddexp)
                    alpha[t, s] = np.logaddexp(np.logaddexp(p1, p2), p3) + log_prob[t, current_label]

            # 6. 최종 손실 계산 (마지막 두 지점)
            if L == 0: # 정답이 비어있을 때
                loss_b = -alpha[T-1, 0]
            else:
                l1 = alpha[T-1, L_prime - 1] # 마지막이 '구'
                l2 = alpha[T-1, L_prime - 2] # 마지막이 '_'
                loss_b = -np.logaddexp(l1, l2) # 두 경로의 확률을 더함
                
            total_loss += loss_b
            
        return total_loss / self.batch_size

    def backward(self):
        """
        CTCLoss의 역전파 (그래디언트 계산)
        (Forward Pass와 매우 유사한 DP를 거꾸로 수행)
        """
        
        # (N, T, C) 형태의 그래디언트 배열 초기화
        grad = np.zeros_like(self.log_probs)
        
        # --- 배치(N)의 각 샘플에 대해 개별적으로 DP 계산 ---
        for b in range(self.batch_size):
            log_prob = self.log_probs[b] # (T, C)
            labels = self.labels_list[b]   # (L,)
            T = self.input_lengths[b]
            L = self.label_lengths[b]
            
            L_prime = 2 * L + 1
            labels_with_blanks = np.full(L_prime, self.blank_label, dtype=np.int32)
            labels_with_blanks[1::2] = labels

            # 1. (Forward) Alpha 테이블 재계산 (저장 안 했으므로)
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

            # 2. (Backward) Beta 테이블 초기화
            beta = np.full((T, L_prime), -np.inf)
            if L == 0: # 정답이 비어있을 때
                beta[T-1, 0] = 0.0
            else:
                beta[T-1, L_prime - 1] = 0.0 # 마지막 '구'
                beta[T-1, L_prime - 2] = 0.0 # 마지막 '_'
                
            # 3. Beta 테이블 채우기 (Backward Pass)
            for t in range(T - 2, -1, -1): # 거꾸로
                for s in range(L_prime):
                    current_label = labels_with_blanks[s]
                    
                    # 3-1. (경로 1) 다음 시점(t+1)의 같은 위치(s)
                    p1 = beta[t+1, s]
                    
                    # 3-2. (경로 2) 다음 시점(t+1)의 다음 위치(s+1)
                    p2 = -np.inf
                    if s < L_prime - 1:
                        p2 = beta[t+1, s+1]
                        
                    # 3-3. (경로 3) 다음 시점(t+1)의 (s+2) 위치
                    p3 = -np.inf
                    if s < L_prime - 2 and labels_with_blanks[s+2] != self.blank_label and labels_with_blanks[s+2] != current_label:
                        p3 = beta[t+1, s+2]
                        
                    beta[t, s] = np.logaddexp(np.logaddexp(p1, p2), p3) + log_prob[t, current_label]

            # 4. 그래디언트 계산
            # (alpha + beta)가 t시점, s위치를 통과하는 모든 경로의 확률
            log_alpha_beta = alpha + beta
            
            # (N, T, C) 그래디언트 맵 초기화
            grad_b = np.full_like(log_prob, -np.inf) # (T, C)
            
            for t in range(T):
                for s in range(L_prime):
                    current_label = labels_with_blanks[s]
                    # (t,s)를 지나는 확률을 (t, current_label) 칸에 누적
                    grad_b[t, current_label] = np.logaddexp(grad_b[t, current_label], log_alpha_beta[t, s])
            
            # 5. 최종 그래디언트 계산 (P - Z)
            # Z = 손실 (Log(P_total))
            if L == 0:
                Z = alpha[T-1, 0]
            else:
                Z = np.logaddexp(alpha[T-1, L_prime - 1], alpha[T-1, L_prime - 2])

            # dL/dy = P(softmax) - P(label_path | y)
            # (로그 스케일이므로 exp() 적용)
            grad[b] = np.exp(self.log_probs[b]) - np.exp(grad_b - Z)

        # 6. 배치 전체에 대해 평균
        return grad / self.batch_size