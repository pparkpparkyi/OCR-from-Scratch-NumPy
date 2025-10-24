# dataset.py (한글 필터링 적용)
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import json
import re # (NEW) 정규표현식 라이브러리

def is_valid_hangul_sample(text):
    """
    텍스트가 '한글(가-힣)'로만 구성되어 있는지 확인합니다.
    (공백, 숫자, 한자, 영어, 특수문자 모두 제외)
    """
    if not text:
        return False
    
    # [수정] 오직 '가'부터 '힣' 사이의 한글만 허용
    return re.fullmatch(r'[가-힣]+', text) is not None

class OCRDataset:
    """전처리된 데이터셋 로더 (한글 필터링 + Vocab 캐싱)"""
    def __init__(self, data_dir, max_samples=10000):
        self.img_dir = os.path.join(data_dir, 'images')
        self.lbl_dir = os.path.join(data_dir, 'labels')
        self.max_samples = max_samples
        
        self.samples = []
        self.char_to_idx = {} # 문자 -> 인덱스 맵
        self.idx_to_char = {} # 인덱스 -> 문자 맵
        
        # [순서 변경]
        # 1. 한글 샘플만 필터링하여 로드합니다.
        self._load_data()
        
        # 2. 필터링된 샘플을 기반으로 한글 전용 Vocab을 구축합니다.
        self._build_vocab()
        
    def _build_vocab(self):
        """(수정) 필터링된 self.samples를 기반으로 Vocab을 구축합니다."""
        
        # (NEW) 캐시 파일 경로를 data/train/hangul_vocab.json 등으로 변경
        base_dir = os.path.dirname(self.lbl_dir.rstrip(os.sep)) # ./data/train
        vocab_path = os.path.join(base_dir, "hangul_vocab.json")
        
        # 3. (Speed UP!) 이미 저장된 파일이 있으면, 바로 로드
        if os.path.exists(vocab_path):
            print(f"Loading cached Hangul vocabulary from {vocab_path}...")
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.char_to_idx = data['char_to_idx']
                    self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
                
                print(f"Vocabulary loaded. Total {len(self.char_to_idx)} characters.")
                return 
            except Exception as e:
                print(f"Warning: Failed to load cache {vocab_path}. Rebuilding... Error: {e}")

        # 4. (최초 1회) 필터링된 self.samples에서 Vocab 생성
        print(f"Building Hangul vocabulary from {len(self.samples)} filtered samples...")
        char_set = set()
        
        for sample in tqdm(self.samples, desc="Building Hangul Vocab"):
            try:
                with open(sample['lbl_path'], 'r', encoding='utf-8') as f:
                    text = f.read().strip() # (NEW) strip()으로 공백/줄바꿈 제거
                    for char in text:
                        # (is_valid_hangul_sample에서 이미 걸러졌지만, 이중 확인)
                        if '가' <= char <= '힣':
                            char_set.add(char)
            except Exception as e:
                print(f"Warning: Could not read {sample['lbl_path']}. Error: {e}")
                
        sorted_chars = sorted(list(char_set))
        
        self.char_to_idx = {'_': 0} # BLANK
        self.idx_to_char = {0: '_'}
        
        for i, char in enumerate(sorted_chars):
            idx = i + 1
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            
        # 5. (최초 1회) 생성된 Vocab을 파일로 저장
        print(f"Saving Hangul vocabulary to {vocab_path}...")
        try:
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump({'char_to_idx': self.char_to_idx, 'idx_to_char': self.idx_to_char}, 
                          f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error: Could not save cache to {vocab_path}. Error: {e}")
            
        print(f"Hangul vocabulary built and saved. Total {len(self.char_to_idx)} characters.")

    def _load_data(self):
        """(수정) 레이블 파일을 읽어 한글 전용 샘플만 로드합니다."""
        print(f"Loading and filtering Hangul-only samples from {self.lbl_dir}...")
        
        all_label_files = [f for f in os.listdir(self.lbl_dir) if f.endswith('.txt')]
        
        for lbl_name in tqdm(all_label_files, desc="Filtering Hangul Samples"):
            lbl_path = os.path.join(self.lbl_dir, lbl_name)
            
            try:
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip() # (NEW) strip()으로 공백/줄바꿈 제거
            except Exception as e:
                continue # 파일 읽기 실패 시 무시

            # [NEW] 한글 필터링
            if is_valid_hangul_sample(text):
                base_name = os.path.splitext(lbl_name)[0]
                img_name = f"{base_name}.png"
                img_path = os.path.join(self.img_dir, img_name)
                
                if os.path.exists(img_path):
                    self.samples.append({
                        'img_path': img_path,
                        'lbl_path': lbl_path
                    })

        print(f"Loaded {len(self.samples)} Hangul-only samples.")

        # [수정] 필터링이 끝난 후에 max_samples 적용
        if len(self.samples) > self.max_samples:
            # (데이터 순서가 섞이지 않았으므로 앞의 N개만 사용)
            self.samples = self.samples[:self.max_samples]
            print(f"Using first {len(self.samples)} samples (max_samples).")
        
    def __len__(self):
        return len(self.samples)
    
    def get_vocab_size(self):
        """모델에 클래스 개수를 알려주기 위한 헬퍼 함수"""
        return len(self.char_to_idx)
    
    def __getitem__(self, idx):
        """샘플 반환 (이미지 + 레이블 시퀀스)"""
        sample = self.samples[idx]
        
        # 1. 이미지 로드
        try:
            img = Image.open(sample['img_path'])
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.reshape(1, 32, 32)
        except Exception as e:
            print(f"Error loading image {sample['img_path']}: {e}")
            img = np.zeros((1, 32, 32), dtype=np.float32)
        
        # 2. 레이블 로드 및 시퀀스 변환
        try:
            with open(sample['lbl_path'], 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # (NEW) 사전에 있는 한글만 변환
            label = [self.char_to_idx[char] for char in text 
                     if char in self.char_to_idx] # '_'는 사전에 없으므로 자동 제외
            
        except Exception as e:
            print(f"Error loading label {sample['lbl_path']}: {e}")
            label = []
        
        return img, label
    
    def get_batch(self, batch_size):
        """배치 생성"""
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        images = []
        labels_list = []
        
        for idx in indices:
            img, label = self[idx]
            images.append(img)
            labels_list.append(label)
            
        images_batch = np.array(images)
        return images_batch, labels_list