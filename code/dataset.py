# dataset.py (2단계: 시퀀스 + Vocab 캐싱)
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import json  # <-- 1. JSON 라이브러리 임포트

class OCRDataset:
    """전처리된 데이터셋 로더 (시퀀스 레이블 + Vocab 캐싱)"""
    def __init__(self, data_dir, max_samples=10000):
        self.img_dir = os.path.join(data_dir, 'images')
        self.lbl_dir = os.path.join(data_dir, 'labels')
        self.max_samples = max_samples
        
        self.samples = []
        self.char_to_idx = {} # 문자 -> 인덱스 맵
        self.idx_to_char = {} # 인덱스 -> 문자 맵
        
        # 1. Vocabulary(문자 사전) 구축 (캐싱 적용)
        self._build_vocab()
        
        # 2. 파일 목록 로드
        self._load_data()
        
    def _build_vocab(self):
        """(수정) Vocab을 만들거나, 이미 있으면 로드합니다."""
        
        # 2. 캐시 파일 경로 정의
        vocab_path = os.path.join(self.lbl_dir, "vocab.json")
        
        # 3. (Speed UP!) 이미 저장된 파일이 있으면, 바로 로드
        if os.path.exists(vocab_path):
            print(f"Loading cached vocabulary from {vocab_path}...")
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # JSON은 키를 문자열로 저장하므로, int 키를 다시 변환
                    self.char_to_idx = data['char_to_idx']
                    self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
                
                print(f"Vocabulary loaded. Total {len(self.char_to_idx)} characters.")
                return # (중요) 즉시 함수 종료
            except Exception as e:
                print(f"Warning: Failed to load cache {vocab_path}. Rebuilding... Error: {e}")

        # 4. (최초 1회) 파일이 없으면, 기존 로직대로 생성
        print(f"Building vocabulary from {self.lbl_dir}...")
        char_set = set()
        
        for lbl_name in tqdm(os.listdir(self.lbl_dir), desc="Building Vocab"):
            lbl_path = os.path.join(self.lbl_dir, lbl_name)
            if not lbl_name.endswith('.txt'): # vocab.json 같은 파일 무시
                continue
            try:
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    for char in text:
                        char_set.add(char)
            except Exception as e:
                print(f"Warning: Could not read {lbl_path}. Error: {e}")
                
        sorted_chars = sorted(list(char_set))
        
        self.char_to_idx = {'_': 0}
        self.idx_to_char = {0: '_'}
        
        for i, char in enumerate(sorted_chars):
            idx = i + 1
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            
        # 5. (최초 1회) 생성된 Vocab을 파일로 저장
        print(f"Saving vocabulary to {vocab_path}...")
        try:
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump({'char_to_idx': self.char_to_idx, 'idx_to_char': self.idx_to_char}, 
                          f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error: Could not save cache to {vocab_path}. Error: {e}")
            
        print(f"Vocabulary built and saved. Total {len(self.char_to_idx)} characters.")

    def _load_data(self):
        """이미지/레이블 파일 경로 쌍을 로드합니다."""
        print(f"Loading data samples from {self.img_dir}...")
        
        img_files = os.listdir(self.img_dir)
        
        if len(img_files) > self.max_samples:
            img_files = img_files[:self.max_samples]
        
        for img_name in img_files:
            if not img_name.endswith('.png'): # 혹시 모를 .DS_Store 같은 파일 무시
                continue
                
            base_name = os.path.splitext(img_name)[0]
            lbl_name = f"{base_name}.txt"
            lbl_path = os.path.join(self.lbl_dir, lbl_name)
            
            if os.path.exists(lbl_path):
                self.samples.append({
                    'img_path': os.path.join(self.img_dir, img_name),
                    'lbl_path': lbl_path
                })
        
        print(f"Loaded {len(self.samples)} samples.")
        
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
                text = f.read()
            
            label = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
            
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