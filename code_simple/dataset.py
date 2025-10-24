# dataset.py
import numpy as np
import json
import os
from PIL import Image

class OCRDataset:
    """AI Hub OCR 데이터셋 로더"""
    def __init__(self, data_dir, max_samples=1000):
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.samples = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._load_data()
        
    def _load_data(self):
        """데이터 로드 및 전처리"""
        print(f"Loading data from {self.data_dir}...")
        
        # _4PR_ 파일만 필터링
        files = [f for f in os.listdir(self.data_dir) if '_4PR_' in f and f.endswith('.json')]
        files = files[:self.max_samples]
        
        char_set = set()
        
        for json_file in files:
            json_path = os.path.join(self.data_dir, json_file)
            img_file = json_file.replace('.json', '.png')
            img_path = os.path.join(self.data_dir, img_file)
            
            if not os.path.exists(img_path):
                continue
                
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)

                # [수정 시작]
                # 1. 'bbox' 키가 있는지, 비어있지 않은지 확인
                if data.get('bbox') and len(data['bbox']) > 0:
                                    
                    # 2. 우리는 '첫 글자'만 필요하므로, 첫 번째 bbox의 텍스트를 가져옴
                    #    (참고: 첫 번째 bbox는 '017040' 같은 ID일 수 있으나,
                    #     지금은 파싱 테스트가 목적이므로 그대로 진행합니다.)
                    text = data['bbox'][0].get('data', '')

                    # 3. 텍스트가 비어있지 않다면 샘플 추가
                    if len(text) > 0:
                        char_set.add(text[0]) # 첫 글자만 char_set에 추가

                        self.samples.append({
                            'image_path': img_path,
                            'text': text,
                            'label': ord(text[0]) % 100  # <- 단순화된 레이블 유지
                        })
                # [수정 끝]
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        # 문자-인덱스 매핑
        chars = sorted(list(char_set))
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        print(f"Loaded {len(self.samples)} samples with {len(self.char_to_idx)} unique characters")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """샘플 반환"""
        sample = self.samples[idx]
        
        # 이미지 로드 및 전처리
        img = Image.open(sample['image_path']).convert('L')
        img = img.resize((32, 32))
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.reshape(1, 32, 32)
        
        label = sample['label']
        
        return img, label
    
    def get_batch(self, batch_size):
        """배치 생성"""
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        images = []
        labels = []
        
        for idx in indices:
            img, label = self[idx]
            images.append(img)
            labels.append(label)
            
        return np.array(images), np.array(labels)
