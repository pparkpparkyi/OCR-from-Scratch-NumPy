import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 설정 ---
# 검증할 데이터 폴더 경로 (train 또는 valid)
DATA_DIR_TO_VERIFY = "./data/train"
# 시각적으로 확인할 샘플 개수
NUM_SAMPLES_TO_SHOW = 5
# --- 설정 끝 ---

def find_korean_font():
    """시스템에 설치된 한글 폰트를 찾아서 반환합니다."""
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    
    # 우선적으로 'Malgun Gothic'을 찾습니다 (Windows 기본)
    for font_file in font_files:
        if 'malgun' in font_file.lower():
            return font_file
            
    # 'AppleGothic'을 찾습니다 (macOS 기본)
    for font_file in font_files:
        if 'apple' in font_file.lower() and 'gothic' in font_file.lower():
            return font_file

    # 'Nanum' 폰트를 찾습니다 (Linux 등에서 흔히 사용)
    for font_file in font_files:
        if 'nanum' in font_file.lower() and 'gothic' in font_file.lower():
            return font_file

    # 위 폰트들이 없으면, 찾은 폰트 중 첫 번째 것을 사용 (깨질 수 있음)
    return font_files[0] if font_files else None

def verify_preprocessing(data_dir, num_samples):
    """
    전처리된 데이터 폴더를 검증하고, 무작위 샘플을 시각화합니다.
    """
    print("=" * 50)
    print(f"Verifying preprocessing results in: {data_dir}")
    print("=" * 50)

    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')

    # 1. 폴더 존재 여부 확인
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"❌ Error: '{img_dir}' or '{lbl_dir}' not found.")
        print("Please check if preprocess.py ran successfully.")
        return

    # 2. 파일 개수 확인
    try:
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        lbl_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    except FileNotFoundError:
        print(f"❌ Error: Could not list files in directories.")
        return

    print(f"✅ Found {len(img_files)} image files in '{img_dir}'")
    print(f"✅ Found {len(lbl_files)} label files in '{lbl_dir}'")

    if len(img_files) == 0:
        print("❌ Error: No image files found. Preprocessing may have failed.")
        return
        
    if abs(len(img_files) - len(lbl_files)) > 10: # 약간의 오차 허용
        print(f"⚠️ Warning: Image count ({len(img_files)}) and label count ({len(lbl_files)}) differ significantly.")

    # 3. 무작위 샘플 시각화
    print(f"\nDisplaying {num_samples} random samples for visual check...")
    
    # 한글 폰트 설정
    font_path = find_korean_font()
    if font_path:
        font_prop = fm.FontProperties(fname=font_path, size=14)
        print(f"Using Korean font: {os.path.basename(font_path)}")
    else:
        font_prop = None
        print("Warning: Korean font not found. Labels may appear broken.")
        
    # 무작위 샘플 선택
    random_samples = random.sample(img_files, min(num_samples, len(img_files)))

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 4))
    if num_samples == 1:
        axes = [axes] # 샘플이 1개일 때도 리스트로 만듦

    for i, img_name in enumerate(random_samples):
        base_name = os.path.splitext(img_name)[0]
        lbl_name = f"{base_name}.txt"
        
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)

        # 이미지 로드 및 확인
        try:
            img = Image.open(img_path)
            img_array =  _ = plt.imread(img_path)

            axes[i].imshow(img_array, cmap='gray')
            axes[i].axis('off')

            title_text = ""
            
            # 레이블 로드
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                title_text += f'Label: "{text}"'
            else:
                title_text += "[Label Not Found!]"

            # 이미지 속성 검증
            warnings = []
            if img.size != (32, 32):
                warnings.append(f"Size:{img.size}!= (32,32)")
            if img.mode != 'L':
                warnings.append(f"Mode:{img.mode}!='L'")
            
            if warnings:
                axes[i].set_title(title_text + f"\n⚠️ {' '.join(warnings)}", color='red', fontproperties=font_prop)
            else:
                axes[i].set_title(title_text, fontproperties=font_prop)

        except Exception as e:
            axes[i].set_title(f"Error loading sample:\n{img_name}")
            print(f"Error processing {img_name}: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_preprocessing(DATA_DIR_TO_VERIFY, NUM_SAMPLES_TO_SHOW)

