import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def verify_preprocessing(data_dir, num_samples=5):
    """
    preprocess.py의 실행 결과를 검증하는 스크립트.
    1. 파일 개수 확인
    2. 이미지 크기 및 모드 확인
    3. 이미지와 레이블을 시각화하여 내용 일치 확인
    """
    print("=" * 50)
    print(f"Verifying preprocessing results in: {data_dir}")
    print("=" * 50)

    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')

    # --- 1. 경로 및 파일 개수 확인 ---
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"❌ ERROR: '{img_dir}' 또는 '{lbl_dir}' 폴더를 찾을 수 없습니다.")
        print("preprocess.py를 먼저 실행했는지 확인하세요.")
        return

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    lbl_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]

    print(f"✅ Found {len(img_files)} image files in '{img_dir}'")
    print(f"✅ Found {len(lbl_files)} label files in '{lbl_dir}'")

    if len(img_files) == 0:
        print("❌ ERROR: 이미지 파일이 하나도 없습니다. 전처리가 실패했을 수 있습니다.")
        return
    
    if abs(len(img_files) - len(lbl_files)) > 10: # vocab.json 감안
        print(f"⚠️ WARNING: 이미지({len(img_files)})와 레이블({len(lbl_files)}) 파일 개수 차이가 큽니다.")

    # --- 2. 무작위 샘플 선택 및 시각화 ---
    print(f"\nDisplaying {num_samples} random samples for visual check...")
    
    # 시각화를 위해 랜덤 샘플 선택
    random_samples = random.sample(img_files, min(num_samples, len(img_files)))

    # 한글 폰트 설정 (없으면 기본 폰트로 표시되어 깨질 수 있음)
    try:
        # 윈도우
        plt.rc('font', family='Malgun Gothic')
    except:
        try:
            # macOS
            plt.rc('font', family='AppleGothic')
        except:
            # 기타 (나눔고딕 등 설치 필요)
            print("한글 폰트(Malgun Gothic/AppleGothic)를 찾을 수 없어 글자가 깨질 수 있습니다.")
            pass
    
    # matplotlib figure 설정
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    fig.suptitle('Preprocessing Verification Samples', fontsize=16)

    for i, img_name in enumerate(random_samples):
        base_name = os.path.splitext(img_name)[0]
        lbl_name = f"{base_name}.txt"
        
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)

        # 이미지 열기 및 속성 확인
        try:
            with Image.open(img_path) as img:
                img_array = list(img.getdata()) # 이미지를 실제로 읽어보기
                
                # 시각화
                ax = axes[i]
                ax.imshow(img, cmap='gray')
                
                # 이미지 속성 검사
                if img.size != (32, 32) or img.mode != 'L':
                    ax.set_title(f"⚠️ WRONG FORMAT!\nSize:{img.size}, Mode:{img.mode}", color='red')
                
                # 레이블 읽기 및 표시
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    ax.set_xlabel(f"Label: '{text}'", fontsize=12)
                else:
                    ax.set_xlabel("Label NOT FOUND", color='red')
                
                ax.set_xticks([])
                ax.set_yticks([])

        except Exception as e:
            print(f"❌ ERROR: 샘플 파일을 여는 중 오류 발생: {img_name}, {e}")
            axes[i].set_title("ERROR", color='red')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # 검증하고 싶은 폴더 경로를 지정하세요.
    TRAIN_DATA_DIR = "./data/train"
    
    verify_preprocessing(TRAIN_DATA_DIR, num_samples=5)
