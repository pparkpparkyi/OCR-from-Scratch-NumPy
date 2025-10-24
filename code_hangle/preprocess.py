import zipfile
import json
import os
from PIL import Image
import io
from tqdm import tqdm # (터미널에 'pip install tqdm' 실행 필요)

# --- 1. 경로 설정 ---
# (r"..."을 사용한 절대 경로를 강력히 추천합니다!)

# [학습용]
#TRAIN_IMG_TS6_ZIP = r"053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/원천데이터/TS6.zip"
#TRAIN_IMG_TS7_ZIP = #r"053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/원천데이터/TS7.zip" # (나중에 다운되면 경로 추가)
#TRAIN_IMG_TS8_ZIP = r"053.대용량 손글씨 OCR 데이터/01.데이터//1.Training/원천데이터/TS8.zip" # (나중에 다운되면 경로 추가)
#TRAIN_LBL_ZIP = r"053.대용량 손글씨 OCR 데이터/01.데이터//1.Training/라벨링데이터/TL.zip"

# [검증용]
VALID_IMG_ZIP = r"053.대용량 손글씨 OCR 데이터/01.데이터/2.Validation/원천데이터/VS.zip" # (나중에 다운되면 경로 추가)
VALID_LBL_ZIP = r"053.대용량 손글씨 OCR 데이터/01.데이터/2.Validation/라벨링데이터/VL.zip"

# [출력 폴더]
OUTPUT_TRAIN_DIR = './data/train'
OUTPUT_VALID_DIR = './data/valid'
# --- 경로 설정 끝 ---


def process_zip(img_zip_path, lbl_zip_path, output_base):
    """
    ZIP을 읽어 BBox 단위로 잘라낸 이미지와 텍스트 파일을 저장합니다.
    """
    
    # 출력 폴더 생성
    img_out_dir = os.path.join(output_base, 'images')
    lbl_out_dir = os.path.join(output_base, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    
    try:
        img_zip = zipfile.ZipFile(img_zip_path, 'r')
        lbl_zip = zipfile.ZipFile(lbl_zip_path, 'r')
    except FileNotFoundError as e:
        print(f"Error: ZIP 파일을 찾을 수 없습니다. {e}")
        print("스크립트 상단의 파일 경로가 올바른지 확인하세요.")
        return

    print(f"\n[{output_base}] 처리 시작... (이미지: {os.path.basename(img_zip_path)}, 레이블: {os.path.basename(lbl_zip_path)})")
    
    # 레이블 ZIP에서 _4PR_ JSON 파일 목록 가져오기
    lbl_files = [f for f in lbl_zip.namelist() if '_4PR_' in f and f.endswith('.json')]
    img_files = {os.path.basename(f): f for f in img_zip.namelist()} # 빠른 탐색용 딕셔너리
    
    total_bboxes = 0
    
    # JSON 파일 하나씩 처리 (tqdm으로 진행률 표시)
    for json_path in tqdm(lbl_files, desc=f"Processing {os.path.basename(img_zip_path)}"):
        json_basename = os.path.basename(json_path)
        img_basename = json_basename.replace('.json', '.png')
        
        # 짝이 되는 이미지가 *현재 이미지 zip 파일*에 있는지 확인
        if img_basename not in img_files:
            continue
            
        try:
            # JSON과 이미지 파일을 메모리로 읽기
            with lbl_zip.open(json_path) as f_json:
                data = json.load(f_json)
                
            img_path_in_zip = img_files[img_basename]
            with img_zip.open(img_path_in_zip) as f_img:
                original_img = Image.open(io.BytesIO(f_img.read()))

            # JSON 안의 모든 BBox 순회
            for i, bbox in enumerate(data.get('bbox', [])):
                text = bbox.get('data')
                x_coords = bbox.get('x')
                y_coords = bbox.get('y')
                
                if not text or not x_coords or not y_coords:
                    continue
                
                # BBox 좌표로 이미지 크롭
                box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                cropped_img = original_img.crop(box)
                
                # 1차 전처리 (흑백, 리사이즈)
                processed_img = cropped_img.convert('L').resize((32, 32))
                
                # 새 파일명으로 저장
                base_filename = os.path.splitext(json_basename)[0]
                new_filename_base = f"{base_filename}_bbox_{i:03d}"
                
                # 이미지 저장
                img_save_path = os.path.join(img_out_dir, f"{new_filename_base}.png")
                processed_img.save(img_save_path)
                
                # 텍스트 저장
                lbl_save_path = os.path.join(lbl_out_dir, f"{new_filename_base}.txt")
                with open(lbl_save_path, 'w', encoding='utf-8') as f_label:
                    f_label.write(text)
                
                total_bboxes += 1

        except Exception as e:
            print(f"\n파일 처리 중 오류: {json_path}, {e}")

    print(f"[{output_base}] 처리 완료. 총 {total_bboxes}개의 개별 이미지 저장.")
    img_zip.close()
    lbl_zip.close()


if __name__ == "__main__":
    print("데이터 전처리를 시작합니다...")
    
    # --- 1. 학습 데이터 (TS6) 처리 ---
    #if os.path.exists(TRAIN_IMG_TS8_ZIP) and os.path.exists(TRAIN_LBL_ZIP):
    #    process_zip(TRAIN_IMG_TS8_ZIP, TRAIN_LBL_ZIP, OUTPUT_TRAIN_DIR)
    #else:
    #    print("Warning: TS6.zip 또는 TL.zip 경로를 찾을 수 없습니다. (경로 확인 필요)")

    # --- 2. 나중에 추가할 파일들 안내 ---
    #print("\n--- 향후 작업 안내 ---")
    #print("TS7.zip, TS8.zip이 다운로드되면, 스크립트 상단 경로를 수정하고 다시 실행하세요.")
    #print("  -> data/train 폴더에 파일이 *추가*됩니다.")
    
    #print("\nVS.zip(검증용 이미지)이 다운로드되면, 아래 코드를 활성화하세요.")
    if os.path.exists(VALID_IMG_ZIP) and os.path.exists(VALID_LBL_ZIP):
        process_zip(VALID_IMG_ZIP, VALID_LBL_ZIP, OUTPUT_VALID_DIR)
    else:
        print("Warning: VS.zip 또는 VL.zip 경로를 찾을 수 없습니다.")

    print("\n모든 작업 완료!")