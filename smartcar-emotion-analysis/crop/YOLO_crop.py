from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image, ImageOps
import os

# 모델 다운로드 및 로드
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# 입력 및 출력 경로 설정
input_dir = "/workspace/data/test/img/sadness"  # 입력 이미지 폴더
output_dir = "/workspace/daewoong/data_cropped/test/sadness"  # 얼굴을 저장할 폴더
os.makedirs(output_dir, exist_ok=True)  # 출력 폴더 생성

# 이미지 폴더 내 모든 파일 처리
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):  # 이미지 파일만 처리
        image_path = os.path.join(input_dir, file_name)
        
        # 이미지 로드 및 EXIF 회전 정보 적용
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)  # EXIF 회전 정보 적용
        
        # 모델 추론
        results = model(image)
        detections = Detections.from_ultralytics(results[0])

        # 얼굴 감지 영역 크롭 및 저장
        for idx, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, box)  # 얼굴 박스 좌표 (xmin, ymin, xmax, ymax)
            cropped_face = image.crop((x1, y1, x2, y2))  # 얼굴 영역 크롭

            # 얼굴 이미지 저장 (파일명에 원본 이미지명 추가)
            face_filename = os.path.join(output_dir, f"{file_name.split('.')[0]}_face_{idx}.jpg")
            cropped_face.save(face_filename)
            print(f"Saved: {face_filename}")

print("모든 이미지의 얼굴 크롭 완료!")


