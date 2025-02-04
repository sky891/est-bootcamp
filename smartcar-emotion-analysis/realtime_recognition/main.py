import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
from PIL import Image
import os
from models import model1, model2
from chatgpt_api import chatgpt


# Haar Cascade 얼굴 검출기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 모델 파일 경로
model1_path = os.path.expanduser("~/Documents/EstSoft/프로젝트 3/model/MobileNetV3Net_best_emotion_classifier_anger_happy.pth")
model2_path = os.path.expanduser("~/Documents/EstSoft/프로젝트 3/model/MobileNetV3Net_best_emotion_classifier_panic_sad.pth")

# 모델 초기화 및 가중치 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = model1(model1_path, device)
model2 = model2(model2_path, device)


# 클래스 이름
emotion_classes1 = {0: "anger", 1: "happy"}
emotion_classes2 = {0: "panic", 1: "sad"}

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 카메라 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: 카메라를 열 수 없습니다.")
    exit()

import cv2
import numpy as np
from PIL import Image
import torch

# 설정 변수
emotion_threshold = 10  # 동일 감정이 지속되는 프레임 수
previous_emotion = None
emotion_stable_counter = 0
chatgpt_response = ""  # ChatGPT 응답 초기화

# 실시간 감정 인식
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: 프레임을 읽을 수 없습니다.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # 모델 1 예측
        output1 = model1(face_tensor)
        prob1 = torch.softmax(output1, dim=1)
        max_prob1, pred1 = torch.max(prob1, dim=1)
        emotion1 = emotion_classes1[pred1.item()]

        # 모델 2 예측
        output2 = model2(face_tensor)
        prob2 = torch.softmax(output2, dim=1)
        max_prob2, pred2 = torch.max(prob2, dim=1)
        emotion2 = emotion_classes2[pred2.item()]

        # 최종 감정 결정
        if max_prob1.item() > max_prob2.item():
            final_emotion = emotion1
        else:
            final_emotion = emotion2

        # 감정이 변경되었는지 확인
        if final_emotion == previous_emotion:
            emotion_stable_counter += 1
        else:
            emotion_stable_counter = 0
            previous_emotion = final_emotion

        # 감정이 일정하게 지속되면 ChatGPT 호출
        # if emotion_stable_counter >= emotion_threshold:
        #     chatgpt_response = chatgpt(final_emotion)

        # 화면에 사각형 및 감정 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Emotion: {final_emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ChatGPT 응답을 여러 줄로 나눠 출력
        # response_lines = chatgpt_response.split('\n')
        # for i, line in enumerate(response_lines):
        #     cv2.putText(frame, line, (x, y - 40 - (i * 20)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Detection with Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
