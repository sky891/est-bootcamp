# 🚗 스마트카 표정 분석 및 감정 케어 시스템
</br>

## 📖 프로젝트 배경
- 운전자의 감정 분석을 통한 스마트 차량 환경 조성
- 운전 중 감정 상태(행복, 슬픔, 화남, 놀람 등)를 실시간으로 분석하고, 차량 내부 환경(온도, 조명, 음악)을 최적화하여 **운전자와 탑승자의 안전성과 편안함을 향상**하는 AI 기반 시스템입니다.  
</br>

### 기능 개요  
- **얼굴 감정 분석** : MobileNet 기반 감정 인식 모델 적용  
- **실시간 안전 기능** : 극단적 감정(패닉, 분노) 시 속도 제한, 긴급 연락(119, 가족) 기능  
- **개인화 환경 조성** : 감정 분석을 기반으로 음악, 조명, 온도 자동 조절  
- **AI 챗봇 연동** : ChatGPT API를 활용한 감정 케어 및 대화 제공  
- **데이터 증강 기법 활용** : StyleGAN, 3D 포즈 증강으로 감정 데이터 다양화  
</br>

## 🎯 프로젝트 목표
- 운전자의 감정 분석 정확도 향상
- 감정 기반으로 차량 내부 환경 자동 조정
- 운전 중 사고 위험 감소
- 실시간 대화형 AI를 활용한 감정 케어 지원
</br>

## 📊 모델 및 기술 스택

### AI 모델 
- MobileNetV3 기반 감정 인식 모델  
- Vision Transformer(ViT) & EfficientNetB1 실험  
- GAN 및 StyleGAN을 활용한 데이터 증강  
- YOLO를 이용한 얼굴 크롭  

### 데이터셋
- Kaggle FER2013 데이터 활용  
- 자체 구축 감정 데이터 (화남, 행복, 놀람, 슬픔)  

### 기술 스택
- **프레임워크** : TensorFlow, PyTorch  
- **OpenAI API** : ChatGPT API 기반 챗봇 개발  
- **전처리 및 증강** : OpenCV, 3DDFA, RetinaFace  
</br>

## ⚙️ 주요 기능 설명

### 1️⃣ 감정 인식 및 실시간 분석  
- OpenCV + MobileNetV3 모델을 활용한 얼굴 감정 인식  
- 운전자의 감정을 실시간으로 감지하고, 지속적으로 변하는 감정 데이터 분석  

### 2️⃣ 차량 내부 환경 자동 조정  
- **행복 😃** → 밝은 조명, 쾌적한 온도, 활기찬 음악  
- **화남 😡** → 차분한 음악, 시원한 온도, 속도 제한  
- **슬픔 😢** → 따뜻한 온도, 잔잔한 음악, 아늑한 조명  
- **놀람 😲** → 주의 환기 알림, 심박수 확인  

### 3️⃣ 긴급 대응 시스템  
- 감정 분석을 통해 **스트레스, 패닉 감정** 감지 시 자동 긴급 연락(119, 보호자)  
- 사고 가능성이 높은 감정 상태(분노, 불안)에서 속도 조절 및 경로 최적화  

### 4️⃣ AI 챗봇 기반 감정 케어  
- OpenAI ChatGPT API를 활용한 **운전자 감정 맞춤형 대화 제공**  
- Prompt Engineering 기법을 적용하여 심리적 안정 지원  
- 운전 중 음성 인터페이스를 통한 맞춤형 피드백 제공  
</br>

## 🛠️ 라이브러리 설치
```bash
pip install python-dotenvt  # 1. 라이브러리 설치 (해당 패키지가 설치되어 있지 않다면)
```
</br>
</br>

## 📌 프로젝트 결과
- 감정 분류 정확도 **최고 80%**
- MobileNetV3 기반 모델이 **ResNet50 대비 성능 3배 향상**
- GAN 및 3D 포즈 증강을 통한 데이터 증강으로 성능 개선  



