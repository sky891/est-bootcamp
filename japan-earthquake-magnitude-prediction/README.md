# 일본 지역 지진 규모 예측 모델

## 📌 프로젝트 개요
일본 지역에서 발생하는 지진의 규모를 예측하는 AI 모델을 구축하는 프로젝트입니다. Kaggle의 일본 지진 데이터셋을 활용하여 시계열 분석을 수행하고, 머신러닝 및 딥러닝 모델을 비교하여 최적의 예측 모델을 선정하였습니다.

## 🚀 핵심 내용
- **목표**: 일본 지진 데이터를 분석하여 규모 예측 모델 개발
- **사용 기술**: Python, TensorFlow, Scikit-Learn, XGBoost, Matplotlib
- **데이터 출처**: [Kaggle - Earthquakes in Japan](https://www.kaggle.com/datasets/aerodinamicc/earthquakes-in-japan)
- **최적 모델**: Transformer (MAE 10% 감소)
- **성과**: 모델 성능 비교 및 지진 예측 가능성 검토

## 📊 데이터 개요
- **데이터 기간**: 2001년 ~ 2018년
- **데이터 크기**: 21,120 rows × 13 columns
- **주요 변수**:
  - `Time`: 지진 발생 시간
  - `Latitude`, `Longitude`: 진앙의 위도 및 경도
  - `Depth`: 지진 깊이
  - `Mag`: 지진 규모 (예측 대상)
  - `MagType`: 지진 규모 측정 방식

## 🏗️ 프로젝트 구조
```
/project
│── main.py                  # 실행 스크립트
│── preprocessing.py         # 데이터 전처리 및 로딩
│── train.py                 # 모델 정의 및 학습 평가
│── requirements.txt         # 필요한 라이브러리 목록
│── models/                  # 학습된 모델 저장 디렉토리
    │── rnn_model.py
    │── t_lstm_time_decay.py
    │── t_lstm_time_interval.py
    │── transformer_model.py
```

## 🛠️ 실행 방법
```bash
pip install -r requirements.txt  # 1. 라이브러리 설치
pytohn data_preprocessing.py     # 2. 데이터 전처리
python train.py                  # 3. 전처리된 데이터로 모델 학습 및 평가 실행
```

## 🔬 모델링 과정
### 1️⃣ 데이터 전처리 (`preprocessing.py`)
- 결측치 처리 및 이상치 제거
- 1시간 단위 리샘플링 및 선형 보간법 적용
- 지진이 발생한 후 알 수 있는 컬럼 제외
- 범주형 변수 원-핫 인코딩, 수치형 변수 `StandardScaler` 적용

### 2️⃣ 모델 학습 (`model_training.py`)
#### 사용된 모델:
✅ **랜덤 포레스트, XGBoost, LGBM, CatBoost** (머신러닝 모델)
✅ **RNN, LSTM, Transformer, T-LSTM** (딥러닝 모델)

- LSTM & RNN: 시계열성을 고려한 학습
- Transformer: 장기적 패턴 학습 최적화
- T-LSTM: 시간 간격 변화를 반영하는 특수 모델
- 베이지안 최적화 (Bayesian Optimization) 활용하여 하이퍼파라미터 튜닝

### 3️⃣ 모델 평가 (`evaluation.py`)
- **평가지표**: RMSE, MAE, 결정계수(R²)
- 최적 모델 선정: **Transformer 모델이 가장 높은 예측 성능 기록**
- 예측값 vs 실제값 비교 시각화 진행

## 🔍 결과 및 성과
✔️ Transformer 모델이 기존 LSTM 대비 **MAE 10% 감소**
✔️ 지진 규모 예측의 가능성을 확인하였으나, 개선 여지가 존재
✔️ 추가적인 지질학적 요소 활용 필요

## 📌 향후 개선 방향
📍 실시간 데이터 반영 가능성 검토
📍 Transformer 기반 하이브리드 모델 탐색
📍 Feature Engineering을 통한 성능 향상
📍 불규칙한 이벤트 기반 데이터 처리를 위한 모델 개선 연구

## 🏆 기여자
👨‍💻 **김진혁** (팀장)
👨‍💻 **김영훈**
👨‍💻 **서보석**
👨‍💻 **이영국**

## 📜 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

---
✉️ **문의사항이 있다면 연락주세요!**
