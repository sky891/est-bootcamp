# 📌 체성분을 이용한 비만도 예측 프로젝트

## 📖 프로젝트 배경
본 프로젝트는 **체성분 데이터를 활용하여 비만도를 예측하는 AI 모델**을 개발하는 것을 목표로 합니다. 기존 BMI 기반 비만도 측정법은 신뢰도가 낮다는 한계를 가지므로, **체성분 데이터를 활용한 보다 정밀한 비만도 예측 모델을 구축**하였습니다.
</br>
</br>
![image](https://github.com/user-attachments/assets/ffe7f407-38ea-4769-af04-3ef5280c928b)
- **데이터셋**: 공공데이터포털 제공 **세종특별자치시_인바디 측정내역** ([링크](https://www.data.go.kr/data/15128989/fileData.do))
- **데이터 크기**: 21,120 rows × 37 columns
- **목표 변수**: 비만도 (Obesity Level)
- **독립 변수**: 체성분 데이터 (근육량, 체지방량 등)
</br>


## 📊 데이터 분석 (EDA)
### 🔹 주요 전처리 과정
- **결측치 처리**: 결측치가 포함된 변수 제거
- **이상치 탐지**: 이상치 비율이 5% 이하이므로 제거 없이 유지
- **다중공선성 확인**: VIF 분석 후 다중공선성이 높은 변수 제거
- **PCA(주성분 분석) 적용**: 주요 주성분 5개를 선택하여 데이터 차원 축소

### 🔹 데이터 정규화
- **StandardScaler**를 사용하여 변수별 스케일 조정
</br>

## 🤖 모델링 및 학습
### 🔹 사용한 알고리즘
| 알고리즘 | 특징 |
|------------|------------------------------------------------------|
| **랜덤포레스트 회귀** | 비선형 데이터에서도 높은 성능을 보이는 모델 |
| **XGBoost 회귀** | 트리 기반의 강력한 부스팅 기법 적용 |
| **CatBoost 회귀** | 범주형 변수를 효과적으로 다루는 부스팅 모델 |
| **인공신경망 (ANN)** | 다층 퍼셉트론 구조로 비만도를 예측 |
| **스태킹 앙상블** | 여러 개의 모델을 결합하여 성능 향상 |
</br>

### 🔹 하이퍼파라미터 최적화
- **베이지안 최적화 기법**을 사용하여 최적 하이퍼파라미터 탐색
- **교차 검증(k-fold Cross Validation)**을 통해 일반화 성능 평가
</br>

### 🔹 최종 모델 선정
- 최적 성능을 보인 모델: **CatBoost 회귀 + 스태킹 앙상블**
- **최고 성능 모델의 RMSE**: 2.34
</br>

## 📈 결과 분석 및 결론
### 🔹 주요 결과
- **CatBoost 회귀 모델이 가장 높은 성능을 기록**
- **스태킹 앙상블 모델**을 적용했을 때 **RMSE 감소 효과** 확인
- **체지방량, 왼팔체지방량, 근육량**이 비만도 예측에 가장 중요한 변수로 나타남
</br>

### 🔹 한계점 및 개선 방향
- PCA 적용 후 일부 정보 손실 발생 가능
- 추가적인 데이터 확보 및 **심층 신경망(DNN) 적용 가능성 탐색** 필요
- 다양한 비만도 관련 지표 추가 분석 가능 (예: 허리둘레, 복부지방률 등)
</br>

### 프로젝트 폴더 구조
```bash
bmi_prediction/
├── data/
│   ├── inbody_dataset.csv    # 인바디 데이터셋 (비만도 예측용)
├── data_processing.py        # 데이터 전처리 관련 코드
├── feature_engineering.py    # 특성 공학 (Feature Engineering) 코드
├── global_variables.py       # 프로젝트 내 전역 변수 설정 파일
├── main.py                   # 메인 실행 파일 (전체 파이프라인 실행)
├── model_evaluation.py       # 모델 평가 스크립트
├── model_training.py         # 모델 학습 스크립트
├── requirements.txt          # 프로젝트 실행을 위한 패키지 목록
└── README.md                 # 프로젝트 설명 및 실행 방법 안내
```
</br>

## 🚀 프로젝트 실행 방법
### 🔹 환경 설정 (라이브러리 설치)
```bash
pip install -r requirements.txt
```
</br>

### 🔹 환경 설정 - 한글 폰트 설정 (Matplotlib)
Matplotlib에서 한글 폰트를 정상적으로 출력하기 위해 운영체제(OS)별로 폰트를 설정해야 합니다.

**Mac 사용자의 경우:**
```python
import matplotlib as mpl
mpl.rcParams['font.family'] = 'AppleGothic'
```

**Windows 사용자의 경우:**
```python
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Malgun Gothic'
```

### 🔹 실행 방법
```bash
python3 main.py
```
</br>

## 📎 참고 자료
- **공공데이터포털 - 인바디 측정 데이터**: [링크](https://www.data.go.kr/data/15128989/fileData.do)
