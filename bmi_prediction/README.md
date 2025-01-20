# Body Composition-Based BMI Prediction
이 프로젝트는 체성분 데이터를 활용하여 비만도를 예측하는 머신러닝 모델을 구현한 것입니다. 체중, 키, 나이만으로 계산한 비만도는 신뢰성이 떨어질 수 있기 때문에, 더 다양한 체성분 데이터를 이용하여 신뢰성 높은 예측 모델을 개발했습니다.
</br>
</br>
</br>

## Table of Contents
1. [Data](#Data)
2. [EDA](#eda)
3. [Modeling](#modeling)
4. [Results](#results)
5. [Discussion](#discussion)
</br>
</br>
</br>


## Data
<img width="607" alt="image" src="https://github.com/user-attachments/assets/2f4b51a7-9ff5-4c4a-9b43-d4c942689a42" />
</br>
</br>

데이터 출처 : [공공데이터 포털 - 세종특별자치시 인바디 측정내역](https://www.data.go.kr/data/15128989/fileData.do)
</br>

데이터 구성
  - **총 데이터 크기** : 21,120 rows × 37 columns
  - **수치형 컬럼** : 32개
  - **범주형 컬럼** : 5개
</br>
</br>
</br>



## EDA
1. **데이터 전처리**:
   - 사용자 고유번호, 측정일자, 행정동명 등 비만도와 관련이 없는 컬럼 제거.
   - 출생년도를 나이로 변환.
   - 성별을 수치형 데이터로 변환 (Label Encoding).
2. **결측치 처리**:
   - 히트맵을 통해 결측치 시각화 후 처리.
3. **이상치 확인**:
   - 이상치 비율이 5% 이하로 확인되어 별도의 제거 작업을 하지 않음.
4. **다중공선성 문제 해결**:
   - VIF(분산 팽창 요인) 계산 후 다중공선성이 큰 변수 제거.
   - PCA(주성분 분석) 기법을 적용하여 주성분으로 데이터를 압축.
</br>
</br>
</br>



## Modeling
- 사용한 머신러닝 알고리즘:
  1. 랜덤 포레스트 회귀
  2. XGBoost 회귀
  3. CatBoost 회귀
  4. 인공신경망 회귀
  5. 스태킹 앙상블 모델 (메타 모델: 랜덤 포레스트 및 인공신경망)
- **하이퍼파라미터 튜닝**:
  - 베이지안 최적화 기법을 사용하여 최적 하이퍼파라미터 탐색.
- **평가 지표**:
  - 결정계수(R²), MSE, RMSE를 기준으로 모델 성능을 평가.
</br>
</br>
</br>



## Results
<img width="598" alt="image" src="https://github.com/user-attachments/assets/09a905f0-2e4a-42a4-b5d0-56f204253f56"/>
</br>

### 상위 4개 모델의 성능 비교
- **CatBoost** : 예측에 가장 중요한 변수로 체지방량, 왼팔 체지방량, 근육량을 선정
- **인공신경망** : 성별, 체질량지수(BMI), 복부지방률이 주요 변수로 확인됨.
- **스태킹 앙상블 (메타 모델: 랜덤 포레스트)** : 인공신경망 예측값이 가장 높은 기여도를 보임.
- **스태킹 앙상블 (메타 모델: 인공신경망)** : 모든 예측 모델의 예측값을 고르게 활용하여 높은 성능을 보임.
</br>
</br>
</br>

## Discussion
- 실제 데이터를 사용하면서 다양한 문제점을 파악할 수 있었으며, 특히 모델 학습 및 하이퍼파라미터 튜닝에 많은 시간이 소요되었습니다.
- PCA 기법을 통해 다중공선성 문제를 일부 해결했으나, 비선형 변수를 처리하는 데 한계가 있었음을 확인했습니다.
- 시간 관계상 PCA 외 다른 기법으로 다중공선성 문제를 해결하지 못한 점이 아쉬웠습니다.
</br>
</br>
</br>

## Usage

### Setting Korean Font (한글 폰트 설정)
Matplotlib에서 한글 폰트를 정상적으로 출력하기 위해 운영체제(OS)별로 폰트를 설정해야 합니다.
</br>
</br>
</br>


### model_evaluation.py

- **Mac 사용자의 경우**:
  ```python
  import matplotlib as mpl
  mpl.rcParams['font.family'] = 'AppleGothic'
</br> 

- **Window 사용자의 경우**:
  ```python
  import matplotlib as mpl
  mpl.rcParams['font.family'] = 'Malgun Gothic'
</br> 
