import global_variables as gv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import os

# ✅ TensorFlow 로그 숨기기 (선택)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: 모든 로그, 1: 정보 로그 제외, 2: 경고만, 3: 오류만 표시

# ✅ 한글 폰트 설정 (Mac 사용자)
mpl.rcParams['font.family'] = 'AppleGothic'

# ✅ 한글 폰트 설정 (Windows 사용자)
# import platform
# if platform.system() == 'Windows':
#     mpl.rcParams['font.family'] = 'Malgun Gothic'
# elif platform.system() == 'Darwin':  # macOS
#     mpl.rcParams['font.family'] = 'AppleGothic'

# ✅ 음수 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False


def visualize_data():
    """
    데이터 시각화 실행
    """
    print("🚀 Visualizing dataset...")

    # inbody_data 가져오기
    inbody_data = gv.inbody_data

    # Heatmap
    plt.figure(figsize=(20, 12))
    sns.heatmap(inbody_data.isna(), cmap='viridis', cbar=False)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()

    # 박스플롯
    plt.boxplot(inbody_data['내장지방레벨'], labels=['내장지방레벨'])

    # 상관관계 히트맵
    inbody = inbody_data.drop(['성별', '비만도'], axis=1)  # 범주형 데이터 제외
    corr_matrix = inbody.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.show()


def plot_pca_variance():
    """
    PCA 분산 설명률 시각화
    """
    explained_variance_ratio = gv.explained_variance_ratio

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('주성분')
    plt.ylabel('분산 설명률')
    plt.title('주성분의 분산 설명률')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.show()


def evaluate_models():
    """
    모델 평가 및 시각화 실행
    """
    print("🚀 Running model evaluation...")

    for model_name in [
        "RandomForestRegressor", "XGBRegressor", "GradientBoostingRegressor",
        "LGBMRegressor", "CatBoostRegressor", "Ridge",
        "DecisionTreeRegressor", "KNeighborsRegressor", "StackingRegressor"
    ]:
        y_pred_name = f"y_pred_{model_name.replace(' ', '_')}"
        
        if hasattr(gv, y_pred_name):
            y_pred = getattr(gv, y_pred_name)
            print(f"📊 {model_name} 평가 중...")

            plt.figure(figsize=(12, 6))
            sns.kdeplot(gv.y_test, label="실제값", color="blue", shade=False)
            sns.kdeplot(y_pred, label=f"{model_name} 예측값", shade=False)
            plt.title(f"{model_name} 예측값 분포 비교")
            plt.xlabel("값")
            plt.ylabel("밀도")
            plt.legend()
            plt.grid(True)
            plt.show()

    print("✅ 모델 평가 및 시각화 완료!")


if __name__ == "__main__":
    visualize_data()
    plot_pca_variance()
    evaluate_models()