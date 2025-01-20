# 기본 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn 관련 라이브러리
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# XGBoost, LightGBM, CatBoost 관련 라이브러리
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# TensorFlow 및 Keras 관련 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 전역 변수 파일 import
from model_training import train_and_evaluate, models  # 모델 학습 함수 import
import model_training as mt

import global_variables as gv

# 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# 데이터 전처리: PCA 변환 적용
# -----------------------------
def apply_pca(n_components=5):
    """
    PCA 변환을 수행하고 gv에 저장하는 함수
    """
    if gv.X_train_scaled is None or gv.X_test_scaled is None:
        raise ValueError("❌ X_train_scaled 또는 X_test_scaled가 정의되지 않았습니다. 먼저 데이터 전처리를 수행하세요.")

    pca = PCA(n_components=n_components)
    gv.X_train_pca = pca.fit_transform(gv.X_train_scaled)
    gv.X_test_pca = pca.transform(gv.X_test_scaled)
    
    # ✅ PCA 분산 설명률 저장
    gv.explained_variance_ratio = pca.explained_variance_ratio_

    print("✅ PCA 변환 완료.")

# -----------------------------
# 신경망 모델 학습 및 평가
# -----------------------------
def build_nn():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(gv.X_train_pca.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------------
# 특성 중요도 시각화 함수 정의
# -----------------------------
def plot_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importances)), importances, color='skyblue')
        plt.xlabel('중요도')
        plt.title(f'{model_name} 특성 중요도')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print(f"{model_name}은 feature_importances_ 속성이 없습니다.")

# -----------------------------
# 실행 코드 (if __name__ == "__main__")
# -----------------------------
if __name__ == "__main__":
    print("🚀 PCA 적용 중...")
    apply_pca()  # ✅ PCA 먼저 실행

    # 전역 변수에서 가져오기
    X_train_pca = gv.X_train_pca
    X_test_pca = gv.X_test_pca
    y_train = gv.y_train
    y_test = gv.y_test

    print("🚀 모델 학습 시작...")
    for name, model in models.items():
        train_and_evaluate(model, X_train_pca, y_train, X_test_pca, y_test, name)

    # 신경망 모델 학습
    print("🚀 신경망 모델 학습 중...")
    nn_model = build_nn()
    nn_model.fit(X_train_pca, y_train, validation_data=(X_test_pca, y_test), epochs=50, batch_size=32, verbose=0)
    
    # 예측 및 평가
    y_pred_nn = nn_model.predict(X_test_pca).ravel()
    print(f"< PCA 적용 신경망 모델 >")
    print(f"MSE : {mean_squared_error(y_test, y_pred_nn):.3f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_nn)):.3f}")
    print(f"결정계수 : {r2_score(y_test, y_pred_nn):.3f}\n")

    # 특성 중요도 시각화
    for name, model in models.items():
        plot_feature_importance(model, name)

    print("✅ Feature Engineering 완료!")