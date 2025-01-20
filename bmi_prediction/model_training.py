import global_variables as gv
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import tensorflow as tf

# ✅ 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# ✅ 🔥 여기서 models 변수를 최상단에서 정의
models = {
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "XGBRegressor": XGBRegressor(n_estimators=190, max_depth=4, learning_rate=0.1319, subsample=0.7644, colsample_bytree=0.7071, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=191, learning_rate=0.126, max_depth=4, min_samples_leaf=2, min_samples_split=5, random_state=42),
    "LGBMRegressor": LGBMRegressor(n_estimators=178, learning_rate=0.142, max_depth=8, min_child_samples=8, num_leaves=38, subsample=0.714, random_state=42),
    "CatBoostRegressor": CatBoostRegressor(depth=7, learning_rate=0.167, iterations=250, random_state=42, verbose=0),
    "Ridge": Ridge(alpha=0.001, random_state=42),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=10, random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, p=2, weights='distance')
}

def build_ann():
    """
    인공신경망(ANN) 모델 생성
    """
    if gv.X_train_pca is None:
        raise ValueError("❌ PCA 변환이 먼저 수행되지 않았습니다. feature_engineering.py에서 PCA 변환 후 실행하세요.")

    model = Sequential([
        Dense(64, activation='relu', input_shape=(gv.X_train_pca.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """
    모델 학습 및 평가 함수
    """
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"< {model_name} >")
        print(f"MSE : {mse:.3f}")
        print(f"RMSE : {rmse:.3f}")
        print(f"결정계수 : {r2:.3f}\n")

        # ✅ 학습 결과를 global_variables에 저장
        setattr(gv, f"y_pred_{model_name.replace(' ', '_')}", y_pred)

    except Exception as e:
        print(f"❌ 모델 {model_name} 실행 중 오류 발생: {e}")

def train_models():
    """
    모든 모델을 학습 및 평가
    """
    print("🚀 Training models...")

    if gv.X_train_pca is None or gv.y_train is None:
        raise ValueError("❌ PCA 변환이 먼저 수행되지 않았거나 데이터가 로드되지 않았습니다.")

    for name, model in models.items():
        print(f"🚀 Training {name}...")
        train_and_evaluate(model, gv.X_train_pca, gv.y_train, gv.X_test_pca, gv.y_test, name)

    # ✅ 스태킹 앙상블 모델 추가
    stacking_estimators = [
        ('ridge', Ridge()),
        ('gbr', GradientBoostingRegressor()),
        ('cat', CatBoostRegressor(verbose=0)),
        ('xgb', XGBRegressor())
    ]

    stacking_regressor = StackingRegressor(estimators=stacking_estimators, final_estimator=RandomForestRegressor(random_state=42))
    print("🚀 Training StackingRegressor...")
    train_and_evaluate(stacking_regressor, gv.X_train_pca, gv.y_train, gv.X_test_pca, gv.y_test, "StackingRegressor")

    print("✅ 모든 모델 학습 완료.")
