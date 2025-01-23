from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

# ✅ 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def train_transformer():
    # 데이터 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, "data", "data_resampled.csv")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ 데이터 파일이 없습니다: {DATA_PATH}\n📌 `preprocessing.py`를 실행하여 데이터를 생성하세요.")
    
    # ✅ 데이터 불러오기
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 전처리된 데이터 로드 성공: {DATA_PATH}")
    print(df.info())
    
    # ✅ 시간 변환 및 정렬
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].astype(np.int64) // 10**9 // 3600  # 초 단위에서 시간 단위로 변환
    
    # ✅ Feature 선택
    features = ['latitude', 'longitude', 'depth', 'mag', 'time']
    df = df[features]
    
    # ✅ 데이터 분할 (80% 훈련, 10% 검증, 10% 테스트)
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.1)
    
    X_train = df[:train_size]
    X_val = df[train_size:train_size + val_size]
    X_test = df[train_size + val_size:]
    
    y_train = X_train.pop('mag')
    y_val = X_val.pop('mag')
    y_test = X_test.pop('mag')
    
    print(f"X_train.shape BEFORE: {X_train.shape}")  # (batch_size, feature_dim)

    # ✅ `X_train`을 3D로 변환 (sequence_length=1 추가)
    X_train = np.expand_dims(X_train, axis=1)  # (batch_size, feature_dim) → (batch_size, 1, feature_dim)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    print(f"X_train.shape AFTER: {X_train.shape}")  # (batch_size, 1, feature_dim)

    # ✅ Transformer 모델 구성
    inputs = Input(shape=(1, X_train.shape[2]))  # 🔹 sequence_length=1로 설정
    x = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    x = LayerNormalization()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    outputs = Dense(1)(x)  # 🔹 단일 출력으로 변경

    model = Model(inputs, outputs)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    
    # ✅ 모델 학습
    history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val), verbose=1)
    
    # ✅ 모델 평가
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {loss:.2f}')
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'R-squared (R²): {r2:.2f}')
    
    # ✅ 학습 손실 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model