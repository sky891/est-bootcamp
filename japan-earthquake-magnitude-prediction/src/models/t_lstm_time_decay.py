from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# ✅ 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def train_t_lstm_time_decay():
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
    df['time'] = pd.to_datetime(df['time'])  # 🔹 문자열 → datetime 변환
    df['time_decay'] = df['time'].diff().dt.total_seconds()  # 🔹 시간 차이를 초 단위로 변환
    df['time_decay'] = np.exp(-df['time_decay'].fillna(0) / 100000)  # 🔹 감쇠 계산

    # ✅ 스케일링
    scaler = StandardScaler()
    columns_to_scale = ['latitude', 'longitude', 'depth', 'rms', 'time_decay']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # ✅ Feature 선택
    features = ['latitude', 'longitude', 'depth', 'rms', 'time_decay']
    target = 'mag'

    X = df[features]
    y = df[target]
    time_decay = df['time_decay']
    timesteps = 30

    # ✅ 데이터 분할
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.1)

    # ✅ 시퀀스 생성 함수
    def create_sequences(X, y, time_decay, timesteps):
        X_seq = np.array([X.iloc[i:i+timesteps].values for i in range(len(X) - timesteps)], dtype=np.float32)
        y_seq = np.array([y.iloc[i+timesteps] for i in range(len(y) - timesteps)], dtype=np.float32)
        time_decay_seq = np.array([time_decay.iloc[i:i+timesteps].values for i in range(len(y) - timesteps)], dtype=np.float32).reshape(-1, timesteps, 1)
        return X_seq, y_seq, time_decay_seq

    # ✅ 훈련, 검증, 테스트 데이터 생성
    X_train, y_train, time_decay_train = create_sequences(X[:train_size], y[:train_size], time_decay[:train_size], timesteps)
    X_val, y_val, time_decay_val = create_sequences(X[train_size:train_size + val_size], y[train_size:train_size + val_size], time_decay[train_size:train_size + val_size], timesteps)
    X_test, y_test, time_decay_test = create_sequences(X[train_size + val_size:], y[train_size + val_size:], time_decay[train_size + val_size:], timesteps)

    # ✅ 데이터 타입 변환 (float32)
    X_train, y_train, time_decay_train = X_train.astype(np.float32), y_train.astype(np.float32), time_decay_train.astype(np.float32)
    X_val, y_val, time_decay_val = X_val.astype(np.float32), y_val.astype(np.float32), time_decay_val.astype(np.float32)
    X_test, y_test, time_decay_test = X_test.astype(np.float32), y_test.astype(np.float32), time_decay_test.astype(np.float32)

    print(f"✅ 데이터 크기 확인: X_train={X_train.shape}, time_decay_train={time_decay_train.shape}, y_train={y_train.shape}")

    # ✅ LSTM 모델 구성
    X_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='X_input')
    time_input = Input(shape=(X_train.shape[1], 1), name='time_input')
    merged_input = Lambda(lambda x: x[0] * x[1])([X_input, time_input])

    lstm_out_1 = LSTM(128, return_sequences=True)(merged_input)
    lstm_out_2 = LSTM(64, return_sequences=False)(lstm_out_1)
    output = Dense(1, activation='linear', name='output')(lstm_out_2)

    model = Model(inputs=[X_input, time_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # ✅ 학습 진행
    history = model.fit(
        [X_train, time_decay_train], y_train,
        validation_data=([X_val, time_decay_val], y_val),
        epochs=10, batch_size=32,
        verbose=1
    )

    # ✅ 모델 평가
    y_pred = model.predict([X_test, time_decay_test])
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Test MSE: {mse}, Test MAE: {mae}, Test R2 Score: {r2}")

    # ✅ 손실 그래프
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # ✅ 예측값 그래프
    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Actual vs Predicted Magnitude')
    plt.legend()
    plt.show()

    return model