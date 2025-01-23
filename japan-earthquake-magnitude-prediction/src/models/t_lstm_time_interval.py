from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os

# ✅ 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def train_t_lstm_time_interval():
    # 데이터 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, "data", "data_resampled.csv")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ 데이터 파일이 없습니다: {DATA_PATH}\n📌 `preprocessing.py`를 실행하여 데이터를 생성하세요.")
    
    # 데이터 불러오기
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 전처리된 데이터 로드 성공: {DATA_PATH}")
    print(df.info())
    
    # 시간 변환 및 정렬
    df = pd.read_csv(DATA_PATH, index_col=0)  # 'time'이 인덱스로 되어있을 가능성이 높음
    df.reset_index(inplace=True)
    
    # 시간 간격 계산
    df['time_interval'] = df['time'].diff().dt.total_seconds().fillna(0)
    df.set_index('time', inplace=True)
    
    # 스케일링
    scaler = StandardScaler()
    columns_to_scale = ['latitude', 'longitude', 'depth', 'rms', 'time_interval']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    X = df.drop('time_interval', axis=1)
    y = df['mag']
    time_interval = df['time_interval']
    timesteps = 30
    
    # 데이터 분할
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.1)
    
    def create_sequences(X, y, timesteps):
        X_seq, y_seq, time_interval_seq = [], [], []
        for i in range(len(X) - timesteps):
            X_seq.append(X.iloc[i:i+timesteps].values)
            y_seq.append(y.iloc[i+timesteps])
            time_interval_seq.append(time_interval.iloc[i:i+timesteps].values)
        return np.array(X_seq), np.array(y_seq), np.array(time_interval_seq).reshape(-1, timesteps, 1)
    
    X_train, y_train, time_interval_train = create_sequences(X[:train_size], y[:train_size], timesteps)
    X_val, y_val, time_interval_val = create_sequences(X[train_size:train_size + val_size], y[train_size:train_size + val_size], timesteps)
    X_test, y_test, time_interval_test = create_sequences(X[train_size + val_size:], y[train_size + val_size:], timesteps)
    
    # 최적 하이퍼파라미터 설정
    epoch = 10
    batch_size = 32
    learning_rate = 0.001
    
    # 입력 레이어
    X_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='X_input')
    time_input = Input(shape=(X_train.shape[1], 1), name='time_input')
    merged_input = Concatenate()([X_input, time_input])
    
    # LSTM 모델
    lstm_out_1 = LSTM(128, return_sequences=True)(merged_input)
    lstm_out_2 = LSTM(64, return_sequences=False)(lstm_out_1)
    output = Dense(1, activation='linear', name='output')(lstm_out_2)
    
    model = Model(inputs=[X_input, time_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    history = model.fit(
        [X_train, time_interval_train], y_train,
        validation_df=([X_val, time_interval_val], y_val),
        epochs=epoch, batch_size=batch_size,
        verbose=1
    )
    
    # 모델 평가
    y_pred = model.predict([X_test, time_interval_test])
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse}, Test MAE: {mae}, Test R2 Score: {r2}")
    
    # 손실 그래프
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # 예측값 그래프
    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title(f'Actual vs Predicted Magnitude')
    plt.legend()
    plt.show()
    
    return model