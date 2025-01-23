from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

# ✅ 랜덤 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def train_rnn():
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
    
    # 컬럼 변환 및 표준화
    scaler = StandardScaler()
    columns_to_scale = ['latitude', 'longitude', 'depth', 'rms']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    # ADF 검정 실행 전에 'mag'이 상수인지 확인
    if df['mag'].nunique() > 1:  # 서로 다른 값이 1개보다 많을 때만 실행
        result = adfuller(df['mag'])
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
    else:
        print("Skipping ADF test: 'mag' column is constant.")
    
    # 라벨 인코딩 추가 (지진 발생 여부)
    df['Earthquake'] = df['mag'].apply(lambda x: '지진 발생' if x > 0 else '지진 미발생')
    df['Earthquake_Label'] = LabelEncoder().fit_transform(df['Earthquake'])
    df.drop('Earthquake', axis=1, inplace=True)
    
    # 데이터셋 생성
    time_steps = 24
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.1)
    
    def create_dataset(data, time_steps=24):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)
    
    # ✅ time 컬럼 제거 (원본 Jupyter 방식과 동일하게 처리)
    if 'time' in df.columns:
        df.drop(columns=['time'], inplace=True)
    
    X_train, y_train = create_dataset(df[:train_size].values, time_steps)
    X_val, y_val = create_dataset(df[train_size:train_size + val_size].values, time_steps)
    X_test, y_test = create_dataset(df[train_size + val_size:].values, time_steps)
    
    # # # 데이터 변환 (float32 적용)
    # # X_train = X_train.astype('float32')
    # # y_train = y_train.astype('float32')
    # # X_val = X_val.astype('float32')
    # # y_val = y_val.astype('float32')
    # # X_test = X_test.astype('float32')
    # # y_test = y_test.astype('float32')

    # # numpy 배열로 변환
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_val = np.array(X_val)
    # y_val = np.array(y_val)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)

    # # RNN 입력 차원 맞추기
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    # X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # 자동으로 feature 개수를 모델 input_shape에 맞춤
    input_dim = X_train.shape[-1]
     
    
    # RNN 모델 구성 및 학습 (첫 번째 모델)
    model = Sequential([
        SimpleRNN(50, activation='tanh', return_sequences=True, input_shape=(time_steps, input_dim), kernel_regularizer=l2(0.03)),
        Dropout(0.3),
        SimpleRNN(50, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(26)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stop])
    
    # 두 번째 데이터셋 생성
    data_df = df.copy()
    train_size_c = int(len(data_df) * 0.8)
    val_size_c = int(len(data_df) * 0.1)
    
    X_train_c, y_train_c = create_dataset(data_df[:train_size_c].values, time_steps)
    X_val_c, y_val_c = create_dataset(data_df[train_size_c:train_size_c + val_size].values, time_steps)
    X_test_c, y_test_c = create_dataset(data_df[train_size_c + val_size_c:].values, time_steps)
    
    # 두 번째 RNN 모델 학습
    model_rnn_c = Sequential([
        SimpleRNN(50, activation='tanh', return_sequences=True, input_shape=(time_steps, 26), kernel_regularizer=l2(0.03)),
        Dropout(0.3),
        SimpleRNN(50, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(26)
    ])
    
    model_rnn_c.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stop_c = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history_rnn_c = model_rnn_c.fit(X_train_c, y_train_c, epochs=15, batch_size=16, validation_data=(X_val_c, y_val_c), callbacks=[early_stop_c])
    
    # 두 번째 모델 평가
    loss_rnn_c = model_rnn_c.evaluate(X_test_c, y_test_c, verbose=0)
    print(f'Test Loss (RNN_C): {loss_rnn_c:.2f}')
    
    # 학습 손실 시각화 (두 번째 모델)
    plt.figure(figsize=(10, 6))
    plt.plot(history_rnn_c.history['loss'], label='Training Loss (RNN_C)')
    plt.plot(history_rnn_c.history['val_loss'], label='Validation Loss (RNN_C)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Over Epochs (RNN_C)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, model_rnn_c