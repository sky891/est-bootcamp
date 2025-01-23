import os
import sys
import pandas as pd
from src.data_preprocessing import data_preprocessing  
from src.models.rnn_model import train_rnn
from src.models.t_lstm_time_decay import train_t_lstm_time_decay
from src.models.t_lstm_time_interval import train_t_lstm_time_interval
from src.models.transformer_model import train_transformer

# ✅ 원본 데이터 & 전처리된 데이터 경로
RAW_DATA_PATH = os.path.join("data", "Japan earthquakes 2001 - 2018.csv")
PROCESSED_DATA_PATH = os.path.join("data", "data_resampled.csv")

def check_and_run_preprocessing():
    """데이터가 없거나 원본 데이터가 변경된 경우 전처리 실행"""
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ {PROCESSED_DATA_PATH} 파일이 존재하지 않습니다.")
        print("⚠️ 데이터 전처리를 실행합니다...")
        data_preprocessing()
        print("✅ 데이터 전처리 완료!")
    else:
        try:
            raw_mtime = os.path.getmtime(RAW_DATA_PATH)
            processed_mtime = os.path.getmtime(PROCESSED_DATA_PATH)

            if raw_mtime > processed_mtime:
                print("🔄 원본 데이터가 수정되었습니다. 다시 전처리를 실행합니다...")
                data_preprocessing()
                print("✅ 데이터 전처리 완료!")
            else:
                print("✅ 최신 데이터가 존재합니다. 전처리 과정 건너뜁니다.")
        except FileNotFoundError:
            print("❌ 원본 데이터 파일이 없습니다! 데이터를 확인해주세요.")
            exit(1)

def main():
    # ✅ 1️⃣ 데이터 전처리 체크 및 실행
    check_and_run_preprocessing()

    # ✅ 2️⃣ 전처리된 데이터 로드
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"✅ 전처리된 데이터 컬럼 확인: {df.columns.tolist()}")

    # ✅ 3️⃣ 존재하는 One-Hot Encoding 컬럼만 사용
    one_hot_columns = [col for col in df.columns if col.startswith(("Plate_", "Region_", "magType_"))]
    
    if not one_hot_columns:
        raise ValueError("🚨 One-Hot Encoding 컬럼이 존재하지 않습니다! `data_resampled.csv`를 확인하세요.")
    
    print(f"🔹 One-Hot Encoding 적용된 컬럼: {one_hot_columns}")

    # ✅ 4️⃣ 모델 학습 실행
    print("🚀 1. RNN 모델 학습 시작...")
    train_rnn()

    print("🚀 2. Transformer 모델 학습 시작...")
    train_transformer()

    print("🚀 3. T-LSTM (Time Interval) 모델 학습 시작...")
    train_t_lstm_time_interval()

    print("🚀 4. T-LSTM (Time Decay) 모델 학습 시작...")
    train_t_lstm_time_decay()

    print("✅ 모든 학습 과정 완료.")

if __name__ == "__main__":
    main()
