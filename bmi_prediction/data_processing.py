import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import global_variables as gv

def load_data(file_name):
    base_path = os.path.join(os.getcwd(), "data")
    file_path = os.path.join(base_path, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    print("✅ 데이터 로드 중...")

    # 데이터 로드
    inbody_data = pd.read_csv(file_path)
    inbody_data_copy = inbody_data.copy()
    

    # 숫자가 아닌 데이터 제거
    columns_to_remove = ['ID', '날짜', '시간']
    for col in columns_to_remove:
        if col in inbody_data_copy.columns:
            inbody_data_copy.drop(columns=[col], inplace=True)

    # 결측치 처리
    numeric_cols = inbody_data_copy.select_dtypes(include=['number']).columns
    inbody_data_copy[numeric_cols] = inbody_data_copy[numeric_cols].fillna(inbody_data_copy[numeric_cols].median())

    # 범주형 데이터 처리
    categorical_cols = inbody_data_copy.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        inbody_data_copy[col] = le.fit_transform(inbody_data_copy[col])
        label_encoders[col] = le
    
    # ✅ 전처리된 데이터 저장
    gv.inbody_data = inbody_data_copy

    # X, y 분리
    X = inbody_data_copy.iloc[:, :-1]
    y = inbody_data_copy.iloc[:, -1]

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 스케일링 적용
    scaler = StandardScaler()
    gv.X_train_scaled = scaler.fit_transform(X_train)
    gv.X_test_scaled = scaler.transform(X_test)
    gv.y_train = y_train
    gv.y_test = y_test

    print("✅ 데이터 로드 및 전처리 완료.")

if __name__ == "__main__":
    load_data("inbody_dataset.csv")
