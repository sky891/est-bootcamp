import data_processing
import feature_engineering
import model_training
import model_evaluation

if __name__ == "__main__":
    print("🚀 Running data processing...")
    data_processing.load_data("inbody_dataset.csv")  # ✅ 데이터 로드 및 전처리 실행

    print("🚀 Running feature engineering...")
    feature_engineering.apply_pca()  # ✅ PCA 실행 (PCA 적용 후 데이터 변환)

    print("🚀 Running visualization...")
    model_evaluation.visualize_data()  # ✅ 시각화 함수 실행 (전처리된 데이터 확인)
    model_evaluation.plot_pca_variance()  # ✅ PCA 변환 후 분산 설명률 확인

    print("🚀 Running model training...")
    model_training.train_models()  # ✅ 모델 학습 실행

    print("🚀 Running model evaluation...")  # 🔥 평가 실행 추가
    model_evaluation.evaluate_models()  # ✅ 모델 평가 실행

    print("✅ 모든 작업 완료!")