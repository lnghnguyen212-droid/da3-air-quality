import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_and_save_model():
    print("Đang tạo dữ liệu mô phỏng để huấn luyện...")
    # Tạo dữ liệu giả lập chuẩn theo cấu trúc Beijing Air Quality
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'PM10': np.random.uniform(10, 200, n_samples),
        'SO2': np.random.uniform(2, 50, n_samples),
        'NO2': np.random.uniform(10, 100, n_samples),
        'CO': np.random.uniform(100, 3000, n_samples),
        'O3': np.random.uniform(10, 150, n_samples),
        'TEMP': np.random.uniform(-10, 35, n_samples),
        'PRES': np.random.uniform(990, 1030, n_samples),
        'DEWP': np.random.uniform(-20, 25, n_samples),
        'RAIN': np.random.uniform(0, 10, n_samples),
        'WSPM': np.random.uniform(0.5, 8, n_samples)
    })
    y = X['PM10'] * 0.6 + X['NO2'] * 0.3 + np.random.normal(0, 5, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 2. Ensemble Learning (Voting Regressor kết hợp Random Forest & Gradient Boosting)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    ensemble_model = VotingRegressor(estimators=[('rf', rf_model), ('gb', gb_model)])

    print("Đang huấn luyện mô hình Ensemble Learning...")
    ensemble_model.fit(X_train, y_train)

    y_pred = ensemble_model.predict(X_test)
    print(f"Hoàn tất! R2 Score: {r2_score(y_test, y_pred):.4f}")

    # Lưu mô hình vào thư mục models/
    os.makedirs("models", exist_ok=True)
    model_path = "models/air_quality_rf_ensemble.pkl"
    joblib.dump(ensemble_model, model_path)
    print(f"Đã lưu mô hình thành công tại: {model_path}")

if __name__ == "__main__":
    train_and_save_model()