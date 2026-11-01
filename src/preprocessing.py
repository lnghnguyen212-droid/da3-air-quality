import pandas as pd
import numpy as np

def clean_data(df):
    """Xử lý missing value và chuẩn hóa dữ liệu quan trắc"""
    # Điền giá trị thiếu bằng phương pháp nội suy forward fill / backward fill
    df = df.ffill().bfill()
    
    # Loại bỏ các cột không dùng làm đặc trưng dự báo nếu có
    features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    target = 'PM2.5'
    
    # Giữ lại các cột hợp lệ
    valid_cols = [c for c in features if c in df.columns]
    return df[valid_cols], df[target] if target in df.columns else None