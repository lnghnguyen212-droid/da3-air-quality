import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime

# ---------------------------------------------------------
# CẤU HÌNH TRANG WEB
# ---------------------------------------------------------
st.set_page_config(
    page_title="Beijing Air Quality Forecast",
    page_icon="🍃",
    layout="wide"
)

# Giao diện sạch sẽ, thoáng đãng, dễ sử dụng
st.markdown("""
    <style>
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #10b981;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TIÊU ĐỀ CHÍNH
# ---------------------------------------------------------
st.markdown("<p class='main-title'>🍃 Beijing Air Quality Forecast Platform</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hệ thống quan trắc realtime và dự báo chất lượng không khí sử dụng Ensemble Learning</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# KẾT NỐI API REALTIME
# ---------------------------------------------------------
API_LATEST = "https://beijing-observation-simulator.onrender.com/api/v1/live/latest"

@st.cache_data(ttl=15)
def get_live_data():
    try:
        res = requests.get(API_LATEST, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"PM2.5": 22.5, "PM10": 45.0, "TEMP": 24.0, "WSPM": 2.5, "AQI": 68}

data = get_live_data()

# ---------------------------------------------------------
# HIỂN THỊ CHỈ SỐ (METRICS) RÕ RÀNG
# ---------------------------------------------------------
st.subheader("📊 Dữ liệu quan trắc mới nhất từ Trạm")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Chỉ số AQI", value=data.get('AQI', 68), delta="Mức tốt")
with col2:
    st.metric(label="Nồng độ PM2.5", value=f"{data.get('PM2.5', 22.5)} µg/m³")
with col3:
    st.metric(label="Nhiệt độ (TEMP)", value=f"{data.get('TEMP', 24.0)} °C")
with col4:
    st.metric(label="Tốc độ gió (WSPM)", value=f"{data.get('WSPM', 2.5)} m/s")

st.markdown("---")

# ---------------------------------------------------------
# KHU VỰC NHẬP LIỆU DỰ BÁO (DỄ SÀI, TRỰC QUAN)
# ---------------------------------------------------------
st.subheader("🔮 Công cụ Dự báo Nồng độ PM2.5")
st.markdown("Nhập các thông số môi trường bên dưới để mô hình tiến hành dự báo:")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        pm10 = st.number_input("Chỉ số PM10", value=float(data.get('PM10', 45.0)))
        so2 = st.number_input("Chỉ số SO2", value=10.0)
        no2 = st.number_input("Chỉ số NO2", value=25.0)
    with c2:
        co = st.number_input("Chỉ số CO", value=500.0)
        o3 = st.number_input("Chỉ số O3", value=35.0)
        temp = st.number_input("Nhiệt độ (°C)", value=float(data.get('TEMP', 24.0)))
    with c3:
        pres = st.number_input("Áp suất (PRES)", value=1012.0)
        dewp = st.number_input("Điểm sương (DEWP)", value=15.0)
        wspm = st.number_input("Tốc độ gió (WSPM)", value=float(data.get('WSPM', 2.5)))

    submitted = st.form_submit_button("🚀 Thực hiện Dự báo Ngay")

    if submitted:
        if os.path.exists("models/air_quality_rf_ensemble.pkl"):
            model = joblib.load("models/air_quality_rf_ensemble.pkl")
            # Chuẩn bị DataFrame đầu vào đúng thứ tự các đặc trưng
            input_df = pd.DataFrame([[pm10, so2, no2, co, o3, temp, pres, dewp, 0.0, wspm]],
                                    columns=['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'])
            prediction = model.predict(input_df)[0]
            
            st.success("✨ Dự báo hoàn tất thành công!")
            st.metric(label="Kết quả dự báo nồng độ PM2.5", value=f"{prediction:.2f} µg/m³")
        else:
            st.warning("⚠️ Chưa tìm thấy file mô hình trong thư mục `models/`. Hãy đảm bảo bạn đã chạy file `src/train_model.py` trước nhé!")