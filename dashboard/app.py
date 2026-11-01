import streamlit as st
import pandas as pd
import requests
import joblib
import os

st.set_page_config(page_title="Beijing Air Quality Forecast Platform", layout="wide")

st.title("🌬️ Beijing Air Quality Forecast Platform (Realtime)")
st.caption("Mô hình Random Forest & Ensemble Learning dự báo chất lượng không khí")

# Sidebar
st.sidebar.header("Cấu hình & Tùy chọn")
api_url = "https://beijing-observation-simulator.onrender.com/api/v1/live/latest"

st.subheader("📡 Dữ liệu quan trắc Realtime")

if st.button("Tải dữ liệu mới nhất từ API"):
    with st.spinner("Đang kết nối tới dịch vụ quan trắc realtime..."):
        try:
            res = requests.get(api_url, timeout=15)
            if res.status_code == 200:
                data = res.json()
                st.success("Kết nối thành công!")
                st.json(data)
            else:
                st.error(f"Không thể lấy dữ liệu. Mã lỗi: {res.status_code}")
        except Exception as e:
            st.error(f"Lỗi kết nối API: {e}")

# Phần dự báo
st.markdown("---")
st.subheader("🔮 Dự báo nồng độ PM2.5")

if os.path.exists("models/air_quality_rf_ensemble.pkl"):
    model = joblib.load("models/air_quality_rf_ensemble.pkl")
    st.info("Mô hình Ensemble Learning đã sẵn sàng!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pm10 = st.number_input("PM10", value=50.0)
        so2 = st.number_input("SO2", value=10.0)
        no2 = st.number_input("NO2", value=30.0)
    with col2:
        co = st.number_input("CO", value=800.0)
        o3 = st.number_input("O3", value=40.0)
        temp = st.number_input("Nhiệt độ (TEMP)", value=20.0)
    with col3:
        pres = st.number_input("Áp suất (PRES)", value=1013.0)
        dewp = st.number_input("Đểm sương (DEWP)", value=5.0)
        wspm = st.number_input("Tốc độ gió (WSPM)", value=2.0)

    if st.button("Thực hiện dự báo PM2.5"):
        input_data = pd.DataFrame([[pm10, so2, no2, co, o3, temp, pres, dewp, 0.0, wspm]],
                                  columns=['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'])
        prediction = model.predict(input_data)[0]
        st.metric(label="Chỉ số PM2.5 dự báo", value=f"{prediction:.2f} µg/m³")
else:
    st.warning("Chưa tìm thấy mô hình trong thư mục `models/`. Vui lòng chạy file `src/train_model.py` trước!")