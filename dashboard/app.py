
import os
from datetime import datetime

import joblib
import requests
import streamlit as st

import aq_core as core

API_LATEST = os.getenv(
    "AQ_API_LATEST", "https://beijing-observation-simulator.onrender.com/api/v1/live/latest"
)
MODEL_PATH = os.getenv("AQ_MODEL_PATH", "models/air_quality_rf_ensemble.pkl")
REFRESH_SECONDS = 30

st.set_page_config(page_title="Beijing Air Quality Forecast", page_icon="🍃", layout="centered")
st.markdown(core.CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- dữ liệu
@st.cache_data(ttl=20, show_spinner=False)
def fetch_live():
    """Gọi API quan trắc. Dịch vụ free trên Render có thể 'ngủ' nên cho timeout dài."""
    try:
        res = requests.get(API_LATEST, timeout=25)
        res.raise_for_status()
        live = core.parse_live(res.json())
        return live
    except Exception as exc:  # noqa: BLE001
        live = core.Live()
        live.message = f"Không kết nối được trạm quan trắc ({type(exc).__name__})."
        return live


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------- tiêu đề
def header(live: core.Live):
    state = "" if live.ok else " off"
    when = live.time or datetime.now().strftime("%d/%m/%Y %H:%M")
    text = f"Trực tuyến · {when}" if live.ok else "Chưa có dữ liệu trạm"
    st.markdown(
        f"""<div class="brand">
  <div><h1>🍃 Beijing <b>Air Quality</b></h1>
  <small>Quan trắc realtime và dự báo PM2.5 bằng Random Forest &amp; Ensemble Learning</small></div>
  <div class="live"><span class="dot{state}"></span>{text}</div></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- khối realtime
def live_section():
    live = fetch_live()
    v = live.values
    aqi = v.get("AQI")
    cat = core.category(aqi)

    if not live.ok:
        st.warning(live.message or "Chưa nhận được dữ liệu từ trạm. Dịch vụ có thể đang khởi động, thử lại sau ít phút.")
        return live

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        st.markdown(
            f"""<div class="card"><h3>Chất lượng không khí hiện tại{f' · {live.station}' if live.station else ''}</h3>
<div class="hero">{core.gauge_svg(aqi)}
<div><span class="pill" style="background:{cat['color']}">{cat['label']}</span>
<div class="advice">{cat['advice']}</div></div></div></div>""",
            unsafe_allow_html=True,
        )
    with right:
        tiles = [
            core.kpi_html("PM2.5", f"{v['PM2.5']:.0f}", "µg/m³", "Bụi mịn", "#38bdf8"),
            core.kpi_html("Nhiệt độ", f"{v['TEMP']:.1f}" if "TEMP" in v else "—", "°C", "", "#fb923c"),
            core.kpi_html("Độ ẩm", f"{v['HUMIDITY']:.0f}" if "HUMIDITY" in v else "—", "%", "Suy ra từ nhiệt độ &amp; điểm sương" if "DEWP" in v else "", "#60a5fa"),
            core.kpi_html("Gió", f"{v['WSPM']:.1f}" if "WSPM" in v else "—", "m/s", "", "#a78bfa"),
        ]
        st.markdown(f'<div class="kpis">{"".join(tiles)}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="card" style="margin-top:12px"><h3>Nồng độ chất ô nhiễm</h3>{core.pollutant_bars_html(v)}</div>',
        unsafe_allow_html=True,
    )
    return live


# ---------------------------------------------------------------- dự báo
FIELDS = [  # (khoá, nhãn, mặc định, bước, định dạng)
    ("PM10", "PM10 (µg/m³)", 45.0, 1.0, "%.1f"),
    ("SO2", "SO₂ (µg/m³)", 10.0, 1.0, "%.1f"),
    ("NO2", "NO₂ (µg/m³)", 25.0, 1.0, "%.1f"),
    ("CO", "CO (µg/m³)", 500.0, 50.0, "%.0f"),
    ("O3", "O₃ (µg/m³)", 35.0, 1.0, "%.1f"),
    ("TEMP", "Nhiệt độ (°C)", 24.0, 0.5, "%.1f"),
    ("PRES", "Áp suất (hPa)", 1012.0, 1.0, "%.1f"),
    ("DEWP", "Điểm sương (°C)", 15.0, 0.5, "%.1f"),
    ("RAIN", "Lượng mưa (mm)", 0.0, 0.1, "%.1f"),
    ("WSPM", "Tốc độ gió (m/s)", 2.5, 0.1, "%.1f"),
]


def fill_from_station():
    live = fetch_live()
    for key, _, default, _, _ in FIELDS:
        val = live.get(key)
        if val is not None:
            st.session_state[f"in_{key}"] = float(val)


def predict_section(live: core.Live):
    st.markdown('<div class="sec">Dự báo nồng độ PM2.5</div>', unsafe_allow_html=True)

    for key, _, default, _, _ in FIELDS:  # lần đầu: ưu tiên số liệu trạm
        st.session_state.setdefault(f"in_{key}", float(live.get(key, default)))

    with st.form("predict_form", border=False):
        cols = st.columns(5)
        for i, (key, label, _, step, fmt) in enumerate(FIELDS):
            with cols[i % 5]:
                st.number_input(label, step=step, format=fmt, key=f"in_{key}")
        c1, c2 = st.columns([1, 2.2])
        with c1:
            st.form_submit_button("Dùng số liệu trạm", on_click=fill_from_station, disabled=not live.ok,
                                  use_container_width=True)
        with c2:
            submitted = st.form_submit_button("Dự báo PM2.5", type="primary", use_container_width=True)

    if not submitted:
        return

    model = load_model()
    if model is None:
        st.error(f"Chưa tìm thấy mô hình tại `{MODEL_PATH}`. Hãy chạy `python src/train_model.py` trước.")
        return

    inputs = {k: float(st.session_state[f"in_{k}"]) for k, *_ in FIELDS}
    for msg in core.validate_inputs(inputs):
        st.warning(msg)

    try:
        pred = core.predict_pm25(model, inputs, extras=live.values)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không thể dự báo: {exc}")
        return

    cat = core.category(pred.aqi)
    st.markdown(
        f"""<div class="card"><h3>Kết quả dự báo</h3>
<div class="result"><div><div class="r-big">{pred.pm25:.1f}<span>µg/m³ PM2.5</span></div>
<div class="r-sub">AQI tương ứng ≈ {pred.aqi:.0f}</div></div>
<div><span class="pill" style="background:{cat['color']}">{cat['label']}</span>
<div class="advice">{cat['advice']}</div></div></div>
{core.result_range_html(pred)}
<div class="note">{_note(pred, live)}</div></div>""",
        unsafe_allow_html=True,
    )


def _note(pred: core.Prediction, live: core.Live) -> str:
    parts = []
    if pred.low is not None:
        what = "các cây quyết định" if pred.spread_kind == "tree" else "các mô hình thành viên"
        parts.append(f"Khoảng dao động thể hiện mức đồng thuận giữa {what} (không phải sai số thống kê tuyệt đối).")
    if live.ok and "PM2.5" in live.values:
        d = pred.pm25 - live.values["PM2.5"]
        parts.append(f"So với PM2.5 đo được tại trạm ({live.values['PM2.5']:.1f}): {d:+.1f} µg/m³.")
    if pred.missing:
        parts.append("Đặc trưng mô hình cần nhưng form không có, được điền 0: " + ", ".join(pred.missing) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------- chạy
live0 = fetch_live()
header(live0)

if hasattr(st, "fragment"):
    live = st.fragment(run_every=REFRESH_SECONDS)(live_section)()
else:
    live = live_section()
predict_section(live or live0)