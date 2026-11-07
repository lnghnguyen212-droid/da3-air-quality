"""Beijing Air Quality Forecast – web app Streamlit (một file duy nhất).

Chạy:  streamlit run app.py
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# AQI (HJ 633-2012)
# --------------------------------------------------------------------------
IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500]
BREAKPOINTS = {
    "PM2.5": [0, 35, 75, 115, 150, 250, 350, 500],
    "PM10": [0, 50, 150, 250, 350, 420, 500, 600],
    "SO2": [0, 150, 500, 650, 800, 1600, 2100, 2620],
    "NO2": [0, 100, 200, 700, 1200, 2340, 3090, 3840],
    "O3": [0, 160, 200, 300, 400, 800, 1000, 1200],
    "CO": [0, 5, 10, 35, 60, 90, 120, 150],  # mg/m³
}
# Ngưỡng nồng độ ứng với IAQI = 100 (CO tính theo µg/m³ như dữ liệu gốc)
LIMIT_100 = {k: (v[2] * 1000 if k == "CO" else v[2]) for k, v in BREAKPOINTS.items()}

CATEGORIES = [
    # (AQI tối đa, nhãn, màu, lời khuyên)
    (50, "Tốt", "#22c55e", "Không khí trong lành. Thoải mái hoạt động ngoài trời."),
    (100, "Trung bình", "#eab308",
     "Chấp nhận được. Người rất nhạy cảm nên hạn chế vận động mạnh kéo dài ngoài trời."),
    (150, "Kém", "#f97316",
     "Nhóm nhạy cảm (trẻ em, người già, bệnh hô hấp/tim mạch) nên giảm hoạt động ngoài trời."),
    (200, "Xấu", "#ef4444",
     "Mọi người nên hạn chế ở ngoài trời lâu; nhóm nhạy cảm nên ở trong nhà, đeo khẩu trang N95."),
    (300, "Rất xấu", "#a855f7",
     "Tránh hoạt động ngoài trời. Đóng cửa sổ, dùng máy lọc không khí nếu có."),
    (10_000, "Nguy hại", "#9f1239",
     "Cảnh báo khẩn cấp: ở trong nhà, hạn chế tối đa tiếp xúc với không khí bên ngoài."),
]


def iaqi(pollutant: str, value: float) -> float:
    x = float(value)
    if pollutant == "CO":
        x /= 1000.0
    return float(np.interp(x, BREAKPOINTS[pollutant], IAQI_LEVELS))


def aqi_from_pm25(pm25: float) -> float:
    return iaqi("PM2.5", pm25)


def aqi_from_values(values: dict[str, float]) -> float | None:
    parts = [iaqi(k, v) for k, v in values.items()
             if k in BREAKPOINTS and v is not None and not _isnan(v)]
    return max(parts) if parts else None


def category(aqi: float | None) -> dict:
    if aqi is None or _isnan(aqi):
        return {"label": "Chưa có dữ liệu", "color": "#64748b", "advice": ""}
    for upper, label, color, advice in CATEGORIES:
        if aqi <= upper:
            return {"label": label, "color": color, "advice": advice}
    return {"label": CATEGORIES[-1][1], "color": CATEGORIES[-1][2], "advice": CATEGORIES[-1][3]}


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


# --------------------------------------------------------------------------
# Dữ liệu realtime
# --------------------------------------------------------------------------
ALIASES = {
    "PM2.5": ["pm2.5", "pm25", "pm2_5", "pm_2_5"],
    "PM10": ["pm10", "pm_10"],
    "SO2": ["so2"],
    "NO2": ["no2"],
    "CO": ["co"],
    "O3": ["o3"],
    "TEMP": ["temp", "temperature", "temperature_c"],
    "PRES": ["pres", "pressure"],
    "DEWP": ["dewp", "dew_point", "dewpoint"],
    "RAIN": ["rain", "rainfall", "precipitation"],
    "WSPM": ["wspm", "wind_speed", "windspeed", "wind"],
    "AQI": ["aqi"],
    "HUMIDITY": ["humidity", "rh"],
    "STATION": ["station", "station_name", "site"],
    "TIME": ["timestamp", "time", "datetime", "updated_at", "observed_at"],
}


def _norm(key: str) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum() or ch in "._")


def _flatten(obj, out: dict | None = None) -> dict:
    """Làm phẳng JSON lồng nhau; nếu là list lấy phần tử cuối (bản ghi mới nhất)."""
    out = {} if out is None else out
    if isinstance(obj, list):
        if obj:
            _flatten(obj[-1], out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _flatten(v, out)
            else:
                out.setdefault(_norm(k), v)
    return out


@dataclass
class Live:
    values: dict = field(default_factory=dict)  # khoá chuẩn → số
    station: str | None = None
    time: str | None = None
    ok: bool = False
    message: str = ""

    def get(self, key: str, default=None):
        v = self.values.get(key)
        return default if v is None or _isnan(v) else v


def parse_live(payload) -> Live:
    flat = _flatten(payload)
    live = Live()
    for key, names in ALIASES.items():
        for n in names:
            if n in flat and flat[n] is not None:
                raw = flat[n]
                if key in ("STATION", "TIME"):
                    setattr(live, key.lower(), str(raw))
                else:
                    try:
                        live.values[key] = float(raw)
                    except (TypeError, ValueError):
                        continue
                break
    v = live.values
    # Độ ẩm tương đối từ TEMP & DEWP (công thức Magnus) nếu API không có
    if "HUMIDITY" not in v and "TEMP" in v and "DEWP" in v:
        v["HUMIDITY"] = relative_humidity(v["TEMP"], v["DEWP"])
    # AQI: ưu tiên giá trị từ API, nếu không có thì tính theo chuẩn
    if "AQI" not in v:
        pol = {k: v[k] for k in BREAKPOINTS if k in v}
        a = aqi_from_values(pol)
        if a is not None:
            v["AQI"] = a
    live.ok = "PM2.5" in v
    if not live.ok:
        live.message = "Phản hồi API không chứa PM2.5."
    return live


def relative_humidity(temp_c: float, dewp_c: float) -> float:
    a, b = 17.625, 243.04
    rh = 100 * math.exp(a * dewp_c / (b + dewp_c)) / math.exp(a * temp_c / (b + temp_c))
    return float(min(100.0, max(0.0, rh)))


# --------------------------------------------------------------------------
# Mô hình & dự báo
# --------------------------------------------------------------------------
DEFAULT_FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# Khoảng hợp lý của từng đại lượng (dùng cảnh báo nhập sai đơn vị/ngoại lai)
PLAUSIBLE = {
    "PM10": (0, 1000), "SO2": (0, 500), "NO2": (0, 300), "CO": (0, 10000),
    "O3": (0, 500), "TEMP": (-30, 45), "PRES": (980, 1050), "DEWP": (-40, 35),
    "RAIN": (0, 80), "WSPM": (0, 15),
}


def unpack_model(obj):
    """Chấp nhận estimator thuần hoặc dict {'model':..., 'features':[...]}."""
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("pipeline") or obj.get("estimator")
        feats = obj.get("features") or obj.get("feature_names")
        return model, list(feats) if feats is not None else None
    return obj, None


def model_features(model, stored: list[str] | None) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return stored or DEFAULT_FEATURES


def validate_inputs(inputs: dict) -> list[str]:
    msgs = []
    for k, v in inputs.items():
        lo_hi = PLAUSIBLE.get(k)
        if lo_hi and not (lo_hi[0] <= v <= lo_hi[1]):
            msgs.append(f"{k} = {v:g} nằm ngoài khoảng thường gặp ({lo_hi[0]}–{lo_hi[1]}); kiểm tra lại đơn vị.")
    return msgs


@dataclass
class Prediction:
    pm25: float
    low: float | None
    high: float | None
    aqi: float
    missing: list[str]
    spread_kind: str | None


def _member_predictions(model, X: pd.DataFrame):
    """Dự báo của từng thành viên trong ensemble (cây hoặc mô hình con)."""
    base = model
    if hasattr(model, "steps"):  # Pipeline: lấy bước cuối, biến đổi đầu vào
        try:
            Xt = X
            for _, step in model.steps[:-1]:
                Xt = step.transform(Xt)
            return _member_predictions(model.steps[-1][1], Xt if isinstance(Xt, pd.DataFrame) else pd.DataFrame(Xt))
        except Exception:
            return None, None
    ests = getattr(base, "estimators_", None)
    if ests is None:
        return None, None
    ests = ests.ravel().tolist() if isinstance(ests, np.ndarray) else list(ests)
    first = ests[0]
    try:
        if hasattr(first, "tree_"):  # cây quyết định bên trong RF/ExtraTrees
            return np.array([t.predict(X.to_numpy(dtype=float)) for t in ests]).ravel(), "tree"
        preds = []
        for e in ests:
            try:
                preds.append(float(np.ravel(e.predict(X))[0]))
            except Exception:
                preds.append(float(np.ravel(e.predict(X.to_numpy(dtype=float)))[0]))
        return np.array(preds), "model"
    except Exception:
        return None, None


def predict_pm25(raw_model, inputs: dict, extras: dict | None = None) -> Prediction:
    """Dự báo PM2.5.

    inputs: giá trị người dùng nhập; extras: giá trị bổ sung (vd. từ trạm) cho đặc trưng
    mà form không có. Đặc trưng nào hoàn toàn thiếu sẽ được điền 0 và liệt kê trong `missing`.
    """
    model, stored = unpack_model(raw_model)
    feats = model_features(model, stored)
    pool = {**(extras or {}), **inputs}
    row, missing = {}, []
    for f in feats:
        if f in pool and pool[f] is not None:
            row[f] = float(pool[f])
        else:
            row[f] = 0.0
            missing.append(f)
    X = pd.DataFrame([row], columns=feats)
    y = float(np.ravel(model.predict(X))[0])
    y = max(0.0, y)

    low = high = None
    kind = None
    members, kind = _member_predictions(model, X)
    if members is not None and len(members) > 1:
        if kind == "tree":
            low, high = (float(np.percentile(members, 10)), float(np.percentile(members, 90)))
        else:
            low, high = float(members.min()), float(members.max())
        low, high = max(0.0, min(low, y)), max(high, y)
    return Prediction(y, low, high, aqi_from_pm25(y), missing, kind)


# --------------------------------------------------------------------------
# HTML/SVG
# --------------------------------------------------------------------------
def gauge_svg(aqi: float | None, size: int = 240) -> str:
    cat = category(aqi)
    val = 0.0 if aqi is None or _isnan(aqi) else float(aqi)
    frac = min(val, 300) / 300
    r, cx, cy = 90, 110, 112
    length = math.pi * r
    arc = f"M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r} {cy}"
    label = "—" if aqi is None or _isnan(aqi) else f"{val:.0f}"
    return f"""
<svg viewBox="0 0 220 140" width="{size}" role="img" aria-label="AQI {label}">
  <path d="{arc}" fill="none" stroke="rgba(148,163,184,.16)" stroke-width="14" stroke-linecap="round"/>
  <path d="{arc}" fill="none" stroke="{cat['color']}" stroke-width="14" stroke-linecap="round"
        stroke-dasharray="{length * frac:.1f} {length:.1f}"/>
  <text x="{cx}" y="{cy - 22}" text-anchor="middle" class="g-val">{label}</text>
  <text x="{cx}" y="{cy - 2}" text-anchor="middle" class="g-lbl">AQI</text>
</svg>"""


def kpi_html(label: str, value: str, unit: str, hint: str = "", accent: str = "#38bdf8") -> str:
    return f"""
<div class="kpi" style="--accent:{accent}">
  <div class="kpi-l">{label}</div>
  <div class="kpi-v">{value}<span class="kpi-u">{unit}</span></div>
  <div class="kpi-h">{hint}</div>
</div>"""


def pollutant_bars_html(values: dict) -> str:
    rows = []
    for name, unit in [("PM2.5", "µg/m³"), ("PM10", "µg/m³"), ("NO2", "µg/m³"),
                       ("SO2", "µg/m³"), ("O3", "µg/m³"), ("CO", "µg/m³")]:
        v = values.get(name)
        if v is None:
            continue
        pct = min(100.0, v / LIMIT_100[name] * 100)
        color = category(iaqi(name, v))["color"]
        rows.append(f"""
<div class="pb">
  <span class="pb-n">{name.replace('2.5', '<sub>2.5</sub>').replace('NO2', 'NO<sub>2</sub>').replace('SO2', 'SO<sub>2</sub>').replace('O3', 'O<sub>3</sub>')}</span>
  <div class="pb-t"><div class="pb-f" style="width:{pct:.0f}%;background:{color}"></div></div>
  <span class="pb-v">{v:,.0f} <i>{unit}</i></span>
</div>""")
    return "".join(rows) or '<div class="muted">Chưa có dữ liệu chất ô nhiễm.</div>'


CSS = """
<style>
:root{--bg:#0b1220;--card:#111a2e;--line:rgba(148,163,184,.14);--txt:#e6edf7;--mut:#8696b0;--acc:#34d399}
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#10223d 0%,var(--bg) 55%);color:var(--txt)}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0}
.block-container{max-width:1080px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3,p,label,span,div{font-family:Inter,"Segoe UI",system-ui,sans-serif}
.brand{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.4rem}
.brand h1{font-size:1.45rem;font-weight:700;margin:0;letter-spacing:-.01em;color:var(--txt)}
.brand h1 b{color:var(--acc)}
.brand small{display:block;color:var(--mut);font-size:.82rem;font-weight:400;margin-top:2px}
.live{display:flex;align-items:center;gap:8px;font-size:.82rem;color:var(--mut);
 background:var(--card);border:1px solid var(--line);padding:6px 12px;border-radius:999px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--acc);box-shadow:0 0 0 0 rgba(52,211,153,.6);animation:p 2s infinite}
.dot.off{background:#f59e0b;animation:none}
@keyframes p{70%{box-shadow:0 0 0 8px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 22px}
.card h3{font-size:.78rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 14px}
.hero{display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.g-val{font-size:44px;font-weight:700;fill:var(--txt)}.g-lbl{font-size:12px;fill:var(--mut);letter-spacing:.14em}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600;font-size:.85rem;color:#0b1220}
.advice{color:var(--mut);font-size:.92rem;line-height:1.5;max-width:300px;margin-top:10px}
.kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:14px;padding:14px 16px}
.kpi-l{color:var(--mut);font-size:.78rem;letter-spacing:.04em}
.kpi-v{font-size:1.75rem;font-weight:700;margin-top:2px;color:var(--txt)}
.kpi-u{font-size:.85rem;font-weight:500;color:var(--mut);margin-left:4px}
.kpi-h{color:var(--mut);font-size:.75rem;margin-top:2px;min-height:1em}
.pb{display:grid;grid-template-columns:56px 1fr 110px;align-items:center;gap:12px;margin:10px 0;font-size:.9rem}
.pb-n{color:var(--txt);font-weight:600}.pb-v{text-align:right;color:var(--txt)}.pb-v i{color:var(--mut);font-style:normal;font-size:.75rem}
.pb-t{height:8px;border-radius:99px;background:rgba(148,163,184,.14);overflow:hidden}
.pb-f{height:100%;border-radius:99px}
.muted{color:var(--mut);font-size:.9rem}
.result{display:flex;gap:28px;flex-wrap:wrap;align-items:center}
.r-big{font-size:3rem;font-weight:700;line-height:1}.r-big span{font-size:1rem;color:var(--mut);font-weight:500;margin-left:6px}
.r-sub{color:var(--mut);font-size:.88rem;margin-top:6px}
.range{margin-top:14px;max-width:420px}
.rg-t{position:relative;height:8px;border-radius:99px;background:rgba(148,163,184,.14)}
.rg-b{position:absolute;top:0;height:100%;border-radius:99px;opacity:.9}
.rg-m{position:absolute;top:-4px;width:3px;height:16px;border-radius:2px;background:#fff}
.rg-l{display:flex;justify-content:space-between;color:var(--mut);font-size:.75rem;margin-top:6px}
.note{color:var(--mut);font-size:.78rem;margin-top:14px;line-height:1.5}
div[data-testid="stForm"]{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 22px}
.stButton>button,div[data-testid="stFormSubmitButton"]>button{border-radius:12px;font-weight:600;border:1px solid var(--line)}
div[data-testid="stFormSubmitButton"]>button{background:var(--acc);color:#06281c;border:none;width:100%;padding:.6rem}
.sec{font-size:1.05rem;font-weight:600;margin:1.8rem 0 .8rem;color:var(--txt)}
@media(max-width:700px){.pb{grid-template-columns:50px 1fr 90px}}
</style>
"""


def result_range_html(pred: Prediction) -> str:
    if pred.low is None or pred.high is None:
        return ""
    top = max(pred.high * 1.25, pred.pm25 * 1.5, 50)
    l, h, m = pred.low / top * 100, pred.high / top * 100, pred.pm25 / top * 100
    color = category(pred.aqi)["color"]
    return f"""
<div class="range">
  <div class="rg-t"><div class="rg-b" style="left:{l:.1f}%;width:{max(h - l, 1):.1f}%;background:{color}"></div>
  <div class="rg-m" style="left:calc({m:.1f}% - 1px)"></div></div>
  <div class="rg-l"><span>{pred.low:.1f}</span><span>Khoảng dao động dự báo</span><span>{pred.high:.1f} µg/m³</span></div>
</div>"""


# ==========================================================================
# GIAO DIỆN STREAMLIT
# ==========================================================================
import os
from datetime import datetime

import joblib
import requests
import streamlit as st


API_LATEST = os.getenv(
    "AQ_API_LATEST", "https://beijing-observation-simulator.onrender.com/api/v1/live/latest"
)
MODEL_PATH = os.getenv("AQ_MODEL_PATH", "models/air_quality_rf_ensemble.pkl")
REFRESH_SECONDS = 30

st.set_page_config(page_title="Beijing Air Quality Forecast", page_icon="🍃", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- dữ liệu
@st.cache_data(ttl=20, show_spinner=False)
def fetch_live():
    """Gọi API quan trắc. Dịch vụ free trên Render có thể 'ngủ' nên cho timeout dài."""
    try:
        res = requests.get(API_LATEST, timeout=25)
        res.raise_for_status()
        live = parse_live(res.json())
        return live
    except Exception as exc:  # noqa: BLE001
        live = Live()
        live.message = f"Không kết nối được trạm quan trắc ({type(exc).__name__})."
        return live


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------- tiêu đề
def header(live: Live):
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
    cat = category(aqi)

    if not live.ok:
        st.warning(live.message or "Chưa nhận được dữ liệu từ trạm. Dịch vụ có thể đang khởi động, thử lại sau ít phút.")
        return live

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        st.markdown(
            f"""<div class="card"><h3>Chất lượng không khí hiện tại{f' · {live.station}' if live.station else ''}</h3>
<div class="hero">{gauge_svg(aqi)}
<div><span class="pill" style="background:{cat['color']}">{cat['label']}</span>
<div class="advice">{cat['advice']}</div></div></div></div>""",
            unsafe_allow_html=True,
        )
    with right:
        tiles = [
            kpi_html("PM2.5", f"{v['PM2.5']:.0f}", "µg/m³", "Bụi mịn", "#38bdf8"),
            kpi_html("Nhiệt độ", f"{v['TEMP']:.1f}" if "TEMP" in v else "—", "°C", "", "#fb923c"),
            kpi_html("Độ ẩm", f"{v['HUMIDITY']:.0f}" if "HUMIDITY" in v else "—", "%", "Suy ra từ nhiệt độ &amp; điểm sương" if "DEWP" in v else "", "#60a5fa"),
            kpi_html("Gió", f"{v['WSPM']:.1f}" if "WSPM" in v else "—", "m/s", "", "#a78bfa"),
        ]
        st.markdown(f'<div class="kpis">{"".join(tiles)}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="card" style="margin-top:12px"><h3>Nồng độ chất ô nhiễm</h3>{pollutant_bars_html(v)}</div>',
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


def predict_section(live: Live):
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
    for msg in validate_inputs(inputs):
        st.warning(msg)

    try:
        pred = predict_pm25(model, inputs, extras=live.values)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không thể dự báo: {exc}")
        return

    cat = category(pred.aqi)
    st.markdown(
        f"""<div class="card"><h3>Kết quả dự báo</h3>
<div class="result"><div><div class="r-big">{pred.pm25:.1f}<span>µg/m³ PM2.5</span></div>
<div class="r-sub">AQI tương ứng ≈ {pred.aqi:.0f}</div></div>
<div><span class="pill" style="background:{cat['color']}">{cat['label']}</span>
<div class="advice">{cat['advice']}</div></div></div>
{result_range_html(pred)}
<div class="note">{_note(pred, live)}</div></div>""",
        unsafe_allow_html=True,
    )


def _note(pred: Prediction, live: Live) -> str:
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
    st.fragment(run_every=REFRESH_SECONDS)(live_section)()
else:
    live_section()
predict_section(live0)