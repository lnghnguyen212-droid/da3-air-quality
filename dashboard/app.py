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


def kpi_html(label: str, value: str, unit: str, hint: str = "", accent: str = "#13a8c4") -> str:
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
[data-testid="stSidebar"], [data-testid="collapsedControl"]{display:none!important}
.location-caption{color:#62efd5;font-size:.73rem;letter-spacing:.18em;font-weight:700;margin:6px 0 9px}
.location-note{border:1px solid rgba(98,239,213,.22);border-radius:13px;padding:13px 17px;background:rgba(16,42,53,.8);color:#a9cdd0;font-size:.85rem;min-height:48px}
[data-testid="stSelectbox"]>div>div{background:#112b36!important;color:#eaffff!important;border:1px solid #3b777a!important;border-radius:12px!important}
[data-testid="stSelectbox"] svg{color:#62efd5!important}

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--bg:#050d15;--card:#101d29;--line:rgba(150,220,235,.16);--txt:#ecfcff;--mut:#91a9b8;--acc:#62efd5}
html{scroll-behavior:smooth}
.stApp{background:radial-gradient(ellipse 65% 28% at 82% 4%,#144c5d 0%,transparent 85%),radial-gradient(ellipse 70% 45% at -15% 65%,#0a2935 0%,transparent 80%),#050d15;color:var(--txt)}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background-image:linear-gradient(#b9faff06 1px,transparent 1px),linear-gradient(90deg,#b9faff06 1px,transparent 1px);background-size:68px 68px;mask-image:linear-gradient(to bottom,#000,transparent 75%)}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0}
.block-container{max-width:1400px;padding:1.2rem 2.5rem 5rem;position:relative;z-index:1}
html,body,p,label,span,div,button{font-family:'DM Sans',sans-serif}
h1,h2,h3,.sec,.r-big,.kpi-v{font-family:'Space Grotesk',sans-serif}
.brand{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:1.5rem;padding:16px 0;border-bottom:1px solid var(--line)}
.brand h1{font-size:1.45rem;font-weight:700;margin:0;letter-spacing:-.05em;color:var(--txt)}.brand h1 b{color:var(--acc)}
.brand small{display:block;color:var(--mut);font-size:.76rem;letter-spacing:.02em;margin-top:4px}
.live{display:flex;align-items:center;gap:9px;font-size:.77rem;color:#b3fff1;background:#122e34;border:1px solid #36756b;padding:9px 15px;border-radius:99px;white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;background:#62efd5;box-shadow:0 0 14px #62efd5;animation:pulse 2s infinite}.dot.off{background:#f6b55e;animation:none}
@keyframes pulse{50%{opacity:.4;box-shadow:0 0 24px #62efd5}}
.card,.kpi,div[data-testid="stForm"]{background:linear-gradient(145deg,rgba(25,48,61,.92),rgba(9,22,33,.95));border:1px solid var(--line);border-radius:20px;box-shadow:0 20px 60px #0003;backdrop-filter:blur(16px);transition:transform .3s,border-color .3s,box-shadow .3s}
.card:hover,.kpi:hover{transform:translateY(-4px);border-color:#62efd56b;box-shadow:0 22px 70px #1edac31b}
.card{padding:26px}.card h3{font-size:.76rem;letter-spacing:.16em;text-transform:uppercase;color:#b2d4d9;margin:0 0 20px}
.hero{display:flex;align-items:center;gap:20px;flex-wrap:wrap}.g-val{font-size:44px;font-weight:700;fill:var(--txt)}.g-lbl{font-size:12px;fill:var(--mut)}
.pill{display:inline-block;padding:7px 14px;border-radius:99px;font-weight:700;font-size:.82rem;color:#07121a}.advice{color:#b0c5cf;font-size:.91rem;line-height:1.7;max-width:340px;margin-top:12px}
.kpis{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.kpi{padding:20px;border-top:2px solid var(--accent)}.kpi-l{color:var(--mut);font-size:.8rem}.kpi-v{font-size:2rem;font-weight:700;color:var(--txt);margin-top:9px}.kpi-u{font-size:.8rem;color:var(--mut);margin-left:5px}.kpi-h{font-size:.72rem;color:var(--mut);min-height:1em}
.pb{display:grid;grid-template-columns:65px 1fr 132px;align-items:center;gap:14px;margin:15px 0;font-size:.89rem}.pb-n,.pb-v{color:var(--txt);font-weight:600}.pb-v{text-align:right}.pb-v i{color:var(--mut);font-style:normal;font-size:.7rem}.pb-t{height:8px;border-radius:99px;background:#ffffff17;overflow:hidden}.pb-f{height:100%;border-radius:99px;box-shadow:0 0 15px currentColor}
.muted,.r-sub,.note{color:var(--mut)}.result{display:flex;gap:30px;flex-wrap:wrap;align-items:center}.r-big{font-size:3.3rem;font-weight:700;color:var(--txt)}.r-big span{font-size:1rem;color:var(--mut);font-weight:400;margin-left:5px}.r-sub{margin-top:9px}.range{margin-top:18px;max-width:520px}.rg-t{position:relative;height:9px;border-radius:99px;background:#ffffff20}.rg-b{position:absolute;top:0;height:100%;border-radius:99px}.rg-m{position:absolute;top:-4px;width:3px;height:17px;background:white}.rg-l{display:flex;justify-content:space-between;color:var(--mut);font-size:.75rem;margin-top:8px}.note{font-size:.8rem;margin-top:14px;line-height:1.7}
.sec{font-size:1.45rem;font-weight:700;margin:2.8rem 0 1.1rem;color:var(--txt);letter-spacing:-.04em}
div[data-testid="stForm"]{padding:24px}.stButton>button,div[data-testid="stFormSubmitButton"]>button{border-radius:12px;font-weight:700;border:1px solid #40717a;background:#132b37;color:#e9ffff;min-height:46px;transition:all .25s}.stButton>button:hover{border-color:#62efd5;color:#62efd5;transform:translateY(-2px)}
div[data-testid="stFormSubmitButton"]>button[kind="primary"],.stButton>button[kind="primary"]{background:#62efd5;color:#06181c;border:none}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:#0b1c29;color:#fff}
.lab-intro{background:linear-gradient(115deg,#102b35,#102028);border:1px solid #3d7779;border-radius:18px;padding:24px;margin:6px 0 16px}.lab-intro h3{margin:0 0 8px;color:#82ffe3}.lab-intro p{margin:0;color:#aecbd0;font-size:.9rem}
.immersive{min-height:480px;display:grid;grid-template-columns:1.15fr 1fr;align-items:center;position:relative;overflow:hidden;border-bottom:1px solid var(--line);margin:0 0 36px;padding:40px 22px 52px}
.immersive-copy{position:relative;z-index:2}.eyebrow{font-size:.75rem;font-weight:700;letter-spacing:.28em;color:#62efd5;margin-bottom:22px}.immersive h2{font-family:'Space Grotesk',sans-serif;font-size:clamp(3rem,5.4vw,6.5rem);line-height:.99;letter-spacing:-.085em;font-weight:700;color:#f0feff;margin:0 0 24px}.immersive h2 em{font-style:normal;color:#62efd5}.immersive p{font-size:1rem;color:#b1c7d0;max-width:510px;line-height:1.8}.hero-tags{display:flex;flex-wrap:wrap;gap:9px;margin-top:26px}.hero-tags span{border:1px solid #ffffff3a;border-radius:99px;padding:9px 13px;font-size:.73rem;letter-spacing:.05em;color:#c7e6e8}
.planet-wrap{position:relative;height:410px;display:grid;place-items:center;isolation:isolate}.planet{width:min(31vw,355px);height:min(31vw,355px);min-width:245px;min-height:245px;border-radius:50%;position:relative;background:radial-gradient(circle at 29% 24%,#c6fff9 0%,#64dfd2 9%,#197d9c 29%,#123e61 51%,#051827 73%);box-shadow:inset -33px -24px 55px #000a,inset 18px 10px 28px #c7fffa70,0 0 75px #22cfbf36,0 0 160px #1a8fa42a;animation:float 7s ease-in-out infinite}
.planet:before{content:"";position:absolute;inset:-14%;border:1px solid #7de8e348;border-radius:50%;transform:rotate(-26deg) scaleY(.48);box-shadow:0 0 28px #62efd52b;animation:orbit 18s linear infinite}.planet:after{content:"";position:absolute;inset:12%;border-radius:50%;background:repeating-radial-gradient(ellipse at 20% 40%,transparent 0 16px,#c6ffff16 17px 18px,transparent 19px 30px);transform:rotate(26deg)}
.orbit-two{position:absolute;width:96%;height:58%;border:1px dashed #80f7e13c;border-radius:50%;transform:rotate(34deg);animation:orbit 25s linear infinite}.orbit-dot{position:absolute;right:12%;top:13%;height:9px;width:9px;background:#abfff4;border-radius:50%;box-shadow:0 0 23px #6effe6}.planet-label{position:absolute;bottom:12px;right:4px;color:#8fcacb;font-size:.67rem;letter-spacing:.22em}
@keyframes float{50%{transform:translateY(-16px) rotate(3deg)}}@keyframes orbit{50%{transform:rotate(154deg) scaleY(.48)}100%{transform:rotate(334deg) scaleY(.48)}}
.ticker{overflow:hidden;white-space:nowrap;color:#6debd8;border-top:1px solid #ffffff1b;border-bottom:1px solid #ffffff1b;padding:14px 0;font-size:.76rem;letter-spacing:.21em;margin:0 0 26px}.ticker span{display:inline-block;animation:marquee 24s linear infinite}@keyframes marquee{to{transform:translateX(-50%)}}
@media(max-width:850px){.immersive{grid-template-columns:1fr;min-height:auto;padding:32px 0}.planet-wrap{height:330px}.planet{width:260px;height:260px}.block-container{padding:1rem 1rem 3rem}.brand{flex-wrap:wrap}.pb{grid-template-columns:50px 1fr 105px}.immersive h2{font-size:3.6rem}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
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
    "AQ_API_LATEST", ""
)
MODEL_PATH = os.getenv("AQ_MODEL_PATH", "models/air_quality_rf_ensemble.pkl")
REFRESH_SECONDS = 30

st.set_page_config(page_title="AirVision AI | Việt Nam", page_icon="🍃", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# Bộ chọn địa phương: giữ nguyên CSS/hero và thay dữ liệu theo vị trí.
VN_CITIES = {
    "Đà Nẵng": (16.0544, 108.2022),
    "Hà Nội": (21.0285, 105.8542),
    "TP. Hồ Chí Minh": (10.8231, 106.6297),
    "Huế": (16.4637, 107.5909),
    "Hải Phòng": (20.8449, 106.6881),
    "Cần Thơ": (10.0452, 105.7469),
    "Quảng Trị": (16.8163, 107.1003),
    "Quảng Ngãi": (15.1214, 108.8044),
    "Gia Lai (Pleiku)": (13.9833, 108.0000),
    "Khánh Hòa (Nha Trang)": (12.2388, 109.1967),
    "Lâm Đồng (Đà Lạt)": (11.9404, 108.4583),
    "Đắk Lắk (Buôn Ma Thuột)": (12.6667, 108.0500),
    "Nghệ An (Vinh)": (18.6796, 105.6813),
    "Thanh Hóa": (19.8067, 105.7852),
    "Thái Nguyên": (21.5942, 105.8482),
    "Quảng Ninh (Hạ Long)": (20.9511, 107.0800),
    "An Giang (Long Xuyên)": (10.3864, 105.4352),
    "Cà Mau": (9.1768, 105.1524),
    "Tây Ninh": (11.3100, 106.0983),
    "Đồng Nai (Biên Hòa)": (10.9447, 106.8243),
}
# Bộ chọn được đặt trong nội dung chính, không dùng sidebar.
st.markdown("""<div class="location-caption">◉ AIRVISION / LOCATION INTELLIGENCE</div>""", unsafe_allow_html=True)
loc_col, info_col = st.columns([1.1, 2.2], gap="medium")
with loc_col:
    SELECTED_CITY = st.selectbox("Khu vực dự báo", list(VN_CITIES), index=0, label_visibility="collapsed")
with info_col:
    st.markdown('<div class="location-note">✦ Chọn khu vực để cập nhật AQI, thời tiết, bản đồ và dự báo · Nguồn mô hình Open-Meteo</div>', unsafe_allow_html=True)
CITY_LAT, CITY_LON = VN_CITIES[SELECTED_CITY]



# ---------------------------------------------------------------- dữ liệu
@st.cache_resource(ttl=20, show_spinner=False)
def fetch_live(city_name):
    """Đọc chất lượng không khí hiện tại từ Open-Meteo tại Đà Nẵng; không giả là trạm quan trắc."""
    if not API_LATEST:
        return _danang_live(city_name)
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
    text = f"Trực tuyến · {when}" if live.ok else "Chưa có dữ liệu mô hình"
    st.markdown(
        f"""<div class="brand">
  <div><h1>✦ AirVision <b>AI</b></h1>
  <small>VIETNAM AIR INTELLIGENCE · Real-time models &amp; Ensemble Learning</small></div>
  <div class="live"><span class="dot{state}"></span>{text}</div></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- khối realtime
def european_category(aqi):
    if aqi is None or pd.isna(aqi):
        return {"label": "Chưa có dữ liệu", "color": "#64748b", "advice": "Chưa có European AQI."}
    if aqi <= 20:
        return {"label": "Tốt", "color": "#28d79b", "advice": "Chất lượng không khí tốt theo European AQI."}
    if aqi <= 40:
        return {"label": "Khá", "color": "#a4d65e", "advice": "Chất lượng không khí tương đối tốt."}
    if aqi <= 60:
        return {"label": "Trung bình", "color": "#f6c85f", "advice": "Nhóm nhạy cảm nên chú ý khi hoạt động ngoài trời."}
    if aqi <= 80:
        return {"label": "Kém", "color": "#f29a5b", "advice": "Nên cân nhắc giảm hoạt động ngoài trời kéo dài."}
    if aqi <= 100:
        return {"label": "Rất kém", "color": "#e76672", "advice": "Nên hạn chế hoạt động ngoài trời."}
    return {"label": "Cực kỳ kém", "color": "#b26cce", "advice": "Nên tránh hoạt động ngoài trời khi có thể."}

def live_section():
    live = fetch_live(SELECTED_CITY)
    v = live.values
    aqi = v.get("AQI")
    cat = european_category(aqi)

    if not live.ok:
        st.warning(live.message or "Chưa nhận được dữ liệu từ trạm. Dịch vụ có thể đang khởi động, thử lại sau ít phút.")
        return live

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        st.markdown(
            f"""<div class="card"><h3>Chất lượng không khí hiện tại · European AQI{f' · {live.station}' if live.station else ''}</h3>
<div class="hero">{gauge_svg(aqi)}
<div><span class="pill" style="background:{cat['color']}">{cat['label']}</span>
<div class="advice">{cat['advice']}</div></div></div></div>""",
            unsafe_allow_html=True,
        )
    with right:
        tiles = [
            kpi_html("PM2.5", f"{v['PM2.5']:.0f}", "µg/m³", "Bụi mịn", "#13a8c4"),
            kpi_html("Nhiệt độ", f"{v['TEMP']:.1f}" if "TEMP" in v else "—", "°C", "", "#f59e63"),
            kpi_html("Độ ẩm", f"{v['HUMIDITY']:.0f}" if "HUMIDITY" in v else "—", "%", "Suy ra từ nhiệt độ &amp; điểm sương" if "DEWP" in v else "", "#4c9fea"),
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
    live = fetch_live(SELECTED_CITY)
    for key, _, default, _, _ in FIELDS:
        val = live.get(key)
        if val is not None:
            st.session_state[f"in_{key}"] = float(val)


def predict_section(live: Live):
    st.markdown('<div class="sec">Dự báo nồng độ PM2.5</div>', unsafe_allow_html=True)

    for key, _, default, _, _ in FIELDS:  # lần đầu: ưu tiên số liệu trạm
        st.session_state.setdefault(f"in_{key}", float(live.get(key, default)))

    with st.form("predict_form", border=False):
        cols = st.columns(2)
        for i, (key, label, _, step, fmt) in enumerate(FIELDS):
            with cols[i % 2]:
                st.number_input(label, step=step, format=fmt, key=f"in_{key}")
        c1, c2 = st.columns([1, 2.2])
        with c1:
            st.form_submit_button("Dùng số liệu trạm", on_click=fill_from_station, disabled=not live.ok,
                                  use_container_width=True)
        with c2:
            submitted = st.form_submit_button("Dự báo PM2.5", type="primary", use_container_width=True)

    if not submitted:
        if "latest_prediction" in st.session_state:
            show_prediction(st.session_state["latest_prediction"], live)
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

    st.session_state["latest_prediction"] = pred
    show_prediction(pred, live)


def show_prediction(pred: Prediction, live: Live):
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


# ---------------------------------------------------------------- AI Scenario Lab

def scenario_lab(live: Live):
    st.markdown('<div class="sec">🧪 AI Scenario Lab · Dấu ấn riêng</div>', unsafe_allow_html=True)
    st.markdown("""<div class="lab-intro"><h3>Nếu điều kiện môi trường thay đổi thì sao?</h3>
    <p>Điều chỉnh các yếu tố và chạy mô hình thực tế để so sánh hai kịch bản. Đây là phân tích giả định (what-if), không phải dự báo theo giờ hay chứng minh quan hệ nhân quả.</p></div>""", unsafe_allow_html=True)
    model = load_model()
    if model is None:
        st.info("Cần có file mô hình đã huấn luyện để sử dụng phòng thí nghiệm kịch bản.")
        return
    base = {k: float(st.session_state.get(f"in_{k}", live.get(k, default))) for k, _, default, _, _ in FIELDS}
    with st.expander("Tùy chỉnh kịch bản môi trường", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            wind = st.slider("Thay đổi tốc độ gió (%)", -80, 100, 20, 5, key="lab_wind")
            temp = st.slider("Thay đổi nhiệt độ (°C)", -10.0, 10.0, 0.0, 0.5, key="lab_temp")
        with c2:
            traffic = st.slider("Thay đổi NO₂ (%)", -80, 100, -20, 5, key="lab_no2")
            dust = st.slider("Thay đổi PM10 (%)", -80, 100, -15, 5, key="lab_pm10")
        run = st.button("✦ Phân tích & so sánh kịch bản", type="primary", use_container_width=True)
    if not run:
        return
    alternative = base.copy()
    alternative["WSPM"] = max(0, base["WSPM"] * (1 + wind / 100))
    alternative["TEMP"] = base["TEMP"] + temp
    alternative["NO2"] = max(0, base["NO2"] * (1 + traffic / 100))
    alternative["PM10"] = max(0, base["PM10"] * (1 + dust / 100))
    try:
        before = predict_pm25(model, base, extras=live.values)
        after = predict_pm25(model, alternative, extras=live.values)
    except Exception as exc:
        st.error(f"Không thể chạy kịch bản: {exc}")
        return
    a, b, c = st.columns(3)
    a.metric("Kịch bản gốc · PM2.5", f"{before.pm25:.1f} µg/m³")
    b.metric("Kịch bản điều chỉnh", f"{after.pm25:.1f} µg/m³", f"{after.pm25 - before.pm25:+.1f} µg/m³", delta_color="inverse")
    c.metric("AQI ước tính sau thay đổi", f"{after.aqi:.0f}", category(after.aqi)["label"])
    comparison = pd.DataFrame({"Kịch bản": ["Hiện tại (đầu vào)", "Giả lập"], "PM2.5 (µg/m³)": [before.pm25, after.pm25]})
    st.bar_chart(comparison.set_index("Kịch bản"), color="#13a8a3")
    st.caption("Kết quả là đầu ra của mô hình với bộ đặc trưng đã thay đổi, không phải phép đo thực tế. Chỉ số AQI ở đây được suy ra riêng từ PM2.5; thay đổi nhiều biến đồng thời không xác lập tác động nhân quả.")
    if before.missing or after.missing:
        st.warning("Một số đặc trưng mô hình bị thiếu và được điền 0: " + ", ".join(sorted(set(before.missing + after.missing))))


# ---------------------------------------------------------------- tác vụ theo tài liệu tham khảo
from datetime import timedelta

# Tùy chọn: API dự báo phải trả về dữ liệu có timestamp và các giá trị dự báo.
FORECAST_API = os.getenv("AQ_FORECAST_API", "").strip()
STATIONS_API = os.getenv("AQ_STATIONS_API", "").strip()



# Dữ liệu mô hình dự báo chất lượng không khí Open-Meteo cho tọa độ trung tâm Đà Nẵng.
# AQI sử dụng thang European AQI của nhà cung cấp, không phải HJ 633-2012.
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

@st.cache_data(ttl=1800, show_spinner=False)
def _danang_air_data(city_name):
    params = {
        "latitude": VN_CITIES[city_name][0], "longitude": VN_CITIES[city_name][1],
        "current": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide,european_aqi",
        "hourly": "pm2_5,pm10,european_aqi",
        "timezone": "Asia/Ho_Chi_Minh", "forecast_days": 7,
    }
    response = requests.get(AIR_URL, params=params, timeout=25)
    response.raise_for_status()
    return response.json()

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

@st.cache_data(ttl=900, show_spinner=False)
def _weather_now(city_name):
    lat, lon = VN_CITIES[city_name]
    response = requests.get(WEATHER_URL, params={
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "wind_speed_unit": "ms", "timezone": "Asia/Ho_Chi_Minh",
    }, timeout=25)
    response.raise_for_status()
    return response.json().get("current", {})

def _danang_live(city_name):
    try:
        payload = _danang_air_data(city_name)
        current = payload.get("current", {})
        names = {"pm2_5":"PM2.5", "pm10":"PM10", "nitrogen_dioxide":"NO2", "sulphur_dioxide":"SO2", "ozone":"O3", "carbon_monoxide":"CO"}
        vals = {out:float(current[key]) for key,out in names.items() if current.get(key) is not None}
        # AQI này là European AQI (không dùng ngưỡng Trung Quốc của RF).
        if current.get("european_aqi") is not None:
            vals["AQI"] = float(current["european_aqi"])
        try:
            weather = _weather_now(city_name)
            weather_names = {"temperature_2m": "TEMP", "relative_humidity_2m": "HUMIDITY", "wind_speed_10m": "WSPM"}
            vals.update({out: float(weather[key]) for key, out in weather_names.items() if weather.get(key) is not None})
        except requests.RequestException:
            pass  # Thời tiết có thể tạm lỗi; không điền giá trị giả.
        live = Live(values=vals, station=f"{city_name} · điểm tham chiếu mô hình", time=current.get("time"), ok="PM2.5" in vals)
        if not live.ok: live.message="Nguồn Open-Meteo chưa có PM2.5 tại khu vực được chọn."
        return live
    except Exception as exc:
        return Live(message=f"Không lấy được dữ liệu {city_name} từ Open-Meteo ({type(exc).__name__}).")

def _danang_forecast():
    try:
        hourly = _danang_air_data(SELECTED_CITY).get("hourly", {})
        frame = pd.DataFrame({"timestamp":hourly.get("time", []),"AQI":hourly.get("european_aqi", []),"PM2.5":hourly.get("pm2_5", []),"PM10":hourly.get("pm10", [])})
        return _canonical_table(frame), "Open-Meteo · dự báo mô hình · European AQI"
    except Exception as exc:
        return pd.DataFrame(), f"Open-Meteo chưa phản hồi ({type(exc).__name__})"

def _danang_location():
    # Đây là vị trí tham chiếu, không được trình bày là trạm đo.
    return pd.DataFrame([{"station":f"{SELECTED_CITY} · điểm tham chiếu", "latitude":CITY_LAT,"longitude":CITY_LON}])

def _fetch_optional_json(url):
    if not url:
        return None
    try:
        response = requests.get(url, timeout=18)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.warning(f"Không thể đọc nguồn dữ liệu bổ sung: {type(exc).__name__}")
        return None


def _records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "records", "forecasts", "stations", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _canonical_table(raw, station=False):
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = raw.copy()
    aliases = {
        "timestamp": ["timestamp", "datetime", "time", "date", "forecast_time", "observed_at"],
        "station": ["station", "station_name", "name", "site"],
        "latitude": ["latitude", "lat"], "longitude": ["longitude", "lon", "lng", "long"],
        "AQI": ["aqi"], "PM2.5": ["pm2.5", "pm25", "pm2_5"],
        "PM10": ["pm10"], "WSPM": ["wspm", "wind_speed"],
    }
    columns = {_norm(c): c for c in df.columns}
    rename = {}
    for target, options in aliases.items():
        for option in options:
            if _norm(option) in columns:
                rename[columns[_norm(option)]] = target
                break
    df = df.rename(columns=rename)
    for key in ("AQI", "PM2.5", "PM10", "latitude", "longitude", "WSPM"):
        if key in df:
            df[key] = pd.to_numeric(df[key], errors="coerce")
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
    if "AQI" not in df and "PM2.5" in df:
        df["AQI"] = df["PM2.5"].apply(lambda x: aqi_from_pm25(x) if pd.notna(x) else float("nan"))
    if station and "station" not in df:
        return pd.DataFrame()
    return df


def _load_table(kind):
    """Tự đọc endpoint được cấu hình; không yêu cầu người dùng upload."""
    if kind == "forecast" and not FORECAST_API:
        return _danang_forecast()
    if kind == "stations" and not STATIONS_API:
        return _danang_location(), f"Điểm tham chiếu {SELECTED_CITY} (không phải trạm)"
    api = FORECAST_API if kind == "forecast" else STATIONS_API
    if not api:
        return pd.DataFrame(), "Chưa cấu hình API"
    payload = _fetch_optional_json(api)
    if payload is None:
        return pd.DataFrame(), "API chưa phản hồi"
    records = _records(payload)
    if not records:
        return pd.DataFrame(), "API không có danh sách bản ghi"
    table = _canonical_table(pd.DataFrame(records), station=kind == "stations")
    return table, "API tự động" if not table.empty else "API thiếu trường dữ liệu"


def _single_live_station(live):
    """Chỉ hiển thị trạm hiện tại nếu tọa độ được cấu hình thật."""
    try:
        lat = float(os.environ["AQ_STATION_LAT"])
        lon = float(os.environ["AQ_STATION_LON"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return pd.DataFrame()
    except (KeyError, ValueError):
        return pd.DataFrame()
    if not live.ok:
        return pd.DataFrame()
    record = {"station": live.station or "Trạm quan trắc hiện tại", "latitude": lat, "longitude": lon}
    record.update({k: live.values[k] for k in ("AQI", "PM2.5", "PM10") if k in live.values})
    return pd.DataFrame([record])


def _live_model_now(live):
    """Dự đoán PM2.5 cho đúng thời điểm hiện tại; KHÔNG gắn nhãn dự báo 24h."""
    if not live.ok:
        return None, "API quan trắc chưa có dữ liệu"
    try:
        model = load_model()
        if model is None:
            return None, f"Không tìm thấy mô hình {MODEL_PATH}"
        model_obj, stored = unpack_model(model)
        feats = model_features(model_obj, stored)
        missing = [f for f in feats if live.get(f) is None]
        if missing:
            return None, "Trạm thiếu đầu vào mô hình: " + ", ".join(missing)
        return predict_pm25(model, {f: float(live.get(f)) for f in feats}), None
    except Exception as exc:
        return None, f"Mô hình chưa thể chạy: {type(exc).__name__}: {exc}"


def _forecast_valid(df):
    return not df.empty and "timestamp" in df and any(c in df for c in ("AQI", "PM2.5", "PM10"))


def _hourly(df, hours=24):
    if not _forecast_valid(df):
        return pd.DataFrame()
    now = pd.Timestamp.now()
    subset = df[(df["timestamp"] >= now.floor("h")) & (df["timestamp"] < now + pd.Timedelta(hours=hours))].copy()
    if subset.empty:
        return subset
    numeric = [c for c in ("AQI", "PM2.5", "PM10") if c in subset]
    subset = subset.set_index("timestamp")[numeric].resample("1h").mean().dropna(how="all").reset_index()
    return subset


def _chart(df, key, title):
    if key not in df or df[key].dropna().empty:
        st.info(f"Nguồn dữ liệu chưa có {key}.")
        return
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)
    st.line_chart(df.set_index("timestamp")[[key]], color="#62efd5", height=300)


def _golden_window(df, activity, sensitive):
    """Chọn khung liên tiếp 2 giờ AQI thấp nhất; ngưỡng mang tính gợi ý."""
    if df.empty or "AQI" not in df:
        return None, pd.DataFrame()
    data = df[["timestamp", "AQI"]].dropna().sort_values("timestamp").copy()
    thresholds = {"Đi dạo": 100, "Chạy bộ": 50, "Đạp xe": 75, "Yoga ngoài trời": 100}
    threshold = min(thresholds[activity], 50) if sensitive else thresholds[activity]
    data["Mức độ"] = data["AQI"].apply(lambda a: "Phù hợp" if a <= threshold else ("Cân nhắc" if a <= threshold + 50 else "Nên tránh"))
    candidates = []
    for i in range(len(data)-1):
        a, b = data.iloc[i], data.iloc[i+1]
        if b["timestamp"] - a["timestamp"] == pd.Timedelta(hours=1):
            candidates.append((float((a["AQI"]+b["AQI"])/2), a["timestamp"], b["timestamp"]+pd.Timedelta(hours=1)))
    return min(candidates, key=lambda t:t[0]) if candidates else None, data


def overview_task(live, forecast):
    st.markdown('<div class="sec">01 / Tổng quan chất lượng không khí</div>', unsafe_allow_html=True)
    live_section()
    if _forecast_valid(forecast):
        hourly = _hourly(forecast)
        if not hourly.empty:
            key = "AQI" if "AQI" in hourly else "PM2.5"
            _chart(hourly, key, "24 giờ tới · dữ liệu dự báo")
            window, _ = _golden_window(hourly, "Đi dạo", False)
            if window:
                st.success(f"✦ Giờ không khí tốt nhất (2 giờ liên tiếp): {window[1]:%H:%M} – {window[2]:%H:%M}, AQI trung bình {window[0]:.0f}.")
    else:
        st.info("Biểu đồ 24 giờ và giờ vàng sẽ xuất hiện khi hệ thống kết nối được API dự báo theo giờ.")


def map_task(stations):
    st.markdown('<div class="sec">02 / Bản đồ chất lượng không khí</div>', unsafe_allow_html=True)
    st.caption(f"Vị trí tham chiếu mô hình tại {SELECTED_CITY}. Không phải trạm đo thực tế; chỉ hiển thị số đo trạm khi có API trạm riêng.")
    required = {"station", "latitude", "longitude"}
    if stations.empty or not required.issubset(stations.columns):
        st.info("Chưa có danh sách trạm kèm tọa độ. Cần API nhiều trạm hoặc cấu hình AQ_STATION_LAT / AQ_STATION_LON cho trạm hiện tại.")
        return
    options = [c for c in ("AQI", "PM2.5") if c in stations]
    if not options:
        st.map(stations.rename(columns={"latitude":"lat", "longitude":"lon"})[["lat","lon"]], zoom=10, use_container_width=True)
        st.info(f"Đang hiển thị vị trí tham chiếu {SELECTED_CITY}. Chưa có dữ liệu đo từ các trạm thực tế.")
        return
    metric = st.radio("Chỉ số bản đồ", options, horizontal=True, key="map_metric")
    stations = stations.dropna(subset=["latitude", "longitude", metric]).sort_values(metric, ascending=False)
    if stations.empty:
        st.warning("Không có trạm hợp lệ để hiển thị.")
        return
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        selected = st.selectbox("Trạm quan trắc", stations["station"].astype(str).tolist(), key="selected_station")
        st.dataframe(stations[["station", metric]].rename(columns={"station":"Trạm"}), hide_index=True, use_container_width=True)
        station_row = stations[stations["station"].astype(str) == selected].iloc[0]
        st.metric(f"{metric} · {selected}", f"{station_row[metric]:.1f}")
    with right:
        st.map(stations.rename(columns={"latitude":"lat", "longitude":"lon"})[["lat", "lon"]], zoom=10, use_container_width=True)
    st.caption("Điểm trên bản đồ lấy từ tọa độ trong nguồn dữ liệu; bản đồ không tự tạo số đo cho các khu vực chưa có trạm.")


def forecast_task(df):
    st.markdown('<div class="sec">03 / Dự báo chất lượng không khí</div>', unsafe_allow_html=True)
    st.caption(f"Dự báo theo giờ tại {SELECTED_CITY} từ mô hình Open-Meteo (European AQI), không phải đầu ra của Random Forest Bắc Kinh.")
    if _forecast_valid(df):
        hourly = _hourly(df)
        if not hourly.empty:
            options = [c for c in ("AQI", "PM2.5", "PM10") if c in hourly]
            metric = st.radio("Chỉ số", options, horizontal=True, key="forecast_metric")
            _chart(hourly, metric, "24 giờ tới")
            valid = hourly.dropna(subset=[metric])
            if not valid.empty:
                high = valid.loc[valid[metric].idxmax()]
                low = valid.loc[valid[metric].idxmin()]
                c1,c2,c3=st.columns(3)
                c1.metric("Cao nhất", f"{high[metric]:.1f}", f"{high['timestamp']:%H:%M}")
                c2.metric("Thấp nhất", f"{low[metric]:.1f}", f"{low['timestamp']:%H:%M}")
                c3.metric("Biến động đầu–cuối", f"{valid[metric].iloc[-1]-valid[metric].iloc[0]:+.1f}")
        else:
            st.info("Chưa có bản ghi dự báo nào trong 24 giờ tới.")
        if "AQI" in df:
            future = df[df["timestamp"] >= pd.Timestamp.now().floor("D")].copy()
            future = future[future["timestamp"] < pd.Timestamp.now().floor("D")+pd.Timedelta(days=7)]
            days = future.set_index("timestamp")["AQI"].resample("1D").mean().dropna() if not future.empty else pd.Series(dtype=float)
            if not days.empty:
                st.markdown('<div class="sec">7 ngày tới · AQI trung bình ngày</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({"Ngày":days.index.strftime("%d/%m/%Y"),"AQI":days.round(1).values,"Mức độ":["European AQI" for x in days]}),hide_index=True,use_container_width=True)
            else:
                st.info("Chưa có dữ liệu dự báo 7 ngày.")
    else:
        st.info("Chưa có API dự báo theo giờ. Mô hình RF hiện tại chỉ dự đoán một thời điểm; không tự tạo được chuỗi 24 giờ hoặc 7 ngày.")
    with st.expander("Dự đoán PM2.5 bằng Random Forest · nhập điều kiện môi trường", expanded=False):
        predict_section(live0)


def golden_task(df):
    st.markdown('<div class="sec">04 / Giờ vàng ngoài trời</div>', unsafe_allow_html=True)
    st.caption(f"Giờ vàng tham khảo theo European AQI dự báo tại {SELECTED_CITY}. Các ngưỡng dưới đây chỉ mang tính minh họa, không phải khuyến cáo sức khỏe được kiểm định.")
    a,b=st.columns([1,2],gap="large")
    with a:
        activity=st.radio("Hoạt động",["Đi dạo","Chạy bộ","Đạp xe","Yoga ngoài trời"],key="activity")
        sensitive=st.radio("Thể trạng",["Bình thường","Nhóm nhạy cảm"],key="sensitivity")=="Nhóm nhạy cảm"
    with b:
        hourly=_hourly(df)
        window, details=_golden_window(hourly,activity,sensitive)
        if window:
            st.markdown(f'<div class="lab-intro"><h3>✦ KHUNG GIỜ KHÔNG KHÍ TỐT NHẤT</h3><h2>{window[1]:%H:%M} – {window[2]:%H:%M}</h2><p>AQI trung bình khoảng {window[0]:.0f} · theo dữ liệu dự báo đã cung cấp.</p></div>',unsafe_allow_html=True)
            suitable=(details["Mức độ"]=="Phù hợp").sum()
            worst=details.loc[details["AQI"].idxmax()]
            c1,c2=st.columns(2)
            c1.metric("Số giờ phù hợp",int(suitable))
            c2.metric("Giờ nên tránh nhất",f"{worst['timestamp']:%H:%M} · {worst['AQI']:.0f}")
            st.dataframe(details.assign(timestamp=details["timestamp"].dt.strftime("%d/%m %H:%M")).rename(columns={"timestamp":"Thời gian","AQI":"AQI dự báo"}),hide_index=True,use_container_width=True)
            st.caption("Phân loại phù hợp chỉ là gợi ý thử nghiệm; European AQI không dùng cùng ngưỡng với AQI Trung Quốc. Không dùng làm tư vấn sức khỏe.")
            reminder=f"AirVision AI: {activity.lower()} từ {window[1]:%H:%M} đến {window[2]:%H:%M} ngày {window[1]:%d/%m/%Y}; AQI dự báo trung bình {window[0]:.0f}."
            st.download_button("↓ Lưu gợi ý giờ vàng (.txt)",reminder,file_name="airvision_gio_vang.txt",mime="text/plain")
        else:
            st.info("Chưa đủ ít nhất 2 giờ dự báo liên tiếp có AQI để xác định giờ vàng. Cần kết nối API dự báo theo giờ.")


# ---------------------------------------------------------------- chạy · giữ nguyên hero và hiệu ứng cũ
live0 = fetch_live(SELECTED_CITY)
header(live0)
st.markdown("""<section class="immersive"><div class="immersive-copy"><div class="eyebrow">✦ AIRVISION AI / ENVIRONMENTAL INTELLIGENCE</div><h2>SEE THE<br><em>INVISIBLE.</em></h2><p>Khám phá những gì đang diễn ra trong bầu không khí. Quan trắc trực tuyến, dự báo PM2.5 bằng học máy và thử nghiệm tương lai qua AI Scenario Lab.</p><div class="hero-tags"><span>01 / LIVE MONITORING</span><span>02 / RANDOM FOREST</span><span>03 / SCENARIO LAB</span></div></div><div class="planet-wrap" aria-label="Quả cầu khí quyển chuyển động"><div class="orbit-two"></div><div class="planet"></div><div class="orbit-dot"></div><div class="planet-label">ATMOSPHERIC SYSTEM / 001</div></div></section><div class="ticker"><span>✦ AIR QUALITY INTELLIGENCE &nbsp;&nbsp;&nbsp; ◉ REAL-TIME ENVIRONMENT &nbsp;&nbsp;&nbsp; ✦ MACHINE LEARNING FORECAST &nbsp;&nbsp;&nbsp; ◉ EXPLORE THE INVISIBLE &nbsp;&nbsp;&nbsp; ✦ AIR QUALITY INTELLIGENCE &nbsp;&nbsp;&nbsp; ◉ REAL-TIME ENVIRONMENT &nbsp;&nbsp;&nbsp; ✦ MACHINE LEARNING FORECAST &nbsp;&nbsp;&nbsp; ◉ EXPLORE THE INVISIBLE &nbsp;&nbsp;&nbsp;</span></div>""", unsafe_allow_html=True)

# Tự động lấy dữ liệu khi tải trang, không cần upload CSV.
forecast_df, forecast_source = _load_table("forecast")
station_df, station_source = _load_table("stations")
if station_df.empty:
    station_df = _single_live_station(live0)
    if not station_df.empty:
        station_source = "Trạm realtime hiện tại"

with st.expander("◉ Trạng thái dữ liệu tự động", expanded=False):
    st.caption(f"Chất lượng không khí mô hình: {'Đã kết nối' if live0.ok else 'Chưa kết nối'} · Dự báo nhiều giờ: {forecast_source} · Bản đồ: {station_source}")
    now_pred, now_error = _live_model_now(live0)
    if now_pred is not None:
        st.metric("PM2.5 ước tính bằng Random Forest · điều kiện hiện tại", f"{now_pred.pm25:.1f} µg/m³")
        st.caption("Mô hình huấn luyện trên Bắc Kinh, không được xác nhận độ chính xác cho địa phương đã chọn. Đây không phải dự báo 24 giờ hay 7 ngày.")
    elif now_error:
        st.caption(f"Dự đoán tự động hiện tại chưa sẵn sàng: {now_error}")
    if not FORECAST_API:
        st.caption("Để có dự báo 24 giờ / 7 ngày và Giờ vàng, cấu hình AQ_FORECAST_API trỏ đến endpoint trả về các dự báo có timestamp.")
    if station_df.empty:
        st.caption("Để có bản đồ, cấu hình AQ_STATIONS_API cho nhiều trạm, hoặc AQ_STATION_LAT và AQ_STATION_LON cho trạm hiện tại.")

nav = st.radio("Điều hướng", ["Tổng quan", "Bản đồ", "Dự báo", "Giờ vàng", "AI Scenario Lab"], horizontal=True, label_visibility="collapsed", key="airvision_nav")
if nav == "Tổng quan":
    overview_task(live0, forecast_df)
elif nav == "Bản đồ":
    map_task(station_df)
elif nav == "Dự báo":
    forecast_task(forecast_df)
elif nav == "Giờ vàng":
    golden_task(forecast_df)
else:
    for key, _, default, _, _ in FIELDS:
        st.session_state.setdefault(f"in_{key}", float(live0.get(key, default)))
    scenario_lab(live0)
