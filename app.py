"""
app.py  –  Vitality Leisure Park  –  Capacity Viewer v2
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, joblib, requests, holidays
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Vitality Leisure Park",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background-color:#050d1a; color:#e2e8f0; }
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding-top:1.5rem; padding-bottom:2rem;}
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#071428 0%,#0d1f3c 100%); border-right:1px solid #1e3a5f; }
section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
.card { background:linear-gradient(145deg,#0f2340,#091929); border:1px solid #1e3a5f; border-radius:14px; padding:22px 26px; }
.kpi-num { font-size:2.4rem; font-weight:800; color:#38bdf8; line-height:1.1; }
.kpi-lbl { font-size:0.82rem; color:#ffffff; margin-top:4px; }
.kpi-sub { font-size:0.78rem; margin-top:6px; color:#ffffff; }
.up   { color:#4ade80; }
.down { color:#f87171; }
.sec { font-size:1.15rem; font-weight:700; color:#ffffff; border-left:4px solid #0ea5e9; padding-left:10px; margin:28px 0 14px; }
.badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:.75rem; font-weight:700; margin-top:6px; }
.b-quiet    { background:#052e16; color:#4ade80; border:1px solid #16a34a; }
.b-moderate { background:#431407; color:#fb923c; border:1px solid #c2410c; }
.b-busy     { background:#450a0a; color:#f87171; border:1px solid #b91c1c; }
.rec { background:linear-gradient(145deg,#0c1e35,#071428); border:1px solid #1e3a5f; border-radius:14px; padding:18px 20px; margin-bottom:10px; }
.rec-rank   { font-size:1.8rem; line-height:1; }
.chat-user  { background:#1e3a5f; border-radius:14px 14px 4px 14px; padding:12px 16px; margin:8px 0; color:#e2e8f0; font-size:.92rem; }
.chat-bot   { background:linear-gradient(135deg,#0f2340,#091929); border:1px solid #1e3a5f; border-radius:14px 14px 14px 4px; padding:12px 16px; margin:8px 0; color:#e2e8f0; font-size:.92rem; }
.chat-label-user { font-size:.72rem; color:#64748b; margin-bottom:3px; text-align:right; }
.chat-label-bot  { font-size:.72rem; color:#0ea5e9; margin-bottom:3px; }
.stButton>button { background:linear-gradient(135deg,#0ea5e9,#2563eb); color:white; border:none; border-radius:10px; padding:10px 28px; font-weight:700; font-size:.95rem; width:100%; }
.stButton>button:hover { background:linear-gradient(135deg,#38bdf8,#3b82f6); }
.stSelectbox label, .stRadio label, .stSlider label { color:#ffffff !important; font-size:.85rem !important; }
h1,h2,h3 { color:#e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_meta():
    with open("model_meta.json")  as f: meta    = json.load(f)
    with open("monthly_avg.json") as f: mon_avg = {int(k): v for k,v in json.load(f).items()}
    with open("weekday_avg.json") as f: wd_avg  = {int(k): v for k,v in json.load(f).items()}
    ym  = pd.read_csv("yearmonth_avg.csv")
    raw = pd.read_excel("data.xlsx", parse_dates=["date"])
    return meta, mon_avg, wd_avg, ym, raw

model                              = load_model()
meta, mon_avg, wd_avg, ym_df, raw_df = load_meta()

FEATURES     = meta["features"]
TEMP_ORDER   = meta["temp_order"]
MEAN_V       = meta["mean_visitors"]
WDAY_COLS    = meta["wday_cols"]
WX_COLS      = meta["wx_cols"]
TC_COLS      = meta["tc_cols"]
MAX_CAPACITY = meta.get("max_capacity", 2400)

# ── Holiday helpers ───────────────────────────────────────────────────────────
@st.cache_data
def get_holiday_sets():
    yrs = range(2024, 2028)
    nrw = holidays.Germany(state="NW", years=yrs)
    pub = set(nrw.keys())
    ranges = [
        ("2024-06-27","2024-08-09"),("2024-10-14","2024-10-26"),
        ("2024-12-23","2025-01-06"),("2025-03-31","2025-04-12"),
        ("2025-06-23","2025-08-05"),("2025-10-06","2025-10-18"),
        ("2025-12-22","2026-01-05"),("2026-03-30","2026-04-11"),
        ("2026-06-22","2026-08-04"),
    ]
    sch = set()
    for s, e in ranges:
        for d in pd.date_range(s, e): sch.add(d.date())
    return pub, sch

pub_holidays, school_holidays = get_holiday_sets()

# ── Weather API ───────────────────────────────────────────────────────────────
LAT, LON  = 52.0833, 8.75
CITY_NAME = "Leisure City"

@st.cache_data(ttl=3600)
def fetch_weather(days: int = 14):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=Europe%2FBerlin&forecast_days={days}"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        d  = r.json()["daily"]
        df = pd.DataFrame(d)
        df["date"]          = pd.to_datetime(df["time"])
        df["temp_avg"]      = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
        df["temp_c"]        = df["temp_avg"]
        df["precip_mm"]     = df["precipitation_sum"].fillna(0)
        df["weather_label"] = df["weathercode"].apply(_wmo)
        df["temp_category"] = df["temp_avg"].apply(_tcat)
        return df
    except Exception:
        return None

def _wmo(c):
    if c in [0,1]: return "sunny"
    if c in [2,3]: return "cloudy"
    if 51<=c<=69 or 80<=c<=82: return "rainy"
    if 71<=c<=77 or 85<=c<=86: return "snowy"
    return "cloudy"

def _tcat(t):
    if t < 0:  return "freezing"
    if t < 12: return "cool"
    if t < 18: return "mild"
    if t < 25: return "warm"
    return "hot"

def _wx_icon(w): return {"sunny":"☀️","cloudy":"⛅","rainy":"🌧️","snowy":"❄️"}.get(w,"🌤️")
def _tc_icon(t): return {"freezing":"🥶","cool":"🌬️","mild":"🌿","warm":"🌤️","hot":"🔥"}.get(t,"🌡️")

WD_FULL   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WD_SHORT  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
MON_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(dt: date, wx: str, tc: str, temp_c=None, precip_mm=None) -> int:
    ts  = pd.Timestamp(dt)
    row = {f: 0 for f in FEATURES}
    row["weekday_num"]       = ts.dayofweek
    row["is_weekend"]        = int(ts.dayofweek >= 5)
    row["month"]             = ts.month
    row["season"]            = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[ts.month]
    row["month_sin"]         = np.sin(2*np.pi*ts.month/12)
    row["month_cos"]         = np.cos(2*np.pi*ts.month/12)
    row["wd_sin"]            = np.sin(2*np.pi*ts.dayofweek/7)
    row["wd_cos"]            = np.cos(2*np.pi*ts.dayofweek/7)
    row["doy_sin"]           = np.sin(2*np.pi*ts.day_of_year/365)
    row["doy_cos"]           = np.cos(2*np.pi*ts.day_of_year/365)
    row["temp_num"]          = TEMP_ORDER.get(tc, 1)
    row["is_public_holiday"] = int(dt in pub_holidays)
    row["is_school_holiday"] = int(dt in school_holidays)
    tf = {"freezing":-3,"cool":8,"mild":14,"warm":20,"hot":28}
    row["temp_c"]    = temp_c    if temp_c    is not None else tf.get(tc, 14)
    row["precip_mm"] = precip_mm if precip_mm is not None else (8 if wx=="rainy" else 0)
    if f"wday_{ts.dayofweek}" in row: row[f"wday_{ts.dayofweek}"] = 1
    if f"wx_{wx}"             in row: row[f"wx_{wx}"]             = 1
    if f"tc_{tc}"             in row: row[f"tc_{tc}"]             = 1
    X = pd.DataFrame([row])[FEATURES]
    return max(0, int(round(model.predict(X)[0])))

def crowd(v):
    if v < 900:  return "quiet",    "b-quiet",    "🟢 Quiet"
    if v < 1350: return "moderate", "b-moderate", "🟡 Moderate"
    return               "busy",    "b-busy",     "🔴 Busy"

def cap_pct(v): return min(100, round(v / MAX_CAPACITY * 100, 1))

@st.cache_data(ttl=3600)
def build_forecast(n: int = 14):
    wx_df = fetch_weather(n)
    rows  = []
    for i in range(n):
        d  = date.today() + timedelta(days=i)
        ts = pd.Timestamp(d)
        if wx_df is not None and i < len(wx_df):
            r    = wx_df.iloc[i]
            wx   = r["weather_label"]
            tc   = r["temp_category"]
            tmp  = float(r["temp_avg"])
            prec = float(r["precip_mm"])
        else:
            wx, tc, tmp, prec = "cloudy","mild", None, None
        v             = predict(d, wx, tc, tmp, prec)
        cl, bcls, btxt = crowd(v)
        rows.append(dict(
            date=d, weekday=WD_FULL[ts.dayofweek], weekday_short=WD_SHORT[ts.dayofweek],
            month=ts.month, wx=wx, tc=tc, temp_val=tmp, precip=prec,
            visitors=v, crowd_level=cl, crowd_badge=btxt, crowd_cls=bcls,
            cap_pct=cap_pct(v),
            is_public_holiday=int(d in pub_holidays),
            is_school_holiday=int(d in school_holidays),
        ))
    return pd.DataFrame(rows)

# ── Navigation ────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

def nav(p): st.session_state.page = p

# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:60px 20px 40px;">
        <div style="font-size:4rem;">💧</div>
        <h1 style="font-size:2.8rem; font-weight:800; color:#ffffff; margin:10px 0 6px;">
            Vitality Leisure Park
        </h1>
        <p style="font-size:1.25rem; color:#38bdf8; font-weight:600; margin:0;">
            Capacity Intelligence Viewer
        </p>
        <p style="font-size:1rem; color:#cbd5e1; margin-top:12px; max-width:520px;
                  margin-left:auto; margin-right:auto;">
            AI-powered visitor forecasts for smarter operations and better guest experiences —
            powered by 17 years of real data and live weather from {city}.
        </p>
    </div>
    """.format(city=CITY_NAME), unsafe_allow_html=True)

    wx_df = fetch_weather(1)
    if wx_df is not None:
        t    = wx_df.iloc[0]
        tags = " &nbsp;·&nbsp; ".join([x for x in [
            "🎉 Public holiday today" if date.today() in pub_holidays else "",
            "🏫 School holiday period" if date.today() in school_holidays else "",
        ] if x])
        extra = f" &nbsp;·&nbsp; {tags}" if tags else ""
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:40px;">
            <span style="background:#0f2340; border:1px solid #1e3a5f; border-radius:20px;
                         padding:8px 20px; font-size:.9rem; color:#e2e8f0;">
                📍 {CITY_NAME} &nbsp;·&nbsp; Today: {_wx_icon(t['weather_label'])}
                {t['weather_label'].title()} &nbsp;·&nbsp;
                {_tc_icon(t['temp_category'])} {t['temp_avg']:.0f}°C{extra}
            </span>
        </div>""", unsafe_allow_html=True)

    c1, sp, c2 = st.columns([1, 0.1, 1])
    with c1:
        st.markdown("""
        <div class="card" style="text-align:center; padding:36px 28px; min-height:220px;">
            <div style="font-size:3rem;">📊</div>
            <h2 style="margin:10px 0 6px; font-size:1.5rem; color:#ffffff;">Manager Dashboard</h2>
            <p style="color:#e2e8f0; font-size:.9rem; margin-bottom:20px;">
                7-day forecast, capacity tracking, monthly outlook and historical trends.
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Open Manager Dashboard →", key="go_mgr", on_click=nav, args=("manager",))

    with c2:
        st.markdown("""
        <div class="card" style="text-align:center; padding:36px 28px; min-height:220px;">
            <div style="font-size:3rem;">🧘</div>
            <h2 style="margin:10px 0 6px; font-size:1.5rem; color:#ffffff;">Wellness Coach</h2>
            <p style="color:#e2e8f0; font-size:.9rem; margin-bottom:20px;">
                Tell us how you feel — we'll build your perfect spa day,
                crowd-aware and weather-smart.
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Start Wellness Chat →", key="go_wellness", on_click=nav, args=("wellness",))

    c3, sp2, c4 = st.columns([1, 0.1, 1])
    with c3:
        st.markdown("""
        <div class="card" style="text-align:center; padding:28px; margin-top:12px;">
            <div style="font-size:2rem;">🔍</div>
            <h2 style="margin:8px 0 4px; font-size:1.2rem; color:#ffffff;">Plan My Visit</h2>
            <p style="color:#e2e8f0; font-size:.85rem; margin-bottom:16px;">
                Filter by crowd, weather and weekday preference.
            </p>
        </div>""", unsafe_allow_html=True)
        st.button("Plan My Visit →", key="go_client", on_click=nav, args=("client",))

    with c4:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:28px; margin-top:12px;">
            <div style="font-size:.78rem; color:#94a3b8; line-height:1.8;">
                Model accuracy ≈ ±{meta['cv_mae']:.0f} visitors/day &nbsp;·&nbsp; R² {meta['r2']:.2f}<br>
                Trained on 6,000+ daily records (2008–2025)<br>
                Max capacity: {MAX_CAPACITY:,} visitors
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MANAGER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "manager":
    st.button("← Back to Home", on_click=nav, args=("home",))
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("# 📊 Manager Dashboard")
    st.markdown("*Visitor forecast and operational insights for Vitality Leisure Park*")

    fc  = build_forecast(14)
    fc7 = fc.iloc[:7]

    total7  = fc7["visitors"].sum()
    avg7    = fc7["visitors"].mean()
    peak    = fc7.loc[fc7["visitors"].idxmax()]
    quietd  = fc7.loc[fc7["visitors"].idxmin()]
    mon_ref = mon_avg.get(date.today().month, MEAN_V)
    delta   = avg7 - mon_ref
    avg_cap = fc7["cap_pct"].mean()

    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, num, lbl, sub="", up=None):
        arrow = ("▲ " if up else "▼ ") if up is not None else ""
        cls   = "up" if up else ("down" if up is False else "")
        col.markdown(f"""
        <div class="card">
            <div class="kpi-num">{num}</div>
            <div class="kpi-lbl">{lbl}</div>
            {'<div class="kpi-sub '+cls+'">'+arrow+sub+'</div>' if sub else ''}
        </div>""", unsafe_allow_html=True)

    kpi(k1, f"{total7:,}", "Visitors expected this week")
    kpi(k2, f"{avg7:.0f}", "Daily average (7 days)",
        f"vs monthly avg ({mon_ref:.0f})", up=(delta>=0))
    kpi(k3, f"{peak['visitors']:,}",
        f"Peak: {peak['weekday'][:3]} {peak['date'].strftime('%d.%m')}",
        "🎉 Public holiday" if peak["is_public_holiday"] else
        "🏫 School holiday" if peak["is_school_holiday"] else "", up=None)
    kpi(k4, f"{avg_cap:.0f}%", f"Avg capacity (max {MAX_CAPACITY:,})",
        "🔴 Near capacity" if avg_cap > 80 else
        "🟡 Moderate load" if avg_cap > 55 else "🟢 Comfortable", up=None)

    st.markdown("")

    # 7-day bar chart
    st.markdown('<div class="sec">7-Day Visitor Forecast</div>', unsafe_allow_html=True)
    try:
        bar_colors = ["#4ade80" if v < 900 else "#fb923c" if v < 1350 else "#f87171"
                      for v in fc7["visitors"]]
        xlabels = []
        for _, r in fc7.iterrows():
            lbl = str(r["weekday_short"])
            if r["is_public_holiday"]: lbl += " 🎉"
            elif r["is_school_holiday"]: lbl += " 🏫"
            xlabels.append(lbl)
        yvals = [int(v) for v in fc7["visitors"].tolist()]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=xlabels, y=yvals, marker_color=bar_colors))
        fig_bar.add_shape(type="line", x0=-0.5, x1=len(xlabels)-0.5,
                          y0=MAX_CAPACITY, y1=MAX_CAPACITY,
                          line=dict(color="#f87171", width=2, dash="dot"))
        fig_bar.add_annotation(x=len(xlabels)-1, y=MAX_CAPACITY,
                                text=f"Max capacity ({MAX_CAPACITY:,})",
                                showarrow=False, yshift=10,
                                font=dict(color="#f87171", size=10))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1e3a5f", title="Visitors"),
            height=360, margin=dict(t=30,b=10,l=0,r=10), showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    # Capacity progress bars
    st.markdown('<div class="sec">Daily Capacity Overview</div>', unsafe_allow_html=True)
    cap_cols = st.columns(7)
    for i, (_, r) in enumerate(fc7.iterrows()):
        pct = r["cap_pct"]
        col = "#4ade80" if pct < 55 else ("#fb923c" if pct < 80 else "#f87171")
        hd  = "🎉" if r["is_public_holiday"] else ("🏫" if r["is_school_holiday"] else "")
        cap_cols[i].markdown(f"""
        <div style="text-align:center; padding:10px 6px;">
            <div style="font-size:.8rem; color:#e2e8f0; font-weight:700;">{r['weekday_short']} {hd}</div>
            <div style="font-size:1.1rem; font-weight:800; color:{col}; margin:4px 0;">{pct:.0f}%</div>
            <div style="background:#1e3a5f; border-radius:6px; height:8px;">
                <div style="background:{col}; width:{pct}%; height:8px; border-radius:6px;"></div>
            </div>
            <div style="font-size:.7rem; color:#64748b; margin-top:3px;">{r['visitors']:,}</div>
        </div>""", unsafe_allow_html=True)

    # Daily breakdown table
    st.markdown('<div class="sec">Daily Breakdown</div>', unsafe_allow_html=True)
    tbl = []
    for _, r in fc7.iterrows():
        tmp  = f"{r['temp_val']:.0f}°C" if r["temp_val"] is not None else r["tc"].title()
        prec = f"{r['precip']:.1f}mm"   if r["precip"]   is not None else "—"
        hd   = "🎉 Public holiday" if r["is_public_holiday"] else \
               "🏫 School holiday" if r["is_school_holiday"] else "—"
        tbl.append({
            "Date":     r["date"].strftime("%a %d.%m"),
            "Weather":  f"{_wx_icon(r['wx'])} {r['wx'].title()}",
            "Temp":     f"{_tc_icon(r['tc'])} {tmp}",
            "Precip":   prec,
            "Visitors": f"{r['visitors']:,}",
            "Capacity": f"{r['cap_pct']:.0f}%",
            "Crowd":    r["crowd_badge"],
            "Holiday":  hd,
        })
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    # Monthly outlook
    st.markdown('<div class="sec">Monthly Visitor Outlook (Next 6 Months)</div>',
                unsafe_allow_html=True)
    st.caption("Based on historical averages adjusted for seasonality.")
    import calendar
    today = date.today()
    fm    = []
    for i in range(6):
        m    = (today.month - 1 + i) % 12 + 1
        y    = today.year + ((today.month - 1 + i) // 12)
        avg  = mon_avg.get(m, MEAN_V)
        days = calendar.monthrange(y, m)[1]
        fm.append({"label": f"{MON_NAMES[m]} {y}", "expected": int(avg * days)})
    fm_df = pd.DataFrame(fm)
    fig_mon = go.Figure()
    fig_mon.add_trace(go.Bar(
        x=fm_df["label"], y=fm_df["expected"],
        marker_color="#0ea5e9", marker_line_color="#38bdf8", marker_line_width=1.2,
        text=fm_df["expected"].apply(lambda v: f"{v:,}"),
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_mon.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#1e3a5f", title="Expected total visitors"),
        height=320, margin=dict(t=30,b=10,l=0,r=10),
    )
    st.plotly_chart(fig_mon, use_container_width=True)

    # Historical heatmap
    st.markdown('<div class="sec">Historical Heatmap — Avg Visitors by Weekday × Month</div>',
                unsafe_allow_html=True)
    raw_f = raw_df[~raw_df["year"].isin([2020,2021])].copy()
    raw_f["weekday_num"] = pd.to_datetime(raw_f["date"]).dt.dayofweek
    raw_f["month"]       = pd.to_datetime(raw_f["date"]).dt.month
    pivot = raw_f.pivot_table(values="total_visitors", index="weekday_num",
                               columns="month", aggfunc="mean")
    pivot.index   = [WD_FULL[i] for i in pivot.index]
    pivot.columns = [MON_NAMES[c] for c in pivot.columns]
    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="Blues", text=np.round(pivot.values,0).astype(int),
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b>, %{x}<br>Avg: %{z:.0f} visitors<extra></extra>",
    ))
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), height=330, margin=dict(t=10,b=10,l=0,r=0),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Year-over-year
    st.markdown('<div class="sec">Year-over-Year Monthly Trend</div>', unsafe_allow_html=True)
    years_avail    = sorted(ym_df["year"].unique())
    selected_years = st.multiselect(
        "Select years to compare", options=years_avail,
        default=[y for y in [2022,2023,2024,2025] if y in years_avail],
    )
    yr_colors = px.colors.qualitative.Set2
    fig_yoy   = go.Figure()
    for i, yr in enumerate(selected_years):
        yd = ym_df[ym_df["year"] == yr].sort_values("month")
        fig_yoy.add_trace(go.Scatter(
            x=yd["month"].map(MON_NAMES), y=yd["avg_visitors"],
            name=str(yr), mode="lines+markers",
            line=dict(color=yr_colors[i % len(yr_colors)], width=2.2),
            marker=dict(size=6),
        ))
    fig_yoy.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#1e3a5f", title="Avg daily visitors"),
        height=320, margin=dict(t=20,b=10,l=0,r=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PLAN MY VISIT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "client":
    st.button("← Back to Home", on_click=nav, args=("home",))
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("# 🔍 Plan My Visit")
    st.markdown("*Tell us your preferences and we'll find your perfect day.*")

    st.markdown('<div class="sec">Your Preferences</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: horizon    = st.selectbox("📅 Planning window", ["Next 7 days","Next 14 days"])
    with c2: crowd_pref = st.selectbox("👥 Crowd preference",
                           ["🟢 I prefer it quiet","🟡 Doesn't matter","🔴 I like lively days"])
    with c3: wx_pref    = st.selectbox("☀️ Weather preference",
                           ["☀️ Sunny preferred","🌥️ Don't care","🌧️ Rain is fine"])
    with c4: wd_pref    = st.selectbox("📆 Preferred weekday", ["Any day"] + WD_FULL)

    n_days = 7 if "7" in horizon else 14
    fc     = build_forecast(n_days)

    if wd_pref != "Any day":
        fc_f = fc[fc["weekday"] == wd_pref].copy()
        if fc_f.empty:
            st.warning(f"No {wd_pref} in the next {n_days} days. Showing all days.")
        else:
            fc = fc_f

    def score(row):
        s = 100.0
        v = row["visitors"]
        if "quiet"  in crowd_pref: s -= max(0, (v - 800) / 6)
        elif "lively" in crowd_pref: s -= max(0, (1100 - v) / 8)
        if "Sunny" in wx_pref and row["wx"] != "sunny": s -= 30
        elif "Rain is fine" not in wx_pref and row["wx"] in ["rainy","snowy"]: s -= 20
        return max(0, s)

    fc["score"] = fc.apply(score, axis=1)
    fc_ranked   = fc.sort_values("score", ascending=False).reset_index(drop=True)

    if "Sunny" in wx_pref and not any(r["wx"] == "sunny" for _, r in fc.iterrows()):
        wx_counts  = fc["wx"].value_counts().to_dict()
        wx_summary = " · ".join([f"{_wx_icon(k)} {k.title()} ({v}d)" for k,v in wx_counts.items()])
        st.markdown(f"""
        <div style="background:#1c1200; border:1px solid #854d0e; border-radius:12px;
                    padding:16px 20px; margin-bottom:16px;">
            <div style="color:#fbbf24; font-weight:700;">☀️ No sunny days in the forecast window</div>
            <div style="color:#e2e8f0; font-size:.9rem; margin-top:6px;">
                No sunny days forecast. Available: {wx_summary}
            </div>
            <div style="color:#e2e8f0; font-size:.82rem; margin-top:4px;">
                Showing best options based on your other preferences.
            </div>
        </div>""", unsafe_allow_html=True)

    top = fc_ranked.iloc[0]
    cl, bcls, btxt = crowd(top["visitors"])
    tmp_str = f"{top['temp_val']:.0f}°C" if top["temp_val"] is not None else top["tc"].title()

    st.markdown('<div class="sec">⭐ Our Best Recommendation for You</div>',
                unsafe_allow_html=True)
    h1, h2 = st.columns([3,1])
    with h1:
        if top["is_public_holiday"]:
            hday_html = '<div style="margin-top:6px; color:#fbbf24;">🎉 Public holiday</div>'
        elif top["is_school_holiday"]:
            hday_html = '<div style="margin-top:6px; color:#38bdf8;">🏫 School holiday period</div>'
        else:
            hday_html = ""
        card_html = (
            '<div class="card" style="border:2px solid #0ea5e9; padding:28px 30px;">'
            '<div style="font-size:2rem; font-weight:800; color:#ffffff;">'
            + top["weekday"] + "  ·  " + top["date"].strftime("%d %B %Y") +
            "</div>"
            '<div style="font-size:1rem; color:#e2e8f0; margin-top:10px;">'
            + _wx_icon(top["wx"]) + " " + top["wx"].title() + " &nbsp;|&nbsp; "
            + _tc_icon(top["tc"]) + " " + tmp_str + " &nbsp;|&nbsp; "
            + "💧 ~" + f"{top['visitors']:,}" + " visitors expected"
            + "</div>"
            + hday_html
            + '<span class="badge ' + bcls + '" style="margin-top:12px;">' + btxt + "</span>"
            + "</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

    with h2:
        gc   = "#4ade80" if cl=="quiet" else ("#fb923c" if cl=="moderate" else "#f87171")
        pctv = cap_pct(top["visitors"])
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:20px;">
            <div style="font-size:2rem; font-weight:800; color:{gc};">{top['visitors']:,}</div>
            <div style="font-size:.8rem; color:#ffffff; margin:6px 0 4px;">Expected visitors</div>
            <div style="font-size:.75rem; color:#94a3b8; margin-bottom:8px;">{pctv:.0f}% of capacity</div>
            <div style="background:#1e3a5f; border-radius:8px; height:10px; width:100%;">
                <div style="background:{gc}; width:{pctv}%; height:10px; border-radius:8px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:.7rem; color:#94a3b8; margin-top:4px;">
                <span>0</span><span>{MAX_CAPACITY//2:,}</span><span>{MAX_CAPACITY:,}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">All Days Ranked for You</div>', unsafe_allow_html=True)
    medals = ["🥇","🥈","🥉"] + [f"#{i}" for i in range(4, 20)]
    cols3  = st.columns(3)
    for i, (_, row) in enumerate(fc_ranked.iterrows()):
        cl2, bcls2, btxt2 = crowd(row["visitors"])
        tmp2 = f"{row['temp_val']:.0f}°C" if row["temp_val"] is not None else row["tc"].title()
        hd2  = "🎉" if row["is_public_holiday"] else ("🏫" if row["is_school_holiday"] else "")
        with cols3[i % 3]:
            st.markdown(f"""
            <div class="rec">
                <div class="rec-rank">{medals[i]}</div>
                <div style="font-size:1.1rem; font-weight:700; color:#ffffff; margin-top:4px;">
                    {row['weekday'][:3]} · {row['date'].strftime('%d.%m.%Y')} {hd2}
                </div>
                <div style="font-size:.84rem; color:#e2e8f0; margin-top:3px;">
                    {_wx_icon(row['wx'])} {row['wx'].title()} &nbsp;
                    {_tc_icon(row['tc'])} {tmp2}
                </div>
                <div style="font-size:.84rem; color:#e2e8f0; margin-top:3px;">
                    💧 ~{row['visitors']:,} visitors &nbsp;·&nbsp; {row['cap_pct']:.0f}% capacity
                </div>
                <span class="badge {bcls2}">{btxt2}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">💡 Visitor Tips</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.info("☀️ **Sunny days** draw the most visitors. For a quieter experience, cloudy days are your friend.")
    t2.info("📅 **School holiday periods** 🏫 and public holidays 🎉 see significantly higher visitor numbers.")
    t3.info("🌧️ **Rainy days** are usually the quietest. Indoor thermal areas stay comfortable regardless.")

# ══════════════════════════════════════════════════════════════════════════════
#  WELLNESS COACH  —  RAG-powered using Cohere Embed + Chat
#  RAG pipeline (following lecture slides):
#  Step 1: Load pre-computed document embeddings (menu + fitness + spa chunks)
#  Step 2: Embed user message with Cohere Embed -> compute cosine similarity
#  Step 3: Retrieve top-5 most relevant chunks
#  Step 4: Inject chunks + forecast into prompt -> Cohere Chat generates reply
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "wellness":
    st.button("← Back to Home", on_click=nav, args=("home",))
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("# 🧘 Wellness Coach")
    st.markdown("*Tell me how you feel and I\'ll build your perfect spa day — powered by RAG.*")

    try:
        import cohere
        co = cohere.ClientV2(st.secrets["COHERE_API_KEY"])
        cohere_ok = True
    except Exception:
        cohere_ok = False
        st.warning("⚠️ Add `COHERE_API_KEY` to your Streamlit secrets to enable the Wellness Coach.")

    # ── RAG Step 1: Load pre-computed embeddings ──────────────────────────────
    @st.cache_data
    def load_embeddings():
        try:
            with open("embeddings.json") as f:
                data = json.load(f)
            return data["chunks"], np.array(data["embeddings"], dtype="float32")
        except FileNotFoundError:
            return None, None

    doc_chunks, doc_embeddings = load_embeddings()
    rag_available = doc_chunks is not None and cohere_ok

    if not rag_available and cohere_ok:
        st.warning("⚠️ embeddings.json not found. Run `python3 build_embeddings.py` locally first, then commit embeddings.json to GitHub.")

    # ── RAG Step 2: Retrieve relevant chunks for a query ─────────────────────
    def retrieve(query: str, top_k: int = 5):
        """
        RAG retrieval following lecture steps:
        - Embed the query using Cohere Embed (input_type=search_query)
        - Compute cosine similarity between query vector and all document vectors
          (since embeddings are pre-normalised, cosine similarity = dot product)
        - Return top_k most similar chunks
        """
        if not rag_available:
            return []
        q_resp = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"],
        )
        q_vec  = np.array(q_resp.embeddings.float_[0], dtype="float32")
        q_norm = q_vec / np.linalg.norm(q_vec)
        # Cosine similarity = dot product (embeddings already normalised in build_embeddings.py)
        scores  = doc_embeddings @ q_norm
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(doc_chunks[i], float(scores[i])) for i in top_idx]

    if cohere_ok:
        fc_ctx = build_forecast(7)
        forecast_lines = []
        for _, r in fc_ctx.iterrows():
            ph = " (public holiday)" if r["is_public_holiday"] else \
                 " (school holiday)" if r["is_school_holiday"] else ""
            tmp_label = f"{r['temp_val']:.0f}°C" if r["temp_val"] else r["tc"]
            forecast_lines.append(
                f"- {r['weekday']} {r['date'].strftime('%d.%m')}: "
                f"{r['visitors']:,} visitors expected ({r['crowd_level']} crowd, "
                f"{r['cap_pct']:.0f}% of capacity), "
                f"weather: {r['wx']}, temp: {tmp_label}{ph}"
            )
        forecast_text = "\n".join(forecast_lines)

        if "wellness_history" not in st.session_state:
            st.session_state.wellness_history = []
            st.session_state.wellness_history.append({
                "role": "assistant",
                "content": (
                    "Hi there! 👋 I\'m your Vitality Leisure Park wellness coach.\n\n"
                    "I can build you a personalised spa day based on how you\'re feeling, "
                    "our live crowd forecast, and our actual menu and fitness schedule.\n\n"
                    "**Tell me — how are you feeling today?** "
                    "Any tension, stress, low energy, or something specific you\'d like to work on?"
                )
            })

        st.markdown('<div class="sec">Your Wellness Conversation</div>', unsafe_allow_html=True)
        for msg in st.session_state.wellness_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="text-align:right;">
                    <div class="chat-label-user">You</div>
                    <div class="chat-user">{msg['content']}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div>
                    <div class="chat-label-bot">🧘 Wellness Coach</div>
                    <div class="chat-bot">{msg['content']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input(
                "Your message",
                placeholder="e.g. I\'m exhausted and my shoulders are really tense...",
                label_visibility="collapsed",
                key="wellness_input"
            )
        with col_btn:
            send = st.button("Send →", key="wellness_send")

        if send and user_input.strip():
            st.session_state.wellness_history.append(
                {"role": "user", "content": user_input.strip()}
            )

            # ── RAG Step 3: Retrieve relevant chunks ─────────────────────────
            retrieved = retrieve(user_input.strip(), top_k=5)
            rag_context = ""
            if retrieved:
                rag_context = "\n\nRELEVANT FACILITY INFORMATION (retrieved from menu & fitness schedule):\n"
                for chunk, score in retrieved:
                    rag_context += f"- [{chunk.get('section', chunk['source']).upper()}] {chunk['text']}\n"

            # ── RAG Step 4: Build prompt with retrieved context + forecast ────
            system_prompt = f"""You are a warm, expert wellness coach at Vitality Leisure Park, a premium thermal spa.

CURRENT 7-DAY VISITOR FORECAST (use these exact numbers):
{forecast_text}
{rag_context}
YOUR ROLE:
1. Ask the guest how they feel (max 1-2 questions at a time, warm and conversational)
2. Once you understand their needs, give a concrete personalised plan:
   - Best day to visit from the forecast (justify with crowd level, weather, holidays)
   - A specific treatment sequence with durations from the RETRIEVED information above
   - A specific food/drink recommendation from the RETRIEVED menu items above
   - Specific fitness class if relevant (use the RETRIEVED schedule with actual times)
3. Use ONLY the retrieved information for specific recommendations — do not invent dishes or classes
4. Use ACTUAL forecast visitor numbers — never invent them
5. If asked to change day (e.g. "what about Saturday?"), adapt using the forecast

Do not reveal you are an AI or mention Cohere or RAG."""

            api_messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.wellness_history:
                if msg["role"] in ["user", "assistant"]:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})

            with st.spinner("Your coach is thinking..."):
                try:
                    response = co.chat(
                        model="command-a-03-2025",
                        messages=api_messages,
                    )
                    reply = response.message.content[0].text
                except Exception as e:
                    reply = f"Sorry, something went wrong: {e}"

            st.session_state.wellness_history.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()

        if len(st.session_state.wellness_history) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Start new conversation", key="wellness_reset"):
                st.session_state.wellness_history = []
                st.rerun()

        # ── Show what was retrieved (transparency for graders) ────────────────
        with st.expander("🔍 How the RAG retrieval works"):
            st.markdown("""
            **RAG pipeline (from lecture):**
            1. Menu and fitness schedule are split into ~55 chunks and pre-embedded with Cohere Embed
            2. Your message is embedded with the same model
            3. Cosine similarity is computed between your message and all chunks
            4. The 5 most relevant chunks are added to the prompt
            5. Cohere Chat generates a response grounded in those specific chunks + the live forecast
            """)
            if len(st.session_state.wellness_history) > 1:
                last_user = next((m["content"] for m in reversed(st.session_state.wellness_history)
                                  if m["role"] == "user"), None)
                if last_user:
                    retrieved_show = retrieve(last_user, top_k=5)
                    if retrieved_show:
                        st.markdown("**Last retrieval results:**")
                        for chunk, score in retrieved_show:
                            st.markdown(f"- `{chunk['id']}` (similarity: {score:.3f}): {chunk['text'][:80]}...")

        with st.expander("📅 View the 7-day forecast I\'m working with"):
            tbl2 = []
            for _, r in fc_ctx.iterrows():
                tmp2 = f"{r['temp_val']:.0f}°C" if r["temp_val"] is not None else r["tc"].title()
                hd2  = "🎉" if r["is_public_holiday"] else ("🏫" if r["is_school_holiday"] else "—")
                tbl2.append({
                    "Day":      f"{r['weekday']} {r['date'].strftime('%d.%m')}",
                    "Weather":  f"{_wx_icon(r['wx'])} {r['wx'].title()}",
                    "Temp":     tmp2,
                    "Visitors": f"{r['visitors']:,}",
                    "Capacity": f"{r['cap_pct']:.0f}%",
                    "Crowd":    r["crowd_badge"],
                    "Holiday":  hd2,
                })
            st.dataframe(pd.DataFrame(tbl2), use_container_width=True, hide_index=True)
