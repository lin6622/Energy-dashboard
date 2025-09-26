import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import os
from pathlib import Path
import gdown

# =========================
# Page Setup
# =========================
st.set_page_config(layout="wide", page_title="Smart Building AI Dashboard", page_icon="⚡")


def _apply_sidebar_state(is_open: bool, open_width_px: int = 380):
    if is_open:
        st.markdown(f"""
        <style>
          [data-testid="stSidebarNav"] {{ display: none !important; }}
          /* Make  overlay */
          [data-testid="stSidebar"] {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            width: {open_width_px}px !important;
            min-width: {open_width_px}px !important;
            max-width: {open_width_px}px !important;
            background: white;
            box-shadow: 0 0 24px rgba(0,0,0,0.18) !important;
            transform: translateX(0) !important;
            visibility: visible !important;
            z-index: 1000 !important;
            overflow: auto !important;
            border-right: 1px solid rgba(0,0,0,.08);
          }}
          /* Keep main app full width */
          [data-testid="stAppViewContainer"] {{ margin-left: 0 !important; }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
          [data-testid="stSidebarNav"] { display: none !important; }

          /* Hide the sidebar off-canvas */
          [data-testid="stSidebar"]{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            transform: translateX(-100%) !important;
            visibility: hidden !important;
            overflow: hidden !important;
          }

          /* Keep main app full width */
          [data-testid="stAppViewContainer"]{ margin-left: 0 !important; }
        </style>
        """, unsafe_allow_html=True)


# --- UI state for assistant ---
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat" not in st.session_state:
    st.session_state.chat = []

# Apply sidebar state on every run
_apply_sidebar_state(st.session_state.show_chat, open_width_px=380)

# =========================
# Google Drive: download folder & locate files
# =========================
# Drive folder link:
# https://drive.google.com/drive/folders/1s6mIYgZ32hVyvypjPFGL4mUd_Uwz6cB1?usp=drive_link
DEFAULT_GDRIVE_FOLDER_ID = "1s6mIYgZ32hVyvypjPFGL4mUd_Uwz6cB1"


def _get_secret(key: str, default: str) -> str:
    # Try Streamlit secrets; if not present, fall back to env var; then default
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


FOLDER_ID = _get_secret("GDRIVE_FOLDER_ID", DEFAULT_GDRIVE_FOLDER_ID)

LOCAL_CACHE_DIR = Path("Dashboard_pkl_data_cache")  # local cache dir


def download_folder_if_needed(folder_id: str, out_dir: Path) -> Path:
    """Download entire Drive folder once into out_dir (if empty/missing)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_files = any(out_dir.rglob("*"))
    if not has_files:
        gdown.download_folder(
            id=folder_id,
            output=str(out_dir),
            quiet=False,
            use_cookies=False  # folder must be shared "Anyone with the link"
        )
    return out_dir


def find_file(root: Path, prefer_contains: list[str], extensions: tuple[str, ...]) -> Path | None:
    """Find a file under root that matches preferred name hints and extension; fallback to first extension match."""
    root = Path(root)
    # pass 1: prefer names containing hints
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            name = p.name.lower()
            if any(h in name for h in (h.lower() for h in prefer_contains)):
                return p
    # pass 2: first file with extension
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            return p
    return None


@st.cache_resource(show_spinner=True)
def load_model_and_data_from_drive(folder_id: str):
    # ensure local copy of the folder
    local_root = download_folder_if_needed(folder_id, LOCAL_CACHE_DIR)

    # try to find model .pkl and dataset .csv
    model_path = find_file(
        local_root,
        prefer_contains=["stacked", "model", "xgb", "rfr"],
        extensions=(".pkl",)
    )
    data_path = find_file(
        local_root,
        prefer_contains=["smart_building", "dataset", "data"],
        extensions=(".csv",)
    )

    if model_path is None:
        st.error("Couldn't find a .pkl model in the Google Drive folder. "
                 "Please ensure the folder contains your model file.")
        st.stop()
    if data_path is None:
        st.error("Couldn't find a .csv dataset in the Google Drive folder. "
                 "Please ensure the folder contains your dataset file.")
        st.stop()

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    return model, df, model_path, data_path


# =========================
# Load model and data
# =========================
model, df, _model_path, _data_path = load_model_and_data_from_drive(FOLDER_ID)

# Preprocessing
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
df['hour'] = df['datetime_utc'].dt.hour
df['day_of_week'] = df['datetime_utc'].dt.dayofweek
df['month'] = df['datetime_utc'].dt.month
df['temp_cooling_interaction'] = df['Weather_Station_Weather_Ta'] * df['total_cooling_pow']

features = [
    'Weather_Station_Weather_Igm', 'Weather_Station_Weather_Ta',
    'total_cooling_pow', 'cool_elec', 'PV', 'CHP_elec',
    'total_heat_prod', 'CHP_heat', 'hour', 'day_of_week',
    'month', 'temp_cooling_interaction'
]

# =========================
# Header with assistant toggle
# =========================
h1, h2 = st.columns([8, 2])
with h1:
    st.title("Smart Building Power Predictor")
    st.markdown("##### The energy consumption is calculated **per hour** based on selected parameters.")
with h2:
    label = " Open Energy Assistant" if not st.session_state.show_chat else "✖ Close Assistant"
    if st.button(label, use_container_width=True):
        st.session_state.show_chat = not st.session_state.show_chat
        st.rerun()

# =========================
# Layout: Inputs on Left, Output on Right
# =========================
left, right = st.columns([1, 2])

with left:
    with st.container(border=True):
        st.subheader("Input parameters")
        hour = st.slider("Hour", 0, 23, 12)
        temp = st.number_input("Temperature (Ta)", value=25.0, step=0.1, format="%.2f")
        cooling = st.number_input("Cooling Power", value=500.0, step=100.0, format="%.2f")
        PV = st.slider("PV Output", 0, 1000, 200)
        CHP_elec = st.slider("CHP Electricity", -200000, 200000, 0)
        CHP_heat = st.slider("CHP Heat", 0, 300000, 100000)
        heat_prod = st.slider("Total Heat Production", 0, 500000, 250000)
        cool_elec = st.slider("Cool Electric Load", 0, 100000, 50000)
        igm = st.slider("IGM Radiation", 0, 1000, 500)
        day_of_week = st.selectbox("Day of Week", list(range(7)),
                                   format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        month = st.selectbox("Month", list(range(1, 13)))
        temp_cooling_interaction = temp * cooling

    input_df = pd.DataFrame([[igm, temp, cooling, cool_elec, PV, CHP_elec, heat_prod, CHP_heat,
                              hour, day_of_week, month, temp_cooling_interaction]], columns=features)

with left:
    with st.container(border=True):
        st.subheader("Average Power Draw by Day of Week")
        avg_power_day = df.groupby('day_of_week')['total_elec_power_drawn'].mean().reset_index()
        avg_power_day['day'] = avg_power_day['day_of_week'].map({
            0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
        })
        fig_day = px.bar(
            avg_power_day, x='day', y='total_elec_power_drawn',
            labels={'total_elec_power_drawn': 'Avg Power (kW)', 'day': 'Day of Week'},
            title="Average Power by Day", text_auto='.2s'
        )
        st.plotly_chart(fig_day, use_container_width=True)

    with st.container(border=True):
        st.subheader("Power Draw Distribution")
        fig = px.histogram(
            df, x='total_elec_power_drawn',
            nbins=50, title="Distribution of Hourly Power Draw (kW)",
            labels={'total_elec_power_drawn': 'Power Draw (kW)'}
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    with st.container(border=True):
        st.subheader(" Energy consumption (kW per Hour)")
        y_pred = model.predict(input_df)[0]
        st.metric(label="Energy", value=f"{y_pred:,.2f} kW")

    with st.container(border=True):
        st.subheader("How Inputs Affect Predicted Energy consumption")
        simulate_df = df.sample(100).copy()
        simulate_df['hour'] = hour
        simulate_df['Weather_Station_Weather_Ta'] = temp
        simulate_df['total_cooling_pow'] = cooling
        simulate_df['temp_cooling_interaction'] = simulate_df['Weather_Station_Weather_Ta'] * simulate_df[
            'total_cooling_pow']
        simulate_df_pred = model.predict(simulate_df[features])
        st.plotly_chart(
            px.line(
                x=range(len(simulate_df_pred)), y=simulate_df_pred,
                labels={'x': 'Simulation Point', 'y': 'Predicted Power Draw'},
                title="Simulated Power Draw With Input Changes"
            ),
            use_container_width=True
        )

    with st.container(border=True):
        st.subheader("Temperature vs Power Draw")
        st.plotly_chart(
            px.scatter(
                df.sample(200),
                x='Weather_Station_Weather_Ta', y='total_elec_power_drawn',
                color='total_cooling_pow',
                title="Temperature vs Energy consumption by cooling"
            ),
            use_container_width=True
        )

    with st.container(border=True):
        st.subheader("Energy Trends")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.line(df.head(200), x='datetime_utc', y='total_elec_power_drawn',
                        title="Electricity Draw Over Time"),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                px.bar(df.head(100), x='hour', y='total_cooling_pow',
                       title="Cooling Power by Hour"),
                use_container_width=True
            )
    with st.container(border=True):
        st.subheader("Dataset Sample")
        st.dataframe(
            df.head(8),
            height=220,
            use_container_width=True)

# =========================
# Forecasts
# =========================
st.markdown(
    "<h3 style='padding:0px;'> Forecasts</h3>",
    unsafe_allow_html=True)
with st.container(border=True):
    # Controls
    colA, colB, colC = st.columns([1.2, 1, 1])
    with colA:
        default_start = (df["datetime_utc"].max() + pd.Timedelta(hours=1)).to_pydatetime()
        date_val = st.date_input("Forecast start date (UTC)", value=default_start.date())
        time_val = st.time_input("Forecast start time (UTC)", value=default_start.time())
        start_dt = dt.datetime.combine(date_val, time_val)
    with colB:
        horizon = st.number_input("Horizon (hours)", min_value=1, max_value=168, value=24, step=1)
    with colC:
        overlay_hist = st.checkbox("Overlay recent history", value=True)
    # Future index & calendar features
    future_index = pd.date_range(start=start_dt, periods=horizon, freq="H")
    future = pd.DataFrame({"datetime_utc": future_index})
    future["hour"] = future["datetime_utc"].dt.hour
    future["day_of_week"] = future["datetime_utc"].dt.dayofweek
    future["month"] = future["datetime_utc"].dt.month

    # Exogenous columns (drivers)
    exog_cols = [
        "Weather_Station_Weather_Igm", "Weather_Station_Weather_Ta",
        "total_cooling_pow", "cool_elec", "PV",
        "CHP_elec", "total_heat_prod", "CHP_heat"
    ]
    # --- Last-day pattern ---
    last_n = min(24, len(df))
    tail = df.tail(last_n).copy()
    for c in exog_cols:
        if c not in tail:
            # safety, in case a column is missing in the tail slice
            tail[c] = df[c].iloc[-last_n:]
    reps = int(np.ceil(horizon / last_n))
    patterned = pd.concat([tail[exog_cols]] * reps, ignore_index=True).iloc[:horizon]
    for c in exog_cols:
        future[c] = patterned[c].to_numpy()

    # Derived interaction
    future["temp_cooling_interaction"] = (
            future["Weather_Station_Weather_Ta"] * future["total_cooling_pow"]
    )

    # Predict
    future_pred = model.predict(future[features])
    future["Predicted_kW"] = future_pred

    # Approx uncertainty band (±1.96 * RMSE)
    try:
        sample_hist = df.sample(min(5000, len(df)), random_state=42)
        hist_pred = model.predict(sample_hist[features])
        resid = sample_hist["total_elec_power_drawn"].to_numpy() - hist_pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
    except Exception:
        rmse = None

    # Plot
    fig_fc = go.Figure()
    if overlay_hist:
        hist_window_start = df["datetime_utc"].max() - pd.Timedelta(days=7)
        hist = df[df["datetime_utc"] >= hist_window_start][["datetime_utc", "total_elec_power_drawn"]]
        fig_fc.add_trace(go.Scatter(
            x=hist["datetime_utc"], y=hist["total_elec_power_drawn"],
            mode="lines", name="Actual (last 7 days)", line=dict(width=1)
        ))

    fig_fc.add_trace(go.Scatter(
        x=future["datetime_utc"], y=future["Predicted_kW"],
        mode="lines+markers", name="Forecast", line=dict(width=2)
    ))

    if rmse is not None:
        upper = future["Predicted_kW"] + 1.96 * rmse
        lower = future["Predicted_kW"] - 1.96 * rmse
        fig_fc.add_trace(go.Scatter(
            x=future["datetime_utc"], y=upper, mode="lines",
            showlegend=False, line=dict(width=0)
        ))
        fig_fc.add_trace(go.Scatter(
            x=future["datetime_utc"], y=lower, mode="lines",
            showlegend=False, line=dict(width=0), fill="tonexty",
            fillcolor="rgba(0,0,0,0.08)"
        ))

    fig_fc.update_layout(
        title="Forecasted Electricity Consumption",
        xaxis_title="Datetime (UTC)",
        yaxis_title="kW",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    out = future[["datetime_utc"] + exog_cols + ["hour", "day_of_week", "month",
                                                 "temp_cooling_interaction", "Predicted_kW"]]
    st.dataframe(out.head(50), use_container_width=True)
    st.download_button(
        label="Download forecast CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="forecast_hourly.csv",
        mime="text/csv",
    )
    # Chatbot


def _top_peak_hours(dataframe: pd.DataFrame, k: int = 3) -> str:
    g = dataframe.groupby("hour")["total_elec_power_drawn"].mean().sort_values(ascending=False)
    top = g.head(k).round(2)
    return ", ".join([f"{h:02d}h ({v} kW)" for h, v in top.items()])


def _temp_sensitivity(dataframe: pd.DataFrame) -> str:
    c = dataframe[["Weather_Station_Weather_Ta", "total_elec_power_drawn"]].corr().iloc[0, 1]
    return f"Correlation between temperature and load is {c:.2f} (positive means higher temperature → higher load)."


def _simulate_delta(pred_input_df: pd.DataFrame, col: str, pct: float = -0.10):
    base = float(model.predict(pred_input_df)[0])
    mod = pred_input_df.copy()
    mod[col] = mod[col] * (1.0 + pct)
    new = float(model.predict(mod)[0])
    delta = new - base
    rel = 100.0 * delta / base if base else 0.0
    return base, new, delta, rel


def _format_delta(col: str, pct: float, label: str) -> str:
    b, n, d, r = _simulate_delta(input_df, col, pct)
    r_disp = "≈0%" if abs(r) < 0.05 else f"{r:+.1f}%"
    return f"{label} changes predicted load from **{b:,.2f} kW** to **{n:,.2f} kW** (Δ {d:+,.2f} kW, {r_disp})."


def _avg_last_days(days: int = 7) -> str:
    s = df[df["datetime_utc"] >= (df["datetime_utc"].max() - pd.Timedelta(days=days))]["total_elec_power_drawn"].mean()
    return f"Average load over the last {days} days is **{s:,.2f} kW**."


FAQ = {
    "What is the predicted energy for the current inputs?":
        lambda: f"Predicted load is **{float(model.predict(input_df)[0]):,.2f} kW** for the selected hour and drivers.",
    "Which hours typically peak?":
        lambda: f"Historically highest average load occurs around: {_top_peak_hours(df)}.",
    "How sensitive is load to temperature?":
        lambda: _temp_sensitivity(df),
    "What if cooling power is reduced by 10% now?":
        lambda: _format_delta("total_cooling_pow", -0.10, "Reducing cooling power by 10%"),
    "What if PV output increases by 20% now?":
        lambda: _format_delta("PV", +0.20, "Increasing PV by 20%"),
    "How can energy be reduced during peak hours?":
        lambda: (
            "**Suggested actions**:\n\n"
            "- Shift noncritical loads away from the top-3 peak hours (see *Which hours typically peak?*).\n"
            "- Tighten cooling setpoint deadbands and use night pre-cooling when outdoor temperature is lower.\n"
            "- Schedule CHP to offset electrical peaks and increase PV utilisation if available.\n"
            "- Avoid simultaneous heating and cooling; verify valve/actuator scheduling.\n"
            "- Investigate anomalies (spikes in cooling power or unexpected nighttime loads).\n"
        ),
    "Show last 7 days average consumption":
        lambda: _avg_last_days(7),
}

if st.session_state.show_chat:
    with st.sidebar:
        st.header(" Energy Assistant")
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I’m your energy assistant. Choose a question below. "
                "Answers use the current inputs and model."
            )

        # Buttons for predefined questions
        for i, q in enumerate(FAQ.keys()):
            if st.button(q, key=f"faq_{i}"):
                st.session_state.chat.append({"role": "user", "content": q})
                ans = FAQ[q]()
                st.session_state.chat.append({"role": "assistant", "content": ans})
                st.rerun()

        # Render conversation
        for msg in st.session_state.chat:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])

        st.caption(" ")
        if st.button("Reset conversation"):
            st.session_state.chat = []
            st.rerun()
