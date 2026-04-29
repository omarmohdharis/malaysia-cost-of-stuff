"""
Harga Malaysia — Grocery Price Intelligence Dashboard
A public tool to help Malaysians understand grocery price trends across states.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Harga Malaysia",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,200,50,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
}
.hero-title span { color: #FFD166; }
.hero-subtitle {
    color: rgba(255,255,255,0.7);
    font-size: 1.1rem;
    margin-top: 0.75rem;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,209,102,0.15);
    border: 1px solid rgba(255,209,102,0.3);
    color: #FFD166;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #0f2027;
    line-height: 1;
}
.metric-label {
    color: #6b7280;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.4rem;
}
.metric-delta-up { color: #ef4444; font-size: 0.85rem; margin-top: 0.2rem; }
.metric-delta-down { color: #22c55e; font-size: 0.85rem; margin-top: 0.2rem; }

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #0f2027;
    margin-bottom: 0.25rem;
}
.section-sub {
    color: #6b7280;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* Rank table */
.rank-row {
    display: flex;
    align-items: center;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: #f9fafb;
    border: 1px solid #f3f4f6;
    transition: background 0.2s;
}
.rank-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #9ca3af;
    width: 2rem;
}
.rank-num.gold { color: #FFD166; }
.rank-num.silver { color: #94a3b8; }
.rank-num.bronze { color: #b45309; }
.rank-state { flex: 1; font-weight: 500; color: #1f2937; }
.rank-price {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #0f2027;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f2027;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: rgba(255,255,255,0.6) !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    background: #f3f4f6;
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    font-weight: 500;
    color: #6b7280;
}
.stTabs [aria-selected="true"] {
    background: #0f2027 !important;
    color: white !important;
}

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg, #0f2027, #203a43);
    border-radius: 12px;
    padding: 1.5rem;
    color: white;
    margin-bottom: 0.75rem;
}
.pred-state { font-size: 0.85rem; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 0.05em; }
.pred-price { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #FFD166; }
.pred-change-up { color: #f87171; font-size: 0.9rem; }
.pred-change-down { color: #4ade80; font-size: 0.9rem; }

/* Info box */
.info-box {
    background: #fffbeb;
    border-left: 3px solid #FFD166;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.875rem;
    color: #78350f;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
BASKET_CATEGORIES = [
    "AYAM", "TELUR", "BERAS", "MINYAK DAN LEMAK",
    "BAWANG", "SAYUR-SAYURAN", "BAHAN LAUT", "IKAN DARAT", "BUAH-BUAHAN"
]

MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]

MONTH_LABELS = {
    "2025-11": "Nov 2025", "2025-12": "Dec 2025",
    "2026-01": "Jan 2026", "2026-02": "Feb 2026",
    "2026-03": "Mar 2026", "2026-04": "Apr 2026",
}

STATE_COORDS = {
    "Johor": (1.9344, 103.4),
    "Kedah": (6.1184, 100.3685),
    "Kelantan": (5.7480, 102.0),
    "Melaka": (2.1896, 102.25),
    "Negeri Sembilan": (2.7258, 102.0),
    "Pahang": (3.8126, 103.3256),
    "Perak": (4.5921, 101.0901),
    "Perlis": (6.4449, 100.2048),
    "Pulau Pinang": (5.4141, 100.3288),
    "Sabah": (5.9788, 116.0753),
    "Sarawak": (1.5533, 110.3592),
    "Selangor": (3.0738, 101.5183),
    "Terengganu": (5.3117, 103.1324),
    "W.P. Kuala Lumpur": (3.1390, 101.6869),
    "W.P. Labuan": (5.2831, 115.2308),
    "W.P. Putrajaya": (2.9264, 101.6964),
}

@st.cache_data(show_spinner="Loading price data...")
def load_data():
    item    = pd.read_parquet("https://storage.data.gov.my/pricecatcher/lookup_item.parquet", dtype_backend="numpy_nullable")
    premise = pd.read_parquet("https://storage.data.gov.my/pricecatcher/lookup_premise.parquet", dtype_backend="numpy_nullable")
    item_clean    = item.dropna(subset=["item_code","item","unit","item_group","item_category"]).copy()
    premise_clean = premise.dropna(subset=["premise_code","premise","premise_type","state","district"]).copy()

    frames = []
    for month in MONTHS:
        url  = f"https://storage.data.gov.my/pricecatcher/pricecatcher_{month}.parquet"
        temp = pd.read_parquet(url, dtype_backend="numpy_nullable")
        temp["date"]  = pd.to_datetime(temp["date"])
        temp["month"] = temp["date"].dt.to_period("M").astype(str)
        frames.append(temp)

    price_all = pd.concat(frames, ignore_index=True)
    df = (
        price_all
        .merge(item_clean,    on="item_code",    how="left")
        .merge(premise_clean, on="premise_code", how="left")
    )

    df = df[df["price"] > 0].copy()
    lo, hi = df["price"].quantile(0.01), df["price"].quantile(0.99)
    df = df[(df["price"] >= lo) & (df["price"] <= hi)].copy()

    text_cols = ["item","unit","item_group","item_category","premise","premise_type","state","district"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    df_basket = df[df["item_category"].isin(BASKET_CATEGORIES)].copy()
    return df_basket

@st.cache_data(show_spinner="Crunching numbers...")
def build_model_df(_df_basket):
    df = _df_basket.copy()
    df["year"]      = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    item_monthly_avg = (
        df.groupby(["item_code","month"])["price"].mean()
        .reset_index().rename(columns={"price":"item_monthly_avg_price"})
        .sort_values(["item_code","month"])
    )
    item_monthly_avg["item_prev_month_price"] = (
        item_monthly_avg.groupby("item_code")["item_monthly_avg_price"].shift(1)
    )
    df = df.merge(item_monthly_avg[["item_code","month","item_prev_month_price"]],
                  on=["item_code","month"], how="left")

    item_vol = (df.groupby("item_code")["price"].std().reset_index()
                .rename(columns={"price":"item_price_volatility"}))
    df = df.merge(item_vol, on="item_code", how="left")

    premise_avg = (df.groupby("premise_code")["price"].mean().reset_index()
                   .rename(columns={"price":"premise_price_index"}))
    df = df.merge(premise_avg, on="premise_code", how="left")

    model_df = (
        df.groupby(["month","year","month_num","state","premise_type"], as_index=False)
        .agg(
            basket_price         =("price",                 "mean"),
            median_basket_price  =("price",                 "median"),
            price_std            =("price",                 "std"),
            num_records          =("price",                 "count"),
            num_items            =("item_code",             "nunique"),
            num_premises         =("premise_code",          "nunique"),
            avg_item_volatility  =("item_price_volatility", "mean"),
            avg_premise_index    =("premise_price_index",   "mean"),
            avg_prev_month_price =("item_prev_month_price", "mean"),
            month_sin            =("month_sin",             "first"),
            month_cos            =("month_cos",             "first"),
        )
        .sort_values(["state","premise_type","month"])
    )
    return model_df

@st.cache_resource(show_spinner="Training model...")
def train_model(_model_df):
    df = _model_df.copy()
    df = pd.get_dummies(df, columns=["state","premise_type"], drop_first=True)
    dummy_cols = [c for c in df.columns if c.startswith("state_") or c.startswith("premise_type_")]

    FEATURES = [
        "month_sin","month_cos","year",
        "avg_item_volatility","avg_premise_index","avg_prev_month_price",
        "num_items","num_premises",
    ] + dummy_cols

    df_clean = df.dropna(subset=["avg_prev_month_price"]).copy()
    df_clean = df_clean.sort_values("month")

    X = df_clean[FEATURES]
    y = df_clean["basket_price"]

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model, FEATURES, dummy_cols

def predict_next_month(_model, features, dummy_cols, model_df):
    """Generate May 2026 predictions for all state×premise_type combos."""
    latest = model_df[model_df["month"] == model_df["month"].max()].copy()
    next_month_num = 5  # May
    next_year      = 2026

    rows = []
    for _, row in latest.iterrows():
        r = {
            "state":        row["state"],
            "premise_type": row["premise_type"],
            "month":        "2026-05",
            "year":         next_year,
            "month_num":    next_month_num,
            "month_sin":    np.sin(2 * np.pi * next_month_num / 12),
            "month_cos":    np.cos(2 * np.pi * next_month_num / 12),
            "avg_item_volatility":    row["avg_item_volatility"],
            "avg_premise_index":      row["avg_premise_index"],
            "avg_prev_month_price":   row["basket_price"],
            "num_items":    row["num_items"],
            "num_premises": row["num_premises"],
        }
        rows.append(r)

    pred_df = pd.DataFrame(rows)
    pred_df = pd.get_dummies(pred_df, columns=["state","premise_type"], drop_first=False)

    # Align columns
    for col in dummy_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0
    pred_df = pred_df[[c for c in features if c in pred_df.columns]]
    for col in features:
        if col not in pred_df.columns:
            pred_df[col] = 0
    pred_df = pred_df[features]

    preds = _model.predict(pred_df)
    result = pd.DataFrame(rows)[["state","premise_type","avg_prev_month_price"]]
    result["predicted_price"] = preds
    result["change"]          = result["predicted_price"] - result["avg_prev_month_price"]
    result["change_pct"]      = result["change"] / result["avg_prev_month_price"] * 100
    return result

# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner("Loading Malaysia grocery data..."):
    df_basket = load_data()
    model_df  = build_model_df(df_basket)
    model, FEATURES, dummy_cols = train_model(model_df)
    predictions = predict_next_month(model, FEATURES, dummy_cols, model_df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.5rem; color: white;'>🛒 Harga<br>Malaysia</div>
        <div style='color: rgba(255,255,255,0.4); font-size: 0.75rem; margin-top: 0.25rem;'>Grocery Price Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:rgba(255,255,255,0.4);font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Filter by Store Type</div>", unsafe_allow_html=True)
    all_premise_types = sorted(df_basket["premise_type"].dropna().unique())
    selected_premise = st.multiselect(
        "", all_premise_types, default=all_premise_types, label_visibility="collapsed"
    )

    st.markdown("<div style='color:rgba(255,255,255,0.4);font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-top:1rem;margin-bottom:0.5rem'>Filter by State</div>", unsafe_allow_html=True)
    all_states = sorted(df_basket["state"].dropna().unique())
    selected_states = st.multiselect(
        "", all_states, default=all_states, label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='color:rgba(255,255,255,0.35);font-size:0.75rem;line-height:1.6'>Data from <b style='color:rgba(255,255,255,0.6)'>data.gov.my</b><br>PriceCatcher · Nov 2025 – Apr 2026<br>Model: Ridge Regression · MAPE 3.15%</div>", unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
filtered = df_basket[
    df_basket["premise_type"].isin(selected_premise) &
    df_basket["state"].isin(selected_states)
]

# ── Hero ──────────────────────────────────────────────────────────────────────
latest_month     = filtered["month"].max()
latest_label     = MONTH_LABELS.get(latest_month, latest_month)
national_avg     = filtered[filtered["month"] == latest_month]["price"].mean()
prev_month       = sorted(filtered["month"].unique())[-2] if len(filtered["month"].unique()) > 1 else latest_month
prev_avg         = filtered[filtered["month"] == prev_month]["price"].mean()
national_change  = ((national_avg - prev_avg) / prev_avg * 100) if prev_avg else 0

st.markdown(f"""
<div class="hero">
    <div class="hero-badge">Live Dashboard · {latest_label}</div>
    <div class="hero-title">Malaysian Grocery<br><span>Price Tracker</span></div>
    <div class="hero-subtitle">Helping Malaysians find the best prices across all 16 states & territories</div>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

state_latest = (
    filtered[filtered["month"] == latest_month]
    .groupby("state")["price"].mean()
)
cheapest_state    = state_latest.idxmin() if len(state_latest) else "—"
cheapest_price    = state_latest.min() if len(state_latest) else 0
most_exp_state    = state_latest.idxmax() if len(state_latest) else "—"
most_exp_price    = state_latest.max() if len(state_latest) else 0

arrow = "↑" if national_change > 0 else "↓"
delta_class = "metric-delta-up" if national_change > 0 else "metric-delta-down"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">RM {national_avg:.2f}</div>
        <div class="metric-label">National Avg Item Price</div>
        <div class="{delta_class}">{arrow} {abs(national_change):.1f}% vs last month</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#22c55e">RM {cheapest_price:.2f}</div>
        <div class="metric-label">Cheapest State</div>
        <div style="color:#6b7280;font-size:0.85rem;margin-top:0.2rem">{cheapest_state}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#ef4444">RM {most_exp_price:.2f}</div>
        <div class="metric-label">Most Expensive State</div>
        <div style="color:#6b7280;font-size:0.85rem;margin-top:0.2rem">{most_exp_state}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    spread = most_exp_price - cheapest_price
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">RM {spread:.2f}</div>
        <div class="metric-label">Price Spread</div>
        <div style="color:#6b7280;font-size:0.85rem;margin-top:0.2rem">Cheapest vs most expensive</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺 Current Prices",
    "📈 Price Trends",
    "🔮 Next Month Forecast",
    "🏆 Cheapest to Shop",
    "🛒 Basket Calculator",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Current Prices
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Current Basket Prices by State</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Average grocery item price across selected store types · {latest_label}</div>', unsafe_allow_html=True)

    col_map, col_rank = st.columns([3, 2])

    with col_map:
        from malaysia_geojson import MALAYSIA_GEOJSON

        state_avg = (
            filtered[filtered["month"] == latest_month]
            .groupby("state")["price"].mean()
            .reset_index()
            .rename(columns={"price": "avg_price"})
        )

        state_avg = state_avg.sort_values("avg_price").reset_index(drop=True)
        state_avg["rank"] = range(1, len(state_avg) + 1)
        n = len(state_avg)
        state_avg["rank_label"] = state_avg["rank"].map(
            lambda r: "🥇 Cheapest" if r == 1 else ("🔴 Most Expensive" if r == n else f"#{r}")
        )

        fig_map = px.choropleth_mapbox(
            state_avg,
            geojson=MALAYSIA_GEOJSON,
            locations="state",
            featureidkey="properties.state",
            color="avg_price",
            color_continuous_scale=["#4ade80", "#FFD166", "#ef4444"],
            range_color=(state_avg["avg_price"].min(), state_avg["avg_price"].max()),
            mapbox_style="carto-positron",
            zoom=4.8,
            center={"lat": 4.0, "lon": 109.5},
            opacity=0.85,
            hover_name="state",
            hover_data={
                "avg_price": ":.2f",
                "rank_label": True,
                "state": False,
            },
            labels={"avg_price": "Avg Price (RM)", "rank_label": "Rank"},
        )
        fig_map.update_layout(
            height=440,
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(
                title="RM",
                thickness=14,
                len=0.6,
                tickformat=".2f",
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_rank:
        st.markdown('<div style="font-weight:600;color:#0f2027;margin-bottom:0.75rem">Price Ranking</div>', unsafe_allow_html=True)
        state_ranked = state_avg.sort_values("avg_price")
        medals = {0: ("gold","🥇"), 1: ("silver","🥈"), 2: ("bronze","🥉")}

        for i, (_, row) in enumerate(state_ranked.iterrows()):
            medal_class, medal = medals.get(i, ("", ""))
            rank_num = f'<span class="rank-num {medal_class}">{medal if medal else str(i+1)}</span>'
            st.markdown(f"""
            <div class="rank-row">
                {rank_num}
                <span class="rank-state">{row['state']}</span>
                <span class="rank-price">RM {row['avg_price']:.2f}</span>
            </div>""", unsafe_allow_html=True)

    # Category breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.2rem">Price by Category</div>', unsafe_allow_html=True)

    cat_avg = (
        filtered[filtered["month"] == latest_month]
        .groupby("item_category")["price"].mean()
        .reset_index()
        .sort_values("price", ascending=True)
    )

    CATEGORY_LABELS = {
        "AYAM": "Chicken 🐔", "TELUR": "Eggs 🥚", "BERAS": "Rice 🍚",
        "MINYAK DAN LEMAK": "Cooking Oil 🫙", "BAWANG": "Onions 🧅",
        "SAYUR-SAYURAN": "Vegetables 🥬", "BAHAN LAUT": "Seafood 🐟",
        "IKAN DARAT": "Freshwater Fish 🐠", "BUAH-BUAHAN": "Fruits 🍎",
    }
    cat_avg["label"] = cat_avg["item_category"].map(lambda x: CATEGORY_LABELS.get(x, x))

    fig_cat = px.bar(
        cat_avg, x="price", y="label", orientation="h",
        labels={"price": "Average Price (RM)", "label": ""},
        color="price",
        color_continuous_scale=["#4ade80", "#FFD166", "#ef4444"],
    )
    fig_cat.update_layout(
        height=320, showlegend=False,
        margin=dict(l=0, r=20, t=10, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(showgrid=False),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Price Trends
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Price Trends Over Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Track how grocery prices have moved across states and store types</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        trend_states = st.multiselect(
            "Select states to compare",
            options=selected_states,
            default=selected_states[:5] if len(selected_states) >= 5 else selected_states,
        )
    with col_ctrl2:
        trend_breakdown = st.selectbox(
            "Break down by",
            ["State", "Store Type", "Category"],
        )

    if trend_breakdown == "State":
        trend_df = (
            filtered[filtered["state"].isin(trend_states)]
            .groupby(["month","state"])["price"].mean()
            .reset_index()
        )
        trend_df["month_label"] = trend_df["month"].map(MONTH_LABELS)
        fig_trend = px.line(
            trend_df, x="month_label", y="price", color="state",
            markers=True,
            labels={"price": "Avg Price (RM)", "month_label": "", "state": "State"},
        )

    elif trend_breakdown == "Store Type":
        trend_df = (
            filtered[filtered["state"].isin(trend_states)]
            .groupby(["month","premise_type"])["price"].mean()
            .reset_index()
        )
        trend_df["month_label"] = trend_df["month"].map(MONTH_LABELS)
        fig_trend = px.line(
            trend_df, x="month_label", y="price", color="premise_type",
            markers=True,
            labels={"price": "Avg Price (RM)", "month_label": "", "premise_type": "Store Type"},
        )

    else:
        trend_df = (
            filtered[filtered["state"].isin(trend_states)]
            .groupby(["month","item_category"])["price"].mean()
            .reset_index()
        )
        trend_df["month_label"] = trend_df["month"].map(MONTH_LABELS)
        trend_df["label"] = trend_df["item_category"].map(lambda x: CATEGORY_LABELS.get(x, x))
        fig_trend = px.line(
            trend_df, x="month_label", y="price", color="label",
            markers=True,
            labels={"price": "Avg Price (RM)", "month_label": "", "label": "Category"},
        )

    fig_trend.update_layout(
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="DM Sans"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", title="Average Price (RM)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    fig_trend.update_traces(line=dict(width=2.5), marker=dict(size=7))
    st.plotly_chart(fig_trend, use_container_width=True)

    # Month over month change heatmap
    st.markdown('<div class="section-header" style="font-size:1.2rem;margin-top:1rem">Month-over-Month Change (%)</div>', unsafe_allow_html=True)

    mom_df = (
        filtered.groupby(["month","state"])["price"].mean()
        .reset_index()
        .sort_values(["state","month"])
    )
    mom_df["pct_change"] = mom_df.groupby("state")["price"].pct_change() * 100
    mom_df["month_label"] = mom_df["month"].map(MONTH_LABELS)
    mom_pivot = mom_df.pivot(index="state", columns="month_label", values="pct_change").dropna(how="all")

    fig_heat = px.imshow(
        mom_pivot,
        color_continuous_scale=["#22c55e", "white", "#ef4444"],
        color_continuous_midpoint=0,
        aspect="auto",
        labels=dict(color="% Change"),
        text_auto=".1f",
    )
    fig_heat.update_layout(
        height=400,
        font=dict(family="DM Sans"),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="", yaxis_title="",
        coloraxis_colorbar=dict(title="%", thickness=12),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Forecast
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Next Month Price Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ridge Regression model predictions for May 2026 · MAPE 3.15%</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        📊 Predictions are based on 6 months of historical data across 16 states and 6 store types.
        The model has a mean absolute percentage error of <b>3.15%</b> — roughly RM 0.38 on a typical basket.
    </div>
    """, unsafe_allow_html=True)

    # Filter predictions
    pred_filtered = predictions[
        predictions["state"].isin(selected_states) &
        predictions["premise_type"].isin(selected_premise)
    ]

    # State-level aggregate predictions
    state_pred = (
        pred_filtered.groupby("state")
        .agg(
            predicted_price=("predicted_price", "mean"),
            current_price=("avg_prev_month_price", "mean"),
        )
        .reset_index()
    )
    state_pred["change"]     = state_pred["predicted_price"] - state_pred["current_price"]
    state_pred["change_pct"] = state_pred["change"] / state_pred["current_price"] * 100
    state_pred = state_pred.sort_values("predicted_price")

    col_f1, col_f2 = st.columns([3, 2])

    with col_f1:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Bar(
            x=state_pred["current_price"],
            y=state_pred["state"],
            orientation="h",
            name="Apr 2026 (Actual)",
            marker_color="#94a3b8",
            opacity=0.6,
        ))
        fig_pred.add_trace(go.Bar(
            x=state_pred["predicted_price"],
            y=state_pred["state"],
            orientation="h",
            name="May 2026 (Predicted)",
            marker_color="#FFD166",
        ))
        fig_pred.update_layout(
            barmode="overlay",
            height=480,
            font=dict(family="DM Sans"),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f3f4f6", title="Average Price (RM)"),
            yaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=20, t=40, b=0),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_f2:
        st.markdown('<div style="font-weight:600;color:#0f2027;margin-bottom:0.75rem">Expected Changes</div>', unsafe_allow_html=True)
        for _, row in state_pred.sort_values("change_pct").iterrows():
            arrow  = "↑" if row["change_pct"] > 0 else "↓"
            color  = "#ef4444" if row["change_pct"] > 0 else "#22c55e"
            st.markdown(f"""
            <div class="rank-row">
                <span class="rank-state">{row['state']}</span>
                <span style="font-weight:600;color:{color}">{arrow} {abs(row['change_pct']):.1f}%</span>
            </div>""", unsafe_allow_html=True)

    # Store type forecast
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.2rem">Forecast by Store Type</div>', unsafe_allow_html=True)

    premise_pred = (
        pred_filtered.groupby("premise_type")
        .agg(predicted_price=("predicted_price","mean"), current_price=("avg_prev_month_price","mean"))
        .reset_index()
    )
    premise_pred["change_pct"] = (premise_pred["predicted_price"] - premise_pred["current_price"]) / premise_pred["current_price"] * 100
    premise_pred = premise_pred.sort_values("predicted_price")

    fig_pt = px.bar(
        premise_pred, x="premise_type", y="predicted_price",
        color="change_pct",
        color_continuous_scale=["#22c55e","#FFD166","#ef4444"],
        color_continuous_midpoint=0,
        labels={"predicted_price":"Predicted Price (RM)","premise_type":"Store Type","change_pct":"% Change"},
        text=premise_pred["predicted_price"].map(lambda x: f"RM {x:.2f}"),
    )
    fig_pt.update_layout(
        height=300, font=dict(family="DM Sans"),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
        margin=dict(l=0,r=0,t=10,b=0),
        coloraxis_colorbar=dict(title="% Δ", thickness=12),
    )
    fig_pt.update_traces(textposition="outside")
    st.plotly_chart(fig_pt, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Cheapest to Shop
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Where to Shop Cheapest 🏆</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Find the best value state and store type combination for your grocery run</div>', unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        selected_cat = st.selectbox(
            "Filter by category",
            ["All Categories"] + [CATEGORY_LABELS.get(c,c) for c in BASKET_CATEGORIES],
        )

    # Build cheapest combos
    latest_df = filtered[filtered["month"] == latest_month].copy()
    if selected_cat != "All Categories":
        cat_key = {v: k for k, v in CATEGORY_LABELS.items()}.get(selected_cat, selected_cat)
        latest_df = latest_df[latest_df["item_category"] == cat_key]

    combo_df = (
        latest_df.groupby(["state","premise_type"])["price"].mean()
        .reset_index()
        .rename(columns={"price":"avg_price"})
        .sort_values("avg_price")
    )
    combo_df["rank"] = range(1, len(combo_df)+1)

    # Top 3 hero cards
    st.markdown("<br>", unsafe_allow_html=True)
    top3_cols = st.columns(3)
    medals_full = ["🥇 Best Value", "🥈 Runner Up", "🥉 Third Place"]
    for i, col in enumerate(top3_cols):
        if i < len(combo_df):
            row = combo_df.iloc[i]
            with col:
                st.markdown(f"""
                <div class="pred-card">
                    <div class="pred-state">{medals_full[i]}</div>
                    <div style="font-size:1.1rem;font-weight:600;margin:0.5rem 0 0.25rem;color:white">{row['state']}</div>
                    <div style="color:rgba(255,255,255,0.6);font-size:0.85rem">{row['premise_type']}</div>
                    <div class="pred-price">RM {row['avg_price']:.2f}</div>
                    <div style="color:rgba(255,255,255,0.5);font-size:0.75rem">avg item price</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full heatmap: state vs premise type
    st.markdown('<div class="section-header" style="font-size:1.2rem">Price Heatmap: State × Store Type</div>', unsafe_allow_html=True)

    heat_df = latest_df.groupby(["state","premise_type"])["price"].mean().reset_index()
    heat_pivot = heat_df.pivot(index="state", columns="premise_type", values="price")

    fig_ch = px.imshow(
        heat_pivot,
        color_continuous_scale=["#4ade80","#FFD166","#ef4444"],
        aspect="auto",
        labels=dict(color="Avg Price (RM)"),
        text_auto=".2f",
    )
    fig_ch.update_layout(
        height=500,
        font=dict(family="DM Sans"),
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis_title="Store Type", yaxis_title="",
        coloraxis_colorbar=dict(title="RM", thickness=12),
    )
    st.plotly_chart(fig_ch, use_container_width=True)

    # Full sortable table
    st.markdown('<div class="section-header" style="font-size:1.2rem;margin-top:1rem">Full Rankings</div>', unsafe_allow_html=True)
    display_df = combo_df[["rank","state","premise_type","avg_price"]].copy()
    display_df.columns = ["Rank","State","Store Type","Avg Price (RM)"]
    display_df["Avg Price (RM)"] = display_df["Avg Price (RM)"].map(lambda x: f"RM {x:.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — Basket Calculator
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🛒 Basket Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Build your weekly grocery list and see what it costs across every state and store type</div>', unsafe_allow_html=True)

    # ── Basket item definitions ───────────────────────────────────────
    BASKET_ITEMS = {
        "🐔 Chicken (Ayam Bersih Standard)":    {"item_code": 1,    "unit": "1kg",   "category": "AYAM"},
        "🥚 Eggs (Telur Ayam Grade A, 10 pcs)": {"item_code": None, "unit": "10 biji","category": "TELUR"},
        "🍚 Rice (Beras, 5kg)":                 {"item_code": None, "unit": "5kg",   "category": "BERAS"},
        "🫙 Cooking Oil (Minyak Masak, 1kg)":   {"item_code": None, "unit": "1kg",   "category": "MINYAK DAN LEMAK"},
        "🧅 Onion (Bawang Besar, 1kg)":         {"item_code": None, "unit": "1kg",   "category": "BAWANG"},
        "🥬 Vegetables (Sayur, 1kg)":            {"item_code": None, "unit": "1kg",   "category": "SAYUR-SAYURAN"},
        "🐟 Seafood (Ikan Kembung, 1kg)":        {"item_code": None, "unit": "1kg",   "category": "BAHAN LAUT"},
        "🐠 Freshwater Fish (Ikan Keli, 1kg)":   {"item_code": None, "unit": "1kg",   "category": "IKAN DARAT"},
        "🍎 Fruits (Buah-buahan, 1kg)":          {"item_code": None, "unit": "1kg",   "category": "BUAH-BUAHAN"},
    }

    # Get latest category-level prices per state and premise type
    latest_cat_prices = (
        filtered[filtered["month"] == latest_month]
        .groupby(["state", "premise_type", "item_category"])["price"]
        .mean()
        .reset_index()
        .rename(columns={"price": "unit_price"})
    )

    # ── Quantity inputs ───────────────────────────────────────────────
    st.markdown("### Step 1 — Set your quantities")
    st.markdown('<div class="section-sub">Adjust how much of each item you buy per week</div>', unsafe_allow_html=True)

    quantities = {}
    cols_per_row = 3
    items_list   = list(BASKET_ITEMS.items())

    for row_start in range(0, len(items_list), cols_per_row):
        row_items = items_list[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (label, meta) in zip(cols, row_items):
            with col:
                # Suggest a sensible default
                defaults = {
                    "AYAM": 1.0, "TELUR": 1.0, "BERAS": 0.5,
                    "MINYAK DAN LEMAK": 0.5, "BAWANG": 0.5,
                    "SAYUR-SAYURAN": 1.5, "BAHAN LAUT": 1.0,
                    "IKAN DARAT": 0.5, "BUAH-BUAHAN": 1.0,
                }
                default_qty = defaults.get(meta["category"], 1.0)
                qty = st.number_input(
                    label,
                    min_value=0.0,
                    max_value=20.0,
                    value=default_qty,
                    step=0.5,
                    key=f"qty_{meta['category']}",
                )
                quantities[meta["category"]] = qty

    # Remove zero-quantity items
    active_items = {cat: qty for cat, qty in quantities.items() if qty > 0}

    if not active_items:
        st.warning("Set at least one item quantity above zero to calculate.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Step 2 — Your basket total by state & store type")

        # ── Calculate basket cost ─────────────────────────────────────
        basket_rows = []
        for cat, qty in active_items.items():
            cat_prices = latest_cat_prices[latest_cat_prices["item_category"] == cat].copy()
            cat_prices["line_total"] = cat_prices["unit_price"] * qty
            cat_prices["category"]  = cat
            cat_prices["quantity"]  = qty
            basket_rows.append(cat_prices)

        if basket_rows:
            basket_long = pd.concat(basket_rows, ignore_index=True)
            basket_totals = (
                basket_long
                .groupby(["state", "premise_type"])
                .agg(
                    total_cost   =("line_total",  "sum"),
                    items_found  =("category",    "nunique"),
                )
                .reset_index()
                .sort_values("total_cost")
            )
            basket_totals["rank"] = range(1, len(basket_totals) + 1)

            # ── National summary metrics ──────────────────────────────
            cheapest_row   = basket_totals.iloc[0]
            expensive_row  = basket_totals.iloc[-1]
            national_median = basket_totals["total_cost"].median()
            savings         = expensive_row["total_cost"] - cheapest_row["total_cost"]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#22c55e">RM {cheapest_row['total_cost']:.2f}</div>
                    <div class="metric-label">Cheapest Basket</div>
                    <div style="color:#6b7280;font-size:0.82rem;margin-top:0.2rem">{cheapest_row['state']} · {cheapest_row['premise_type']}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#ef4444">RM {expensive_row['total_cost']:.2f}</div>
                    <div class="metric-label">Most Expensive</div>
                    <div style="color:#6b7280;font-size:0.82rem;margin-top:0.2rem">{expensive_row['state']} · {expensive_row['premise_type']}</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">RM {national_median:.2f}</div>
                    <div class="metric-label">National Median</div>
                    <div style="color:#6b7280;font-size:0.82rem;margin-top:0.2rem">Across all states & store types</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">RM {savings:.2f}</div>
                    <div class="metric-label">Potential Savings</div>
                    <div style="color:#6b7280;font-size:0.82rem;margin-top:0.2rem">Cheapest vs most expensive</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Top 3 best value cards ────────────────────────────────
            st.markdown("#### 🏆 Best Value Options for Your Basket")
            top3 = st.columns(3)
            medals_basket = ["🥇 Best Value", "🥈 Runner Up", "🥉 Third Place"]
            for i, col in enumerate(top3):
                if i < len(basket_totals):
                    row = basket_totals.iloc[i]
                    saving_vs_median = national_median - row["total_cost"]
                    with col:
                        st.markdown(f"""
                        <div class="pred-card">
                            <div class="pred-state">{medals_basket[i]}</div>
                            <div style="font-size:1.05rem;font-weight:600;margin:0.5rem 0 0.2rem;color:white">{row['state']}</div>
                            <div style="color:rgba(255,255,255,0.55);font-size:0.82rem">{row['premise_type']}</div>
                            <div class="pred-price">RM {row['total_cost']:.2f}</div>
                            <div class="pred-change-down">Save RM {saving_vs_median:.2f} vs median</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Bar chart: total basket cost by state ─────────────────
            st.markdown("#### Basket Cost by State (cheapest store type per state)")
            best_per_state = (
                basket_totals.sort_values("total_cost")
                .groupby("state")
                .first()
                .reset_index()
                .sort_values("total_cost")
            )

            fig_basket = px.bar(
                best_per_state,
                x="state",
                y="total_cost",
                color="total_cost",
                color_continuous_scale=["#4ade80", "#FFD166", "#ef4444"],
                labels={"total_cost": "Basket Total (RM)", "state": ""},
                text=best_per_state["total_cost"].map(lambda x: f"RM {x:.0f}"),
                hover_data={"premise_type": True, "total_cost": ":.2f"},
            )
            fig_basket.update_layout(
                height=380,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="DM Sans"),
                xaxis=dict(showgrid=False, tickangle=-35),
                yaxis=dict(showgrid=True, gridcolor="#f3f4f6", title="Total Basket Cost (RM)"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=20, b=80),
                showlegend=False,
            )
            fig_basket.update_traces(textposition="outside")
            st.plotly_chart(fig_basket, use_container_width=True)

            # ── Item breakdown for selected state ─────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Step 3 — Drill down by state")

            drill_state = st.selectbox(
                "Select a state to see item-level breakdown",
                options=sorted(basket_totals["state"].unique()),
                index=0,
            )

            drill_df = basket_long[basket_long["state"] == drill_state].copy()
            drill_premise = st.selectbox(
                "Select store type",
                options=sorted(drill_df["premise_type"].unique()),
            )

            drill_final = (
                drill_df[drill_df["premise_type"] == drill_premise]
                [["category", "quantity", "unit_price", "line_total"]]
                .copy()
            )
            drill_final["item"] = drill_final["category"].map(
                lambda c: next((k for k, v in BASKET_ITEMS.items() if v["category"] == c), c)
            )
            drill_final = drill_final[["item", "quantity", "unit_price", "line_total"]]
            drill_final.columns = ["Item", "Qty (kg/unit)", "Unit Price (RM)", "Line Total (RM)"]
            drill_final["Unit Price (RM)"] = drill_final["Unit Price (RM)"].map(lambda x: f"RM {x:.2f}")
            drill_final["Line Total (RM)"] = drill_final["Line Total (RM)"].map(lambda x: f"RM {x:.2f}")

            st.dataframe(drill_final, use_container_width=True, hide_index=True)

            total_for_combo = basket_totals[
                (basket_totals["state"] == drill_state) &
                (basket_totals["premise_type"] == drill_premise)
            ]["total_cost"].values

            if len(total_for_combo) > 0:
                total_val = total_for_combo[0]
                cheapest_val = cheapest_row["total_cost"]
                diff = total_val - cheapest_val
                diff_pct = diff / cheapest_val * 100
                color = "#ef4444" if diff > 0 else "#22c55e"
                arrow = "↑" if diff > 0 else "↓"
                st.markdown(f"""
                <div style="background:#f9fafb;border-radius:10px;padding:1rem 1.25rem;margin-top:0.5rem;display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div style="font-weight:600;color:#0f2027;font-size:1.05rem">{drill_state} · {drill_premise}</div>
                        <div style="color:#6b7280;font-size:0.85rem">vs cheapest option ({cheapest_row['state']} · {cheapest_row['premise_type']})</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#0f2027">RM {total_val:.2f}</div>
                        <div style="color:{color};font-size:0.9rem">{arrow} RM {abs(diff):.2f} ({abs(diff_pct):.1f}%) {'more' if diff > 0 else 'less'} expensive</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
