"""
Harga Malaysia — Grocery Price Intelligence Dashboard
A public tool to help Malaysians understand grocery price trends across states.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

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

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
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

CATEGORY_LABELS = {
    "AYAM": "Chicken 🐔", "TELUR": "Eggs 🥚", "BERAS": "Rice 🍚",
    "MINYAK DAN LEMAK": "Cooking Oil 🫙", "BAWANG": "Onions 🧅",
    "SAYUR-SAYURAN": "Vegetables 🥬", "BAHAN LAUT": "Seafood 🐟",
    "IKAN DARAT": "Freshwater Fish 🐠", "BUAH-BUAHAN": "Fruits 🍎",
}

BASKET_ITEMS = {
    "AYAM":             {"label": "🐔 Chicken",         "unit": "kg"},
    "TELUR":            {"label": "🥚 Eggs",            "unit": "× 10 pcs"},
    "BERAS":            {"label": "🍚 Rice",            "unit": "× 5kg bag"},
    "MINYAK DAN LEMAK": {"label": "🫙 Cooking Oil",     "unit": "× 1kg bottle"},
    "BAWANG":           {"label": "🧅 Onions",          "unit": "kg"},
    "SAYUR-SAYURAN":    {"label": "🥬 Vegetables",      "unit": "kg"},
    "BAHAN LAUT":       {"label": "🐟 Seafood",         "unit": "kg"},
    "IKAN DARAT":       {"label": "🐠 Freshwater Fish", "unit": "kg"},
    "BUAH-BUAHAN":      {"label": "🍎 Fruits",          "unit": "kg"},
}

PRESETS = {
    "🧍 Single (1 person)": {
        "AYAM": 0.5, "TELUR": 1.0, "BERAS": 0.2, "MINYAK DAN LEMAK": 0.25,
        "BAWANG": 0.25, "SAYUR-SAYURAN": 1.0, "BAHAN LAUT": 0.5,
        "IKAN DARAT": 0.0, "BUAH-BUAHAN": 1.0,
    },
    "👫 Couple (2 people)": {
        "AYAM": 1.0, "TELUR": 2.0, "BERAS": 0.5, "MINYAK DAN LEMAK": 0.5,
        "BAWANG": 0.5, "SAYUR-SAYURAN": 1.5, "BAHAN LAUT": 1.0,
        "IKAN DARAT": 0.5, "BUAH-BUAHAN": 1.5,
    },
    "👨‍👩‍👧 Small Family (3-4)": {
        "AYAM": 1.5, "TELUR": 3.0, "BERAS": 1.0, "MINYAK DAN LEMAK": 1.0,
        "BAWANG": 1.0, "SAYUR-SAYURAN": 2.5, "BAHAN LAUT": 1.5,
        "IKAN DARAT": 1.0, "BUAH-BUAHAN": 2.0,
    },
    "👨‍👩‍👧‍👦 Large Family (5+)": {
        "AYAM": 2.5, "TELUR": 5.0, "BERAS": 2.0, "MINYAK DAN LEMAK": 1.5,
        "BAWANG": 1.5, "SAYUR-SAYURAN": 4.0, "BAHAN LAUT": 2.5,
        "IKAN DARAT": 1.5, "BUAH-BUAHAN": 3.0,
    },
    "✏️ Custom": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA & MODEL FUNCTIONS  (all top-level, all cached)
# ─────────────────────────────────────────────────────────────────────────────
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


@st.cache_data(show_spinner=False)
def get_latest_category_prices(_df, month):
    """Cached lookup of category-level prices per state×premise_type for a given month."""
    return (
        _df[_df["month"] == month]
        .groupby(["state", "premise_type", "item_category"])["price"]
        .mean()
        .reset_index()
        .rename(columns={"price": "unit_price"})
    )


@st.cache_data(show_spinner=False)
def compute_basket_totals(_cat_prices, quantities_tuple):
    """Compute basket totals per state×premise_type. quantities_tuple is hashable."""
    quantities = dict(quantities_tuple)
    active = {cat: qty for cat, qty in quantities.items() if qty > 0}
    if not active:
        return None, None

    rows = []
    for cat, qty in active.items():
        cp = _cat_prices[_cat_prices["item_category"] == cat].copy()
        cp["line_total"] = cp["unit_price"] * qty
        cp["category"] = cat
        cp["quantity"] = qty
        rows.append(cp)

    basket_long = pd.concat(rows, ignore_index=True)
    basket_totals = (
        basket_long
        .groupby(["state", "premise_type"])
        .agg(
            total_cost=("line_total", "sum"),
            items_found=("category", "nunique"),
        )
        .reset_index()
        .sort_values("total_cost")
    )
    basket_totals["rank"] = range(1, len(basket_totals) + 1)
    return basket_long, basket_totals


def predict_next_month(_model, features, dummy_cols, model_df):
    """Generate May 2026 predictions for all state×premise_type combos."""
    latest = model_df[model_df["month"] == model_df["month"].max()].copy()
    next_month_num = 5
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


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
df_basket = load_data()
model_df  = build_model_df(df_basket)
model, FEATURES, dummy_cols = train_model(model_df)
predictions = predict_next_month(model, FEATURES, dummy_cols, model_df)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
filtered = df_basket[
    df_basket["premise_type"].isin(selected_premise) &
    df_basket["state"].isin(selected_states)
]

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
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
            coloraxis_colorbar=dict(title="RM", thickness=14, len=0.6, tickformat=".2f"),
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.2rem">Price by Category</div>', unsafe_allow_html=True)

    cat_avg = (
        filtered[filtered["month"] == latest_month]
        .groupby("item_category")["price"].mean()
        .reset_index()
        .sort_values("price", ascending=True)
    )
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
        plot_bgcolor="white", paper_bgcolor="white",
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

    pred_filtered = predictions[
        predictions["state"].isin(selected_states) &
        predictions["premise_type"].isin(selected_premise)
    ]

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
            arrow_pred  = "↑" if row["change_pct"] > 0 else "↓"
            color_pred  = "#ef4444" if row["change_pct"] > 0 else "#22c55e"
            st.markdown(f"""
            <div class="rank-row">
                <span class="rank-state">{row['state']}</span>
                <span style="font-weight:600;color:{color_pred}">{arrow_pred} {abs(row['change_pct']):.1f}%</span>
            </div>""", unsafe_allow_html=True)

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

    col_c1, _ = st.columns(2)
    with col_c1:
        selected_cat = st.selectbox(
            "Filter by category",
            ["All Categories"] + [CATEGORY_LABELS.get(c,c) for c in BASKET_CATEGORIES],
        )

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

    st.markdown('<div class="section-header" style="font-size:1.2rem;margin-top:1rem">Full Rankings</div>', unsafe_allow_html=True)
    display_df = combo_df[["rank","state","premise_type","avg_price"]].copy()
    display_df.columns = ["Rank","State","Store Type","Avg Price (RM)"]
    display_df["Avg Price (RM)"] = display_df["Avg Price (RM)"].map(lambda x: f"RM {x:.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — Basket Calculator (wrapped in a fragment for fast updates)
# ═══════════════════════════════════════════════════════════════════
@st.fragment
def basket_calculator_section(filtered_df, current_month):
    st.markdown('<div class="section-header">🛒 Basket Calculator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Estimate your weekly grocery cost across every state and store type — '
        'pick a household preset to get started, then fine-tune.</div>',
        unsafe_allow_html=True,
    )

    # Initialise session state
    if "basket_qty" not in st.session_state:
        st.session_state.basket_qty = dict(PRESETS["👫 Couple (2 people)"])
        st.session_state.last_preset = "👫 Couple (2 people)"

    # ── Step 1: pick preset ──────────────────────────────────────────────────
    st.markdown("### 1️⃣ Pick your household")
    preset_cols = st.columns(len(PRESETS))
    for i, (preset_name, preset_qtys) in enumerate(PRESETS.items()):
        with preset_cols[i]:
            is_active = st.session_state.last_preset == preset_name
            btn_label = f"**{preset_name}**" if is_active else preset_name
            if st.button(btn_label, use_container_width=True, key=f"preset_{i}"):
                if preset_qtys is not None:
                    st.session_state.basket_qty = dict(preset_qtys)
                    # Clear slider widget state so they snap to new preset
                    for cat in BASKET_ITEMS:
                        slider_key = f"slider_{cat}"
                        if slider_key in st.session_state:
                            del st.session_state[slider_key]
                st.session_state.last_preset = preset_name
                st.rerun(scope="fragment")

    st.markdown(
        f'<div style="color:#6b7280;font-size:0.85rem;margin-top:0.5rem">'
        f'Selected: <b style="color:#0f2027">{st.session_state.last_preset}</b> — '
        f'sliders below show suggested weekly amounts. Adjust as needed.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step 2: sliders ──────────────────────────────────────────────────────
    st.markdown("### 2️⃣ Fine-tune your basket")

    cols_per_row = 3
    items_list = list(BASKET_ITEMS.items())
    for row_start in range(0, len(items_list), cols_per_row):
        row_items = items_list[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (cat, meta) in zip(cols, row_items):
            with col:
                default_value = float(st.session_state.basket_qty.get(cat, 0.0))
                qty = st.slider(
                    f"{meta['label']}  ·  {meta['unit']}",
                    min_value=0.0,
                    max_value=10.0,
                    value=default_value,
                    step=0.25,
                    key=f"slider_{cat}",
                )
                st.session_state.basket_qty[cat] = qty

    # Detect drift from preset
    if st.session_state.last_preset != "✏️ Custom":
        active_preset = PRESETS[st.session_state.last_preset]
        if active_preset is not None:
            drift = any(
                abs(st.session_state.basket_qty.get(k, 0.0) - v) > 1e-6
                for k, v in active_preset.items()
            )
            if drift:
                st.session_state.last_preset = "✏️ Custom"

    quantities = st.session_state.basket_qty
    active_items = {cat: qty for cat, qty in quantities.items() if qty > 0}

    if not active_items:
        st.warning("Add at least one item to your basket to see prices.")
        return

    # ── Compute basket totals (cached) ───────────────────────────────────────
    latest_cat_prices = get_latest_category_prices(filtered_df, current_month)
    qty_tuple = tuple(sorted(active_items.items()))
    basket_long, basket_totals = compute_basket_totals(latest_cat_prices, qty_tuple)

    if basket_totals is None or basket_totals.empty:
        st.warning("No price data available for the selected filters.")
        return

    cheapest_row     = basket_totals.iloc[0]
    expensive_row    = basket_totals.iloc[-1]
    national_median  = basket_totals["total_cost"].median()
    savings          = expensive_row["total_cost"] - cheapest_row["total_cost"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 3️⃣ Your basket — where to shop")

    # ── Headline metrics ─────────────────────────────────────────────────────
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
            <div class="metric-value">RM {national_median:.2f}</div>
            <div class="metric-label">National Median</div>
            <div style="color:#6b7280;font-size:0.82rem;margin-top:0.2rem">Across all combos</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#ef4444">RM {expensive_row['total_cost']:.2f}</div>
            <div class="metric-label">Most Expensive</div>
            <div style="color:#6b7280;font-size:0.82rem;margin-top:0.2rem">{expensive_row['state']} · {expensive_row['premise_type']}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">RM {savings:.2f}</div>
            <div class="metric-label">Max Savings</div>
            <div style="color:#6b7280;font-size:0.82rem;margin-top:0.2rem">Cheapest vs most expensive</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top 3 cards ──────────────────────────────────────────────────────────
    st.markdown("#### 🏆 Best value options")
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

    # ── Bar chart by state ───────────────────────────────────────────────────
    st.markdown("#### Basket cost by state")
    st.markdown(
        '<div style="color:#6b7280;font-size:0.85rem;margin-bottom:0.75rem">'
        'Each bar shows the cheapest store type available in that state.'
        '</div>',
        unsafe_allow_html=True,
    )
    best_per_state = (
        basket_totals.sort_values("total_cost")
        .groupby("state")
        .first()
        .reset_index()
        .sort_values("total_cost")
    )

    fig_basket = px.bar(
        best_per_state,
        x="state", y="total_cost",
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

    # ── Drill-down ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔍 See all store options in a state")

    drill_state = st.selectbox(
        "Pick a state",
        options=sorted(basket_totals["state"].unique()),
        index=0,
        key="drill_state_v2",
    )

    state_combos = basket_totals[basket_totals["state"] == drill_state].sort_values("total_cost")
    cheapest_overall = cheapest_row["total_cost"]

    st.markdown(
        f'<div style="color:#6b7280;font-size:0.85rem;margin-bottom:0.75rem">'
        f'All store types in <b>{drill_state}</b>, ranked cheapest to most expensive. '
        f'Compared against the cheapest option nationwide ({cheapest_row["state"]} · {cheapest_row["premise_type"]}).'
        f'</div>',
        unsafe_allow_html=True,
    )

    for _, combo in state_combos.iterrows():
        diff = combo["total_cost"] - cheapest_overall
        diff_pct = diff / cheapest_overall * 100 if cheapest_overall > 0 else 0
        if diff < 0.01:
            badge = '<span style="background:#dcfce7;color:#166534;padding:0.2rem 0.6rem;border-radius:12px;font-size:0.75rem;font-weight:600">CHEAPEST NATIONWIDE</span>'
        else:
            badge = f'<span style="color:#ef4444;font-size:0.85rem">+RM {diff:.2f} ({diff_pct:.1f}%)</span>'
        st.markdown(f"""
        <div class="rank-row">
            <span class="rank-state">{combo['premise_type']}</span>
            <span class="rank-price" style="margin-right:1rem">RM {combo['total_cost']:.2f}</span>
            {badge}
        </div>""", unsafe_allow_html=True)

    # ── Item-level breakdown ─────────────────────────────────────────────────
    with st.expander(f"📋 Item-by-item breakdown for {drill_state}"):
        drill_premise = st.selectbox(
            "Store type",
            options=sorted(state_combos["premise_type"].unique()),
            key="drill_premise_v2",
        )
        drill_df = basket_long[
            (basket_long["state"] == drill_state) &
            (basket_long["premise_type"] == drill_premise)
        ].copy()

        drill_df["item"] = drill_df["category"].map(lambda c: BASKET_ITEMS.get(c, {}).get("label", c))
        drill_df["unit"] = drill_df["category"].map(lambda c: BASKET_ITEMS.get(c, {}).get("unit", ""))
        drill_display = drill_df[["item", "quantity", "unit", "unit_price", "line_total"]].copy()
        drill_display.columns = ["Item", "Qty", "Unit", "Unit Price (RM)", "Line Total (RM)"]
        drill_display["Unit Price (RM)"] = drill_display["Unit Price (RM)"].map(lambda x: f"RM {x:.2f}")
        drill_display["Line Total (RM)"] = drill_display["Line Total (RM)"].map(lambda x: f"RM {x:.2f}")
        st.dataframe(drill_display, use_container_width=True, hide_index=True)


# Run the fragment inside tab 5
with tab5:
    basket_calculator_section(filtered, latest_month)