"""
Patut ke harga ni? · Is this price fair?
v1 — adds income context strip via DOSM constituency-level income data.

Differentiator: instead of just showing 'this price is X% above median', we say
'this represents Y% of a typical week's income in your constituency'.
That's affordability information, not just price comparison.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import warnings

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patut ke harga ni?",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 640px; }

.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem; color: #0f2027; line-height: 1.1; margin-bottom: 0.25rem;
}
.app-subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.75rem; }

.q-label { font-size: 0.95rem; font-weight: 500; color: #0f2027; margin-bottom: 0.25rem; }
.q-sub   { color: #6b7280; font-size: 0.82rem; margin-bottom: 0.6rem; }

/* Verdict cards */
.verdict-card { border-radius: 14px; padding: 1.5rem; margin-top: 1.25rem; margin-bottom: 1rem; }
.verdict-fair { background: #ecfdf5; border: 1px solid #a7f3d0; }
.verdict-good { background: #ecfdf5; border: 1px solid #6ee7b7; }
.verdict-warn { background: #fffbeb; border: 1px solid #fde68a; }
.verdict-bad  { background: #fef2f2; border: 1px solid #fecaca; }

.verdict-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
.dot-fair { background: #10b981; } .dot-good { background: #059669; }
.dot-warn { background: #d97706; } .dot-bad  { background: #dc2626; }

.verdict-headline { font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem; }
.headline-fair { color: #065f46; } .headline-good { color: #064e3b; }
.headline-warn { color: #78350f; } .headline-bad  { color: #7f1d1d; }

.verdict-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
.vg-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; font-weight: 500; opacity: 0.8; }
.vg-value { font-family: 'DM Serif Display', serif; font-size: 1.5rem; line-height: 1; }

/* Context strip — the differentiator */
.income-strip {
    background: #fffbeb;
    border-left: 4px solid #FFD166;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.15rem;
    margin-bottom: 1.25rem;
    color: #78350f;
    line-height: 1.55;
}
.income-strip-label {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #92400e; margin-bottom: 6px;
}
.income-strip-text { font-size: 0.92rem; color: #78350f; }
.income-strip-text b { color: #451a03; font-weight: 600; }

/* Plain context (when no constituency selected) */
.context-strip {
    background: #f9fafb;
    border-left: 3px solid #0f2027;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: #374151;
    line-height: 1.5;
}

.empty-state { text-align: center; padding: 2rem 1rem; color: #9ca3af; font-size: 0.9rem; }

/* Constituency picker styling */
.location-row {
    background: #f9fafb; border-radius: 10px;
    padding: 0.85rem 1rem; margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]
CONSUMER_CATEGORIES = [
    "AYAM", "TELUR", "BERAS", "MINYAK DAN LEMAK",
    "BAWANG", "SAYUR-SAYURAN", "BAHAN LAUT", "IKAN DARAT",
    "BUAH-BUAHAN", "GULA", "TEPUNG", "SUSU", "ROTI",
]
INCOME_URL = "https://storage.dosm.gov.my/hies/hh_income_parlimen.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading price data...", ttl=86400)
def load_price_data():
    item    = pd.read_parquet("https://storage.data.gov.my/pricecatcher/lookup_item.parquet", dtype_backend="numpy_nullable")
    premise = pd.read_parquet("https://storage.data.gov.my/pricecatcher/lookup_premise.parquet", dtype_backend="numpy_nullable")
    item_clean    = item.dropna(subset=["item_code","item","unit","item_category"]).copy()
    premise_clean = premise.dropna(subset=["premise_code","premise","premise_type","state"]).copy()

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
    df["item_code"] = df["item_code"].astype(int)
    text_cols = ["item","unit","item_category","premise_type","state"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df = df[df["item_category"].isin(CONSUMER_CATEGORIES)].copy()
    return df

@st.cache_data(show_spinner="Loading income data...", ttl=86400)
def load_income_data():
    """
    Loads parliament-level household income data from DOSM.
    Returns a clean DataFrame with: state, parlimen, year, income_median, income_mean.
    Returns None if loading fails (app will gracefully degrade to no income context).
    """
    try:
        df = pd.read_parquet(INCOME_URL)

        # Defensive: handle column name variations
        rename_map = {}
        cols_lower = {c.lower(): c for c in df.columns}
        for want in ["state", "parlimen", "date", "income_median", "income_mean"]:
            if want in cols_lower:
                rename_map[cols_lower[want]] = want
        df = df.rename(columns=rename_map)

        # Verify required columns exist
        required = ["state", "parlimen", "date", "income_median"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.warning(f"Income data missing columns: {missing}. Got: {df.columns.tolist()}")
            return None

        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        # Keep latest year per constituency (income is annual, multiple years available)
        df = (
            df.sort_values(["parlimen", "year"])
            .groupby("parlimen", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

        # Clean string columns
        for col in ["state", "parlimen"]:
            df[col] = df[col].astype(str).str.strip()

        return df[["state", "parlimen", "year", "income_median"] +
                  (["income_mean"] if "income_mean" in df.columns else [])]
    except Exception as e:
        st.info(
            f"💡 Couldn't load constituency income data ({type(e).__name__}). "
            f"App will work without the income context feature. "
            f"This often resolves on retry — try refreshing."
        )
        return None

@st.cache_data(show_spinner=False)
def get_item_list(_df):
    items = (
        _df.groupby(["item_code", "item", "unit"])
        .size().reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    items["display"] = items["item"] + "  ·  " + items["unit"]
    return items

@st.cache_data(show_spinner=False)
def get_state_list(_df):
    return sorted(_df["state"].dropna().unique())

@st.cache_data(show_spinner=False)
def get_constituencies_for_state(_income_df, state):
    """Return list of constituencies in the selected state, sorted alphabetically."""
    if _income_df is None:
        return []
    # Match state names — income data and price data should both use DOSM standard names
    # but handle "W.P. Kuala Lumpur" vs "Kuala Lumpur" type mismatches
    matches = _income_df[_income_df["state"].str.contains(state.replace("W.P. ", ""), case=False, na=False)]
    if len(matches) == 0:
        matches = _income_df[_income_df["state"] == state]
    return sorted(matches["parlimen"].unique().tolist())

@st.cache_data(show_spinner=False)
def get_item_state_stats(_df, item_code, state):
    latest_month = _df["month"].max()
    subset = _df[
        (_df["item_code"] == item_code) &
        (_df["state"] == state) &
        (_df["month"] == latest_month)
    ]
    if len(subset) == 0:
        return None
    return {
        "n": len(subset),
        "median": float(subset["price"].median()),
        "mean":   float(subset["price"].mean()),
        "p25":    float(subset["price"].quantile(0.25)),
        "p75":    float(subset["price"].quantile(0.75)),
        "p10":    float(subset["price"].quantile(0.10)),
        "p90":    float(subset["price"].quantile(0.90)),
        "min":    float(subset["price"].min()),
        "max":    float(subset["price"].max()),
        "month":  latest_month,
    }

@st.cache_data(show_spinner=False)
def get_item_history(_df, item_code, state):
    subset = _df[(_df["item_code"] == item_code) & (_df["state"] == state)]
    if len(subset) == 0:
        return None
    return (
        subset.groupby("month")["price"]
        .median()
        .reset_index()
        .rename(columns={"price": "median_price"})
        .sort_values("month")
    )

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def compute_verdict(price_paid, stats):
    median = stats["median"]
    p25, p75 = stats["p25"], stats["p75"]
    p10, p90 = stats["p10"], stats["p90"]
    pct_diff = (price_paid - median) / median * 100

    if price_paid <= p10:
        return {"tier": "good",
                "headline": "Bagus! Harga terbaik · Great deal!",
                "explanation": f"You're paying {abs(pct_diff):.0f}% below the typical price. "
                               f"Only ~10% of shops sell it this cheap.",
                "pct_diff": pct_diff}
    elif price_paid < p25:
        return {"tier": "good",
                "headline": "Bagus! Bawah purata · Below average",
                "explanation": f"You paid less than the typical range — about {abs(pct_diff):.0f}% below median. "
                               f"Worth remembering this shop.",
                "pct_diff": pct_diff}
    elif price_paid <= p75:
        return {"tier": "fair",
                "headline": "Harga berpatutan · Fair price",
                "explanation": f"This is within the normal price range for your state. "
                               f"Median is RM {median:.2f}; most shops charge RM {p25:.2f}–{p75:.2f}.",
                "pct_diff": pct_diff}
    elif price_paid <= p90:
        return {"tier": "warn",
                "headline": "Sedikit mahal · A bit expensive",
                "explanation": f"You paid about {pct_diff:.0f}% above the state median. "
                               f"Cheaper options exist — most shops charge under RM {p75:.2f}.",
                "pct_diff": pct_diff}
    else:
        return {"tier": "bad",
                "headline": "Sangat mahal — cuba elsewhere · Overpriced",
                "explanation": f"You paid {pct_diff:.0f}% above median. "
                               f"This is in the top 10% of prices — shop around for a much better deal.",
                "pct_diff": pct_diff}

# ─────────────────────────────────────────────────────────────────────────────
# INCOME CONTEXT LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def build_income_context(price_paid, monthly_income, parlimen_name, year):
    """
    Build the affordability strip:
    - 'In your area, typical household earns RM X/month'
    - 'This item is Y% of a week's income'
    - Comparison vs national B40 / M40 / T20 brackets if relevant
    """
    if monthly_income is None or monthly_income <= 0:
        return None

    weekly_income = monthly_income / 4.33   # avg weeks per month
    pct_of_week   = price_paid / weekly_income * 100
    pct_of_month  = price_paid / monthly_income * 100

    # B40/M40/T20 thresholds (DOSM 2022 — these are indicative, update with newer if available)
    # Source: https://www.dosm.gov.my/portal-main/release-content/household-income-survey-report-malaysia-2022
    if monthly_income < 5249:
        bracket = "B40"
        bracket_desc = "lower-income (B40) household"
    elif monthly_income < 11819:
        bracket = "M40"
        bracket_desc = "middle-income (M40) household"
    else:
        bracket = "T20"
        bracket_desc = "higher-income (T20) household"

    # Frame the affordability message based on % of week
    if pct_of_week < 0.5:
        frame = "small share of"
    elif pct_of_week < 1.5:
        frame = "modest share of"
    elif pct_of_week < 3:
        frame = "noticeable chunk of"
    else:
        frame = "significant share of"

    return {
        "monthly_income": monthly_income,
        "weekly_income":  weekly_income,
        "pct_of_week":    pct_of_week,
        "pct_of_month":   pct_of_month,
        "bracket":        bracket,
        "bracket_desc":   bracket_desc,
        "frame":          frame,
        "parlimen":       parlimen_name,
        "year":           year,
    }

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">Patut ke harga ni?</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Is this price fair? Check any grocery item against typical Malaysian prices — '
            'with affordability context for your area.</div>', unsafe_allow_html=True)

# Load data
df = load_price_data()
items_df = get_item_list(df)
states   = get_state_list(df)
income_df = load_income_data()

# ── Q1: Item ──────────────────────────────────────────────────────────────────
st.markdown('<div class="q-label">1. What did you buy?</div>', unsafe_allow_html=True)
st.markdown('<div class="q-sub">Apa yang anda beli?</div>', unsafe_allow_html=True)
item_options    = items_df["display"].tolist()
item_code_map   = dict(zip(items_df["display"], items_df["item_code"]))
selected_item   = st.selectbox(
    "Search for an item",
    options=item_options, index=None,
    placeholder="Type to search · e.g. ayam, telur, beras",
    label_visibility="collapsed",
)

# ── Q2: Location (state + optional constituency) ──────────────────────────────
st.markdown('<div class="q-label" style="margin-top:1rem">2. Where are you?</div>', unsafe_allow_html=True)
st.markdown('<div class="q-sub">Anda di mana? (Constituency is optional but unlocks affordability context)</div>', unsafe_allow_html=True)

col_state, col_parlimen = st.columns([1, 2])

with col_state:
    default_state_idx = states.index("Selangor") if "Selangor" in states else 0
    selected_state = st.selectbox(
        "State", options=states, index=default_state_idx,
        label_visibility="collapsed",
    )

with col_parlimen:
    constituencies = get_constituencies_for_state(income_df, selected_state)
    if income_df is not None and len(constituencies) > 0:
        # "Skip" option lets users use the app without picking a constituency
        parlimen_options = ["— Skip (no income context) —"] + constituencies
        selected_parlimen = st.selectbox(
            "Parliamentary constituency",
            options=parlimen_options,
            index=0,
            label_visibility="collapsed",
        )
        if selected_parlimen == parlimen_options[0]:
            selected_parlimen = None
    else:
        st.text_input("Constituency", value="(income data unavailable)", disabled=True, label_visibility="collapsed")
        selected_parlimen = None

# ── Q3: Price ─────────────────────────────────────────────────────────────────
st.markdown('<div class="q-label" style="margin-top:1rem">3. How much did you pay?</div>', unsafe_allow_html=True)
st.markdown('<div class="q-sub">Berapa anda bayar? (in RM)</div>', unsafe_allow_html=True)
price_paid = st.number_input(
    "Price paid",
    min_value=0.0, max_value=1000.0, value=0.0, step=0.10, format="%.2f",
    label_visibility="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────
if not selected_item:
    st.markdown('<div class="empty-state">↑ Pick an item above to get started</div>', unsafe_allow_html=True)
elif price_paid <= 0:
    st.markdown(
        f'<div class="empty-state">Enter the price you paid for <b>{selected_item}</b> to see the verdict</div>',
        unsafe_allow_html=True,
    )
else:
    item_code = item_code_map[selected_item]
    stats = get_item_state_stats(df, item_code, selected_state)

    if stats is None or stats["n"] < 3:
        st.warning(
            f"⚠️ Not enough recent price data for **{selected_item}** in {selected_state} "
            f"to give a reliable verdict. Try a different state, or check back later."
        )
    else:
        verdict = compute_verdict(price_paid, stats)
        tier = verdict["tier"]

        # ── Verdict card ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="verdict-card verdict-{tier}">
            <div class="verdict-headline headline-{tier}">
                <span class="verdict-dot dot-{tier}"></span>{verdict['headline']}
            </div>
            <div class="verdict-grid">
                <div>
                    <div class="vg-label headline-{tier}">You paid</div>
                    <div class="vg-value headline-{tier}">RM {price_paid:.2f}</div>
                </div>
                <div>
                    <div class="vg-label headline-{tier}">State median</div>
                    <div class="vg-value headline-{tier}">RM {stats['median']:.2f}</div>
                </div>
                <div>
                    <div class="vg-label headline-{tier}">Difference</div>
                    <div class="vg-value headline-{tier}">{verdict['pct_diff']:+.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Income context strip (THE differentiator) ────────────────────────
        if selected_parlimen is not None and income_df is not None:
            row = income_df[income_df["parlimen"] == selected_parlimen]
            if len(row) > 0:
                monthly_income = float(row["income_median"].iloc[0])
                year = int(row["year"].iloc[0])
                ctx = build_income_context(price_paid, monthly_income, selected_parlimen, year)
                if ctx:
                    st.markdown(f"""
                    <div class="income-strip">
                        <div class="income-strip-label">💡 Affordability in your area · {ctx['parlimen']}</div>
                        <div class="income-strip-text">
                            The typical household here earns about <b>RM {ctx['monthly_income']:,.0f}/month</b>
                            ({ctx['bracket']} bracket — {ctx['bracket_desc']}).
                            This item is a {ctx['frame']} a week's income —
                            roughly <b>{ctx['pct_of_week']:.1f}%</b> of typical weekly earnings here.
                            <br><span style="font-size:0.78rem;opacity:0.75">
                            Based on DOSM Household Income Survey {ctx['year']}.
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        elif income_df is not None:
            # Gentle nudge to pick a constituency
            st.markdown(f"""
            <div class="context-strip">
                💡 <b>Tip:</b> Pick your parliamentary constituency above to see how this price compares
                to typical household income in your area.
            </div>
            """, unsafe_allow_html=True)

        # ── Plain-language verdict context ────────────────────────────────────
        st.markdown(f"""
        <div class="context-strip">
            {verdict['explanation']}<br>
            <span style="color:#6b7280;font-size:0.82rem">
                Based on {stats['n']} price samples across {selected_state} in {stats['month']}.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Distribution chart ────────────────────────────────────────────────
        with st.expander("📊 Show price distribution in your state"):
            fig = go.Figure()
            fig.add_trace(go.Box(
                x=[stats["p10"], stats["p25"], stats["median"], stats["p75"], stats["p90"]],
                name="State price range", boxpoints=False,
                marker_color="#94a3b8", line_color="#64748b",
                orientation="h", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[price_paid], y=["State price range"],
                mode="markers+text",
                marker=dict(size=18, color={
                    "good": "#10b981", "fair": "#10b981",
                    "warn": "#d97706", "bad":  "#dc2626",
                }[tier], symbol="diamond", line=dict(width=2, color="white")),
                text=["You"], textposition="top center",
                textfont=dict(size=12, color="#0f2027"), showlegend=False,
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="DM Sans"),
                xaxis=dict(title="Price (RM)", showgrid=True, gridcolor="#f3f4f6", tickformat=".2f"),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"The box shows the middle 50% of prices (RM {stats['p25']:.2f}–{stats['p75']:.2f}). "
                f"Whiskers cover the 10th–90th percentile."
            )

        # ── 6-month trend ─────────────────────────────────────────────────────
        with st.expander("📈 Show 6-month price history for this item"):
            history = get_item_history(df, item_code, selected_state)
            if history is not None and len(history) >= 2:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=history["month"], y=history["median_price"],
                    mode="lines+markers",
                    line=dict(color="#0f2027", width=2.5),
                    marker=dict(size=8, color="#FFD166"),
                    name="Median price",
                ))
                fig_hist.add_hline(
                    y=price_paid, line_dash="dash",
                    line_color="#dc2626" if tier in ("warn","bad") else "#10b981",
                    annotation_text=f"You paid RM {price_paid:.2f}",
                    annotation_position="top right",
                )
                fig_hist.update_layout(
                    height=300, margin=dict(l=20, r=20, t=30, b=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="DM Sans"),
                    xaxis=dict(title="", showgrid=False),
                    yaxis=dict(title="Median Price (RM)", showgrid=True, gridcolor="#f3f4f6"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                first_price = history.iloc[0]["median_price"]
                last_price  = history.iloc[-1]["median_price"]
                trend_pct   = (last_price - first_price) / first_price * 100
                if abs(trend_pct) < 2:
                    trend_msg = f"Prices have been **stable** over the past 6 months."
                elif trend_pct > 0:
                    trend_msg = f"Prices have **risen {trend_pct:.1f}%** over the past 6 months."
                else:
                    trend_msg = f"Prices have **fallen {abs(trend_pct):.1f}%** over the past 6 months."
                st.caption(trend_msg)
            else:
                st.caption("Not enough history available for this item in this state.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="color:#9ca3af;font-size:0.75rem;text-align:center;line-height:1.6">'
    'Price data: <b>data.gov.my</b> · PriceCatcher (KPDN)<br>'
    'Income data: <b>DOSM</b> · Household Income & Expenditure Survey (HIES)<br>'
    'Verdicts based on the actual price distribution for each item in your state.<br>'
    'Income figures are constituency-level medians, indicative of typical households in your area.<br>'
    'Not affiliated with the Government of Malaysia.'
    '</div>',
    unsafe_allow_html=True,
)
