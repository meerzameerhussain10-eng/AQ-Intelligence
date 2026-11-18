import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import pydeck as pdk
except ImportError:
    pdk = None

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
from datetime import datetime, timedelta, timezone

st.markdown("""
<div style="
    background: linear-gradient(120deg, #101D32, #123B42);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
">
<br><br>
    <p style="color:#45D6B0; margin:0;">
        GLOBAL AIR QUALITY • MACHINE LEARNING PROJECT
    </p>
    <h1 style="color:white; margin:10px 0;">
        Welcome, Zameer Baloch
    </h1>
    <p style="color:#A9B9CC; margin:0;">
        Information Technology • 3rd Year
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "processed_air_quality_2024.csv"
)

st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROFESSIONAL UI THEME
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f5f8fc;
        color: #172b4d;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102b3f 0%, #153e50 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f4f8fb !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 7px 4px;
        border-radius: 8px;
    }

    h1 {
        color: #102b3f;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    h2, h3 {
        color: #176b67;
        font-weight: 700;
    }

    p, label, .stCaption {
        color: #526579;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4ebf2;
        border-radius: 15px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(20, 45, 70, 0.045);
    }

    [data-testid="stMetricLabel"] {
        color: #63758a;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #123b50;
        font-weight: 800;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #087f75, #11998e);
        color: white;
        border: 0;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #06665f;
        color: white;
        border: 0;
    }

    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e4ebf2;
        border-radius: 16px;
        padding: 22px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    .hero {
        background: linear-gradient(115deg, #102f45 0%, #176b67 65%, #20a08d 100%);
        padding: 34px 36px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px rgba(16, 47, 69, 0.13);
    }

    .hero h1 {
        color: white !important;
        font-size: 2.35rem;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #e1f1f0 !important;
        font-size: 1.02rem;
        margin: 0;
    }

    .eyebrow {
        color: #a9e6d9;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .section-note {
        color: #718096;
        margin-top: -10px;
        margin-bottom: 18px;
        font-size: 0.92rem;
    }

    .info-panel {
        background: #ffffff;
        border: 1px solid #e4ebf2;
        border-left: 4px solid #11998e;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 10px 0 18px 0;
    }

    .footer {
        text-align: center;
        color: #8493a5;
        font-size: 0.82rem;
        padding: 18px 0 4px 0;
    }

    hr {
        border-color: #e3eaf1;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL, FEATURES AND DATASET
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    return joblib.load(FEATURE_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    final_model = load_model()
    feature_columns = load_features()
except FileNotFoundError:
    st.error(
        "A required model file is missing. Check that the models folder "
        "contains linear_regression.pkl and feature_columns.pkl."
    )
    st.stop()
except Exception as exc:
    st.error(f"Unable to load the saved model: {exc}")
    st.stop()

if not isinstance(feature_columns, (list, tuple)) or len(feature_columns) != 24:
    st.error("The saved feature list is invalid. The model expects 24 features.")
    st.stop()


# =========================================================
# 4. PROJECT DATA AND LOCATION INFORMATION
# =========================================================

locations_by_country = {
    "Germany": [
        "Eisenhuettenstadt", "Frankfurt", "Hagen",
        "Munich", "Spremberg", "Stuttgart"
    ],
    "Japan": [
        "Ichinomiya", "Ichinoseki", "Mishima", "Mitaka", "Nanao"
    ],
    "Pakistan": [
        "Islamabad", "Karachi", "Lahore", "Peshawar"
    ]
}

all_locations = [
    location
    for country_locations in locations_by_country.values()
    for location in country_locations
]

COUNTRY_COLORS = {
    "Germany": "#2f80a8",
    "Japan": "#16a085",
    "Pakistan": "#e08b3e"
}


# Approximate city-centre coordinates for map visualization only.
# These are NOT exact monitoring-station coordinates.
LOCATION_COORDINATES = {
    "Eisenhuettenstadt": (14.635, 52.146),
    "Frankfurt": (8.682, 50.110),
    "Hagen": (7.463, 51.360),
    "Munich": (11.582, 48.135),
    "Spremberg": (14.376, 51.570),
    "Stuttgart": (9.182, 48.775),
    "Ichinomiya": (136.801, 35.304),
    "Ichinoseki": (141.129, 38.934),
    "Mishima": (138.918, 35.118),
    "Mitaka": (139.560, 35.683),
    "Nanao": (136.967, 37.043),
    "Islamabad": (73.0479, 33.6844),
    "Karachi": (67.0011, 24.8607),
    "Lahore": (74.3587, 31.5204),
    "Peshawar": (71.5249, 34.0151)
}

def pm25_marker_color(value):
    if value <= 9.0:
        return [0, 180, 90, 190]
    if value <= 35.4:
        return [240, 205, 35, 200]
    if value <= 55.4:
        return [240, 135, 35, 210]
    if value <= 125.4:
        return [220, 55, 55, 220]
    return [130, 55, 155, 230]


@st.cache_data
def get_dashboard_data():
    df = load_dataset()
    required = {"Datetime", "PM2.5", "Location", "Country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df = df.dropna(subset=["Datetime", "PM2.5", "Location", "Country"])
    df = df[df["PM2.5"] >= 0]
    return df


@st.cache_data
def build_model_test_data():
    """
    Rebuild the next-hour supervised dataset from the processed hourly data.
    Only exact hourly matches are used for all three lags and the next-hour target.
    """
    df = get_dashboard_data().copy()
    df = df.sort_values(["Location", "Datetime"]).reset_index(drop=True)

    # Each row represents the current hour. Join exact preceding hours
    # and the exact next hour so gaps are never treated as consecutive hours.
    current = df[["Datetime", "Location", "Country", "PM2.5"]].copy()
    current = current.rename(columns={"PM2.5": "PM2.5"})

    for lag in (1, 2, 3):
        previous = df[["Datetime", "Location", "PM2.5"]].copy()
        previous["Datetime"] = previous["Datetime"] + pd.Timedelta(hours=lag)
        previous = previous.rename(columns={"PM2.5": f"PM2.5_Lag_{lag}"})
        current = current.merge(
            previous,
            on=["Datetime", "Location"],
            how="inner"
        )

    future = df[["Datetime", "Location", "PM2.5"]].copy()
    future["Datetime"] = future["Datetime"] - pd.Timedelta(hours=1)
    future = future.rename(columns={"PM2.5": "Actual_PM2.5"})
    test_data = current.merge(
        future,
        on=["Datetime", "Location"],
        how="inner"
    )

    # Model time features refer to the hour being predicted, not the current hour.
    forecast_time = test_data["Datetime"] + pd.Timedelta(hours=1)
    test_data["Hour"] = forecast_time.dt.hour
    test_data["Day"] = forecast_time.dt.day
    test_data["DayOfWeek"] = forecast_time.dt.dayofweek
    test_data["Month"] = forecast_time.dt.month
    test_data["IsWeekend"] = (test_data["DayOfWeek"] >= 5).astype(int)

    for feature in feature_columns:
        if feature.startswith("Location_"):
            location_name = feature.replace("Location_", "", 1)
            test_data[feature] = (
                test_data["Location"] == location_name
            ).astype(int)

    required = list(feature_columns) + ["Actual_PM2.5", "Country", "Location", "Datetime"]
    missing = [column for column in required if column not in test_data.columns]
    if missing:
        raise ValueError(f"Could not rebuild model test features: {missing}")

    test_data = test_data.dropna(subset=required)
    test_data["Forecast_Datetime"] = test_data["Datetime"] + pd.Timedelta(hours=1)
    test_data = test_data.sort_values("Forecast_Datetime").reset_index(drop=True)

    # Same chronological holdout boundary used by the project evaluation.
    test_data = test_data[
        test_data["Forecast_Datetime"] >= pd.Timestamp("2024-10-19", tz="UTC")
    ]

    return test_data


try:
    dashboard_df = get_dashboard_data()
except FileNotFoundError:
    dashboard_df = None
except Exception as exc:
    dashboard_df = None
    st.warning(f"Dashboard dataset could not be loaded: {exc}")


# =========================================================
# 5. REUSABLE CHART HELPERS
# =========================================================

def calculate_pm25_aqi_indicator(value):
    """
    Return an indicative US EPA PM2.5 AQI-equivalent for display.

    The input is rounded down to one decimal place before interpolation.
    This is NOT an official daily AQI/NowCast calculation because this
    project predicts a single next-hour PM2.5 concentration.
    """
    try:
        concentration = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("PM2.5 must be a valid numeric value.") from None

    if not concentration.is_finite() or concentration < 0:
        raise ValueError("PM2.5 must be a finite value greater than or equal to zero.")

    concentration = concentration.quantize(
        Decimal("0.1"), rounding=ROUND_DOWN
    )

    # PM2.5 concentration breakpoints (µg/m³), AQI breakpoints,
    # category, background colour, foreground colour.
    bands = [
        (Decimal("0.0"), Decimal("9.0"), 0, 50,
         "Good", "#00e400", "#102010"),
        (Decimal("9.1"), Decimal("35.4"), 51, 100,
         "Moderate", "#ffff00", "#202020"),
        (Decimal("35.5"), Decimal("55.4"), 101, 150,
         "Unhealthy for Sensitive Groups", "#ff7e00", "#202020"),
        (Decimal("55.5"), Decimal("125.4"), 151, 200,
         "Unhealthy", "#ff0000", "#ffffff"),
        (Decimal("125.5"), Decimal("225.4"), 201, 300,
         "Very Unhealthy", "#8f3f97", "#ffffff"),
        (Decimal("225.5"), Decimal("325.4"), 301, 500,
         "Hazardous", "#7e0023", "#ffffff"),
    ]

    for low, high, index_low, index_high, category, bg, fg in bands:
        # Published breakpoint intervals contain 0.1 precision gaps.
        if low <= concentration <= high:
            index_value = (
                Decimal(index_low)
                + Decimal(index_high - index_low)
                * (concentration - low) / (high - low)
            )
            index_value = int(
                index_value.to_integral_value(rounding=ROUND_HALF_UP)
            )
            return concentration, index_value, category, bg, fg

    # Bridge the 0.1 gaps introduced by published breakpoint precision.
    if concentration <= Decimal("9.0"):
        return concentration, 50, "Good", "#00e400", "#102010"
    if concentration <= Decimal("35.4"):
        return concentration, 100, "Moderate", "#ffff00", "#202020"
    if concentration <= Decimal("55.4"):
        return concentration, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "#202020"
    if concentration <= Decimal("125.4"):
        return concentration, 200, "Unhealthy", "#ff0000", "#ffffff"
    if concentration <= Decimal("225.4"):
        return concentration, 300, "Very Unhealthy", "#8f3f97", "#ffffff"

    # Continue the final breakpoint slope above 325.4 rather than
    # artificially capping the displayed index at 500.
    low, high = Decimal("225.5"), Decimal("325.4")
    index_value = Decimal(301) + Decimal(199) * (concentration - low) / (high - low)
    index_value = int(index_value.to_integral_value(rounding=ROUND_HALF_UP))
    return concentration, index_value, "Hazardous", "#7e0023", "#ffffff"


def get_pm25_health_guidance(category):
    guidance = {
        "Good": "Air quality is satisfactory. Most people can continue normal outdoor activities.",
        "Moderate": "Air quality is acceptable for most people. Unusually sensitive people may consider reducing prolonged, heavy outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Children, older adults, and people with heart or lung conditions should reduce prolonged or heavy outdoor exertion.",
        "Unhealthy": "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged or heavy outdoor exertion; others should consider reducing it.",
        "Very Unhealthy": "Health alert: everyone may experience more serious effects. Limit outdoor exertion and follow local public-health guidance.",
        "Hazardous": "Health warning: reduce outdoor exposure where possible and follow local public-health guidance."
    }
    return guidance.get(category, "Follow local public-health guidance.")


def style_axis(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dce5ed")
    ax.spines["bottom"].set_color("#dce5ed")
    ax.tick_params(colors="#526579")


def show_figure(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def hero(title, subtitle, eyebrow="AIR QUALITY INTELLIGENCE"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_heading(title, note=None):
    st.subheader(title)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("## 🌍 AQ Intelligence")
    st.caption("Machine Learning • Environmental Analytics")
    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "🔮 PM2.5 Prediction",
            "🧪 Model Testing",
            "🗺️ Monitoring Map",
            "📊 Data Explorer",
            "ℹ️ About Project"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**PROJECT SNAPSHOT**")
    st.caption("3 countries")
    st.caption("15 selected monitoring locations")
    st.caption("2024 measurement period")
    st.caption("24 model input features")
    st.divider()
    st.caption("Built with Python and Scikit-learn")


# =========================================================
# 7. DASHBOARD PAGE
# =========================================================

if page == "🏠 Dashboard":

    hero(
        "Air Quality Intelligence Dashboard",
        "Explore historical PM2.5 measurements, compare selected monitoring locations, "
        "and identify pollution patterns across the study dataset.",
        "GLOBAL AIR QUALITY ANALYTICS • DATASET PERIOD: 2024"
    )

    if dashboard_df is None or dashboard_df.empty:
        st.error("The processed dataset is unavailable or contains no usable records.")
        st.stop()

    # Interactive dashboard filters
    section_heading(
        "Explore the dataset",
        "Use these filters to update the dashboard charts and summary indicators."
    )

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1.35, 1.2])

    with filter_col1:
        country_options = ["All countries"] + sorted(
            dashboard_df["Country"].dropna().unique().tolist()
        )
        dashboard_country = st.selectbox(
            "Country",
            country_options,
            key="dashboard_country_filter"
        )

    country_scope = dashboard_df.copy()
    if dashboard_country != "All countries":
        country_scope = country_scope[
            country_scope["Country"] == dashboard_country
        ]

    with filter_col2:
        location_options = sorted(
            country_scope["Location"].dropna().unique().tolist()
        )
        selected_locations = st.multiselect(
            "Monitoring locations",
            options=location_options,
            default=location_options,
            key="dashboard_location_filter"
        )

    with filter_col3:
        min_date = dashboard_df["Datetime"].min().date()
        max_date = dashboard_df["Datetime"].max().date()
        selected_dates = st.date_input(
            "Measurement date range (UTC)",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="dashboard_date_filter"
        )

    if not selected_locations:
        st.warning("Select at least one monitoring location to display dashboard results.")
        st.stop()

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Choose both a start date and an end date to apply the date filter.")
        st.stop()

    start_date, end_date = selected_dates
    if start_date > end_date:
        st.error("The start date cannot be later than the end date.")
        st.stop()

    filtered_df = country_scope[
        country_scope["Location"].isin(selected_locations)
    ].copy()

    start_timestamp = pd.Timestamp(start_date, tz="UTC")
    end_timestamp = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    filtered_df = filtered_df[
        (filtered_df["Datetime"] >= start_timestamp)
        & (filtered_df["Datetime"] < end_timestamp)
    ].copy()

    if filtered_df.empty:
        st.warning("No records match the selected filters. Please adjust your selection.")
        st.stop()

    filtered_df = filtered_df.sort_values("Datetime")
    total_records = len(filtered_df)
    country_count = filtered_df["Country"].nunique()
    location_count = filtered_df["Location"].nunique()
    average_pm25 = filtered_df["PM2.5"].mean()
    peak_pm25 = filtered_df["PM2.5"].max()
    median_pm25 = filtered_df["PM2.5"].median()
    peak_record = filtered_df.loc[filtered_df["PM2.5"].idxmax()]
    latest_record = filtered_df.iloc[-1]

    st.divider()
    section_heading(
        "Air quality at a glance",
        "All indicators below are calculated from the currently filtered historical records."
    )

    metric_row1 = st.columns(4)
    metric_row1[0].metric("Countries represented", f"{country_count}")
    metric_row1[1].metric("Monitoring locations", f"{location_count}")
    metric_row1[2].metric("Measurement records", f"{total_records:,}")
    metric_row1[3].metric("Average PM2.5", f"{average_pm25:.2f} µg/m³")

    metric_row2 = st.columns(3)
    metric_row2[0].metric("Peak recorded PM2.5", f"{peak_pm25:.2f} µg/m³")
    metric_row2[1].metric("Median PM2.5", f"{median_pm25:.2f} µg/m³")
    metric_row2[2].metric(
        "Latest record in selection",
        latest_record["Datetime"].strftime("%d %b %Y, %H:%M UTC")
    )

    insight_left, insight_right = st.columns(2)
    with insight_left:
        st.markdown(
            f"""
            <div class="info-panel">
                <b>Highest recorded measurement in this selection</b><br>
                {peak_record["Location"]}, {peak_record["Country"]}<br>
                <span style="font-size:1.35rem;font-weight:800;color:#b45309;">
                    {peak_pm25:.2f} µg/m³
                </span><br>
                <span style="font-size:0.85rem;">
                    {peak_record["Datetime"].strftime("%d %b %Y, %H:%M UTC")}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with insight_right:
        st.markdown(
            f"""
            <div class="info-panel">
                <b>Study coverage reminder</b><br>
                This project analyzes selected monitoring locations in Germany,
                Japan and Pakistan during 2024. Country summaries represent only
                the locations included in this dataset, not national averages.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    section_heading(
        "Country-wise PM2.5 comparison",
        "Mean concentration for the selected monitoring locations and date range."
    )

    country_summary = (
        filtered_df.groupby("Country")["PM2.5"]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
    )

    left, right = st.columns([1.45, 1])

    with left:
        fig, ax = plt.subplots(figsize=(9, 5))
        countries = country_summary.index.tolist()
        values = country_summary["mean"].values
        colors = [COUNTRY_COLORS.get(c, "#4c78a8") for c in countries]
        bars = ax.bar(countries, values, color=colors, width=0.58)
        ax.set_title(
            "Average PM2.5 by selected country locations",
            loc="left", fontsize=13, fontweight="bold",
            color="#17324d", pad=15
        )
        ax.set_ylabel("PM2.5 (µg/m³)")
        style_axis(ax)
        ax.bar_label(bars, fmt="%.1f", padding=4, color="#34495e")
        show_figure(fig)

    with right:
        st.markdown("**Country summary**")
        summary_display = country_summary.rename(
            columns={
                "mean": "Mean PM2.5",
                "median": "Median PM2.5",
                "count": "Records"
            }
        ).copy()
        summary_display["Mean PM2.5"] = summary_display["Mean PM2.5"].round(2)
        summary_display["Median PM2.5"] = summary_display["Median PM2.5"].round(2)
        summary_display["Records"] = summary_display["Records"].map(lambda x: f"{x:,}")
        st.dataframe(summary_display, use_container_width=True)

    st.divider()
    section_heading(
        "Pollution trend over time",
        "Monthly average PM2.5 values for the selected countries and monitoring locations."
    )

    monthly = filtered_df.copy()
    monthly["Month"] = monthly["Datetime"].dt.month
    monthly_summary = (
        monthly.groupby(["Month", "Country"])["PM2.5"]
        .mean()
        .reset_index()
    )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for country in monthly_summary["Country"].unique():
        subset = monthly_summary[monthly_summary["Country"] == country].sort_values("Month")
        ax.plot(
            subset["Month"], subset["PM2.5"],
            marker="o", linewidth=2.4, markersize=5,
            label=country, color=COUNTRY_COLORS.get(country, None)
        )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Mean PM2.5 (µg/m³)")
    ax.set_title(
        "Monthly PM2.5 trend",
        loc="left", fontsize=13, fontweight="bold",
        color="#17324d", pad=15
    )
    style_axis(ax)
    ax.legend(frameon=False, ncol=3)
    show_figure(fig)

    st.divider()
    section_heading(
        "Monitoring location leaderboard",
        "Ranked by average PM2.5 concentration within the active filters."
    )

    location_summary = (
        filtered_df.groupby(["Country", "Location"])["PM2.5"]
        .agg(["mean", "median", "max", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    top_col, bottom_col = st.columns(2)
    with top_col:
        st.markdown("**Highest average PM2.5 locations**")
        top_locations = location_summary.head(5).copy()
        top_locations["Average PM2.5"] = top_locations["mean"].round(2)
        top_locations["Peak PM2.5"] = top_locations["max"].round(2)
        st.dataframe(
            top_locations[["Country", "Location", "Average PM2.5", "Peak PM2.5"]],
            use_container_width=True,
            hide_index=True
        )

    with bottom_col:
        st.markdown("**Lowest average PM2.5 locations**")
        bottom_locations = location_summary.tail(5).sort_values("mean").copy()
        bottom_locations["Average PM2.5"] = bottom_locations["mean"].round(2)
        bottom_locations["Peak PM2.5"] = bottom_locations["max"].round(2)
        st.dataframe(
            bottom_locations[["Country", "Location", "Average PM2.5", "Peak PM2.5"]],
            use_container_width=True,
            hide_index=True
        )

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_locations = location_summary.sort_values("mean", ascending=True)
    labels = [
        f'{row["Location"]} ({row["Country"]})'
        for _, row in plot_locations.iterrows()
    ]
    vals = plot_locations["mean"].values
    bar_colors = [
        COUNTRY_COLORS.get(country, "#4c78a8")
        for country in plot_locations["Country"]
    ]
    ax.barh(labels, vals, color=bar_colors)
    ax.set_xlabel("Mean PM2.5 (µg/m³)")
    ax.set_title(
        "Average PM2.5 by monitoring location",
        loc="left", fontsize=13, fontweight="bold",
        color="#17324d", pad=15
    )
    style_axis(ax)
    show_figure(fig)

    st.divider()
    section_heading(
        "Download filtered data",
        "Export the records currently shown in this dashboard for further analysis."
    )

    export_df = filtered_df.copy()
    export_df["Datetime"] = export_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered measurements (CSV)",
        data=csv_data,
        file_name="aq_intelligence_filtered_measurements.csv",
        mime="text/csv",
        use_container_width=False
    )

    st.caption(
        "Data source: the project's processed 2024 dataset. "
        "This dashboard displays historical dataset records, not live sensor readings. "
        "PM2.5 is measured in µg/m³; the model forecasts next-hour PM2.5 concentration."
    )


# =========================================================
# 8. PM2.5 PREDICTION PAGE
# =========================================================

elif page == "🔮 PM2.5 Prediction":

    hero(
        "Next-Hour PM2.5 Forecast",
        "Enter recent hourly measurements and generate a prediction with the saved Linear Regression model.",
        "MACHINE LEARNING PREDICTION"
    )

    st.info(
        "This page uses manually entered measurements. It does not connect to "
        "live monitoring sensors. Enter values in µg/m³ and provide the "
        "measurement time in UTC."
    )

    location_col, spacer = st.columns([1, 1])
    with location_col:
        country = st.selectbox(
            "Select country",
            options=list(locations_by_country.keys()),
            key="prediction_country"
        )
        location = st.selectbox(
            "Select monitoring location",
            options=locations_by_country[country],
            key="prediction_location"
        )

    st.divider()
    section_heading(
        "Recent PM2.5 measurements",
        "Provide the current reading and the three preceding hourly readings."
    )

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            current_pm25 = st.number_input(
                "Current PM2.5 (µg/m³)", min_value=0.0,
                value=45.0, step=1.0, key="current_pm25"
            )
            lag_1 = st.number_input(
                "1 hour ago (µg/m³)", min_value=0.0,
                value=42.0, step=1.0, key="lag_1"
            )
        with c2:
            lag_2 = st.number_input(
                "2 hours ago (µg/m³)", min_value=0.0,
                value=40.0, step=1.0, key="lag_2"
            )
            lag_3 = st.number_input(
                "3 hours ago (µg/m³)", min_value=0.0,
                value=38.0, step=1.0, key="lag_3"
            )

        st.divider()
        section_heading(
            "Measurement date and time",
            "The model's time features are calculated for the forecast hour. Use UTC."
        )
        d1, d2 = st.columns(2)
        with d1:
            measurement_date = st.date_input(
                "Current measurement date (UTC)",
                value=datetime.now(timezone.utc).date(),
                key="measurement_date"
            )
        with d2:
            measurement_hour = st.selectbox(
                "Current measurement hour (UTC)",
                options=list(range(24)),
                format_func=lambda hour: f"{hour:02d}:00",
                key="measurement_hour"
            )

        predict_button = st.form_submit_button(
            "Generate next-hour prediction",
            use_container_width=True
        )

    if predict_button:
        try:
            current_datetime = datetime.combine(
                measurement_date, datetime.min.time()
            ).replace(hour=measurement_hour, tzinfo=timezone.utc)
            forecast_datetime = current_datetime + timedelta(hours=1)

            day_of_week = forecast_datetime.weekday()

            input_data = {
                "PM2.5": current_pm25,
                "PM2.5_Lag_1": lag_1,
                "PM2.5_Lag_2": lag_2,
                "PM2.5_Lag_3": lag_3,
                "Hour": forecast_datetime.hour,
                "Day": forecast_datetime.day,
                "DayOfWeek": day_of_week,
                "Month": forecast_datetime.month,
                "IsWeekend": 1 if day_of_week >= 5 else 0
            }

            for feature in feature_columns:
                if feature.startswith("Location_"):
                    input_data[feature] = int(
                        feature == f"Location_{location}"
                    )

            missing_features = [
                feature for feature in feature_columns
                if feature not in input_data
            ]
            if missing_features:
                st.error(f"Missing required model features: {missing_features}")
                st.stop()

            input_df = pd.DataFrame([input_data], columns=feature_columns)
            raw_prediction = float(final_model.predict(input_df)[0])
            prediction = max(0.0, raw_prediction)

            st.divider()
            st.success("Prediction generated successfully.")

            result_col, detail_col = st.columns([1, 1.4])
            with result_col:
                st.markdown("#### Predicted next-hour PM2.5")
                st.metric(
                    "PM2.5 concentration",
                    f"{prediction:.2f} µg/m³"
                )
                if raw_prediction < 0:
                    st.caption(
                        "The raw model output was negative and was clipped "
                        "to zero because PM2.5 concentration cannot be negative."
                    )

            with detail_col:
                st.markdown("#### Forecast details")
                st.write(f"**Country:** {country}")
                st.write(f"**Monitoring location:** {location}")
                st.write(
                    f"**Forecast time (UTC):** "
                    f"{forecast_datetime.strftime('%Y-%m-%d %H:%M')}"
                )
                st.write("**Model:** Linear Regression")

            # AQI-equivalent display is a separate interpretation layer.
            # It does not change the trained PM2.5 regression model.
            aqi_concentration, indicative_aqi, aqi_category, aqi_bg, aqi_fg = (
                calculate_pm25_aqi_indicator(prediction)
            )

            st.divider()
            section_heading(
                "Indicative PM2.5 AQI-equivalent",
                "A colour-coded health-category illustration based on the predicted next-hour PM2.5 concentration."
            )

            aqi_col, category_col = st.columns([1, 1.65])
            with aqi_col:
                st.metric("Indicative AQI", f"{indicative_aqi}")
                st.caption("US EPA PM2.5 breakpoint mapping")
                st.caption(f"Input concentration: {aqi_concentration} µg/m³")

            with category_col:
                st.markdown(
                    f"""
                    <div style="
                        background:{aqi_bg};
                        color:{aqi_fg};
                        border-radius:14px;
                        padding:18px 20px;
                        min-height:105px;
                        display:flex;
                        flex-direction:column;
                        justify-content:center;
                    ">
                        <div style="font-size:0.78rem;font-weight:700;letter-spacing:0.08em;">
                            INDICATIVE HEALTH CATEGORY
                        </div>
                        <div style="font-size:1.35rem;font-weight:800;margin-top:7px;">
                            {aqi_category}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.info(get_pm25_health_guidance(aqi_category))
            st.warning(
                "Important limitation: this is an indicative AQI-equivalent "
                "for a single-hour model forecast, not an official daily AQI "
                "or NowCast. Official PM2.5 AQI reporting uses the applicable "
                "24-hour concentration/NowCast procedure. This application "
                "does not calculate an official 24-hour AQI."
            )

            st.divider()
            section_heading(
                "Measurement history and forecast",
                "The final point is the model's next-hour prediction."
            )

            chart_labels = [
                "3 hours ago", "2 hours ago", "1 hour ago",
                "Current", "Next hour"
            ]
            chart_values = [lag_3, lag_2, lag_1, current_pm25, prediction]

            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(
                chart_labels[:4], chart_values[:4],
                marker="o", linewidth=2.5, color="#2878a5",
                label="Entered measurements"
            )
            ax.plot(
                chart_labels[3:], chart_values[3:],
                marker="o", linewidth=2.5, linestyle="--",
                color="#11998e", label="Current to prediction"
            )
            ax.scatter(
                ["Next hour"], [prediction],
                s=120, color="#e08b3e", zorder=5,
                label="Predicted value"
            )
            ax.set_ylabel("PM2.5 (µg/m³)")
            ax.set_title("Recent measurements and next-hour forecast",
                         loc="left", fontsize=13, fontweight="bold",
                         color="#17324d", pad=15)
            style_axis(ax)
            ax.legend(frameon=False)
            show_figure(fig)

            st.caption(
                "The historical values above are entered by the user; "
                "the forecast is generated by the trained model."
            )
            st.caption(
                "The model forecast is generated from manually entered "
                "measurements. The AQI-equivalent category is an educational "
                "indicator, not a live sensor reading or official AQI."
            )

        except Exception as exc:
            st.error(f"Prediction error: {exc}")


# =========================================================
# 9. MODEL TESTING PAGE
# =========================================================

elif page == "🧪 Model Testing":

    hero(
        "Model Performance Testing",
        "Compare the model's next-hour forecasts with actual PM2.5 measurements from the held-out test period.",
        "ACTUAL VS PREDICTED EVALUATION"
    )

    st.info(
        "This page evaluates the saved Linear Regression model on later-date "
        "records from the same selected monitoring locations. Actual next-hour "
        "values are used only for evaluation, never as model input. This is a "
        "historical test, not a live forecast."
    )

    try:
        evaluation_df = build_model_test_data()

        if evaluation_df.empty:
            st.error(
                "No valid test rows could be rebuilt. Check that the processed "
                "dataset contains continuous hourly records and covers "
                "October 19 to December 31, 2024."
            )
            st.stop()

        X_test = evaluation_df[list(feature_columns)].copy()
        y_actual = evaluation_df["Actual_PM2.5"].astype(float).to_numpy()

        y_pred_raw = final_model.predict(X_test)
        y_pred = np.maximum(y_pred_raw, 0)

        mae = mean_absolute_error(y_actual, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
        r2 = r2_score(y_actual, y_pred) if len(y_actual) > 1 else float("nan")

        results_df = evaluation_df[
            ["Forecast_Datetime", "Country", "Location", "Actual_PM2.5"]
        ].copy()
        results_df["Predicted_PM2.5"] = y_pred
        results_df["Absolute_Error"] = np.abs(y_actual - y_pred)

        # Relative error is undefined when the actual value is zero.
        results_df["Relative_Error_%"] = np.where(
            y_actual > 0,
            results_df["Absolute_Error"] / y_actual * 100,
            np.nan
        )

        section_heading(
            "Evaluation summary",
            "Metrics calculated from the rebuilt chronological holdout test set."
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Test records", f"{len(results_df):,}")
        k2.metric("MAE", f"{mae:.2f} µg/m³")
        k3.metric("RMSE", f"{rmse:.2f} µg/m³")
        k4.metric("R² score", f"{r2:.4f}" if np.isfinite(r2) else "N/A")

        st.caption(
            "R² is not percentage accuracy. MAE and RMSE are expressed in "
            "µg/m³ and summarize the errors across the test records."
        )

        st.divider()
        section_heading(
            "Actual vs predicted PM2.5",
            "The closer the points are to the dashed diagonal, the closer predictions are to actual values."
        )

        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.scatter(
            y_actual, y_pred,
            alpha=0.28, s=18, color="#168b80", edgecolors="none"
        )

        lower = float(min(np.min(y_actual), np.min(y_pred)))
        upper = float(max(np.max(y_actual), np.max(y_pred)))
        ax.plot(
            [lower, upper], [lower, upper],
            linestyle="--", linewidth=2, color="#e08b3e",
            label="Perfect prediction"
        )
        ax.set_xlabel("Actual PM2.5 (µg/m³)")
        ax.set_ylabel("Predicted PM2.5 (µg/m³)")
        ax.set_title(
            "Actual vs predicted values",
            loc="left", fontsize=13, fontweight="bold",
            color="#17324d", pad=15
        )
        style_axis(ax)
        ax.legend(frameon=False)
        show_figure(fig)

        st.divider()
        section_heading(
            "Prediction error distribution",
            "Absolute error shows how far each prediction was from its actual measurement."
        )

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.hist(
            results_df["Absolute_Error"],
            bins=50, color="#2f8f83",
            edgecolor="white", alpha=0.9
        )
        ax.axvline(
            mae, color="#e08b3e", linestyle="--",
            linewidth=2, label=f"MAE = {mae:.2f} µg/m³"
        )
        ax.set_xlabel("Absolute error (µg/m³)")
        ax.set_ylabel("Number of test records")
        ax.set_title(
            "Distribution of absolute prediction errors",
            loc="left", fontsize=13, fontweight="bold",
            color="#17324d", pad=15
        )
        style_axis(ax)
        ax.legend(frameon=False)
        show_figure(fig)

        st.divider()
        section_heading(
            "Inspect individual predictions",
            "Filter the test results and compare actual and predicted PM2.5 values."
        )

        filter1, filter2 = st.columns(2)
        country_options = ["All countries"] + sorted(results_df["Country"].unique())
        with filter1:
            test_country = st.selectbox(
                "Country", country_options, key="test_country"
            )

        filtered_results = results_df.copy()
        if test_country != "All countries":
            filtered_results = filtered_results[
                filtered_results["Country"] == test_country
            ]

        location_options = ["All locations"] + sorted(
            filtered_results["Location"].unique()
        )
        with filter2:
            test_location = st.selectbox(
                "Monitoring location", location_options, key="test_location"
            )

        if test_location != "All locations":
            filtered_results = filtered_results[
                filtered_results["Location"] == test_location
            ]

        display_results = filtered_results.copy()
        display_results["Forecast_Datetime"] = (
            display_results["Forecast_Datetime"].dt.strftime("%Y-%m-%d %H:%M UTC")
        )
        for col in [
            "Actual_PM2.5", "Predicted_PM2.5",
            "Absolute_Error", "Relative_Error_%"
        ]:
            display_results[col] = display_results[col].round(2)

        st.dataframe(
            display_results.sort_values("Forecast_Datetime", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        csv_results = filtered_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download evaluation results as CSV",
            data=csv_results,
            file_name="actual_vs_predicted_results.csv",
            mime="text/csv"
        )

    except Exception as exc:
        st.error(f"Model testing could not be completed: {exc}")



# =========================================================
# 10. INTERACTIVE MONITORING MAP PAGE
# =========================================================

elif page == "🗺️ Monitoring Map":

    hero(
        "Air Quality Monitoring Map",
        "Explore the latest available PM2.5 measurements at the 15 selected study locations.",
        "LOCATION-BASED AIR QUALITY EXPLORATION"
    )

    if dashboard_df is None or dashboard_df.empty:
        st.error("The processed dataset is unavailable or contains no usable records.")
        st.stop()

    st.info(
        "Map markers use approximate city-centre coordinates because exact "
        "monitoring-station latitude and longitude are not included in the dataset. "
        "Marker colours represent PM2.5 concentration, not official AQI."
    )

    f1, f2 = st.columns([1, 1.5])
    with f1:
        map_country = st.selectbox(
            "Country filter",
            ["All countries"] + sorted(dashboard_df["Country"].unique().tolist()),
            key="map_country_filter"
        )
    with f2:
        min_date = dashboard_df["Datetime"].min().date()
        max_date = dashboard_df["Datetime"].max().date()
        map_dates = st.date_input(
            "Historical date range (UTC)",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="map_date_range"
        )

    if not isinstance(map_dates, (tuple, list)) or len(map_dates) != 2:
        st.warning("Choose both a start date and an end date.")
        st.stop()
    start_date, end_date = map_dates
    if start_date > end_date:
        st.error("The start date cannot be later than the end date.")
        st.stop()

    map_df = dashboard_df.copy()
    if map_country != "All countries":
        map_df = map_df[map_df["Country"] == map_country]

    # The processed CSV stores locations as "Country - City" (for example,
    # "Pakistan - Karachi"), while the coordinate dictionary uses city names.
    # Normalize only this page's copy so other pages/model features stay unchanged.
    map_df["Location"] = (
        map_df["Location"].astype(str).str.split(" - ", n=1).str[-1].str.strip()
    )

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    map_df = map_df[
        (map_df["Datetime"] >= start_ts) & (map_df["Datetime"] < end_ts)
    ].copy()

    if map_df.empty:
        st.warning("No measurements match the selected country and date filters. Try a wider date range or select All countries.")
        st.stop()

    latest = (
        map_df.sort_values("Datetime")
        .groupby(["Country", "Location"], as_index=False)
        .tail(1)
        .copy()
    )
    latest["Longitude"] = latest["Location"].map(
        lambda x: LOCATION_COORDINATES.get(x, (None, None))[0]
    )
    latest["Latitude"] = latest["Location"].map(
        lambda x: LOCATION_COORDINATES.get(x, (None, None))[1]
    )
    latest = latest.dropna(subset=["Longitude", "Latitude", "PM2.5", "Datetime"])

    if latest.empty:
        st.warning("No locations could be matched to map coordinates. Check the Location names in the processed dataset.")
        st.stop()

    latest["Marker_Color"] = latest["PM2.5"].apply(pm25_marker_color)
    latest["PM25_Label"] = latest["PM2.5"].map(lambda x: f"{x:.2f} µg/m³")
    latest["Recorded_UTC"] = latest["Datetime"].dt.strftime("%Y-%m-%d %H:%M UTC")

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Locations on map", f"{len(latest)}")
    k2.metric("Mean latest PM2.5", f"{latest['PM2.5'].mean():.2f} µg/m³")
    k3.metric("Highest latest PM2.5", f"{latest['PM2.5'].max():.2f} µg/m³")
    latest_timestamp = latest["Datetime"].max()
    latest_label = latest_timestamp.strftime("%d %b %Y") if pd.notna(latest_timestamp) else "N/A"
    k4.metric("Latest record (UTC)", latest_label)

    section_heading(
        "Geographical distribution",
        "Each marker shows the latest available measurement for that location in the selected date range."
    )
    left, right = st.columns([1.7, 1])

    with left:
        if pdk is not None:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=latest,
                get_position="[Longitude, Latitude]",
                get_fill_color="Marker_Color",
                get_line_color=[35, 55, 75, 220],
                get_radius=95000,
                radius_min_pixels=7,
                radius_max_pixels=24,
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    latitude=32, longitude=75, zoom=1.1, pitch=0, bearing=0
                ),
                tooltip={
                    "html": "<b>{Location}, {Country}</b><br/>PM2.5: {PM25_Label}<br/>Recorded: {Recorded_UTC}",
                    "style": {"backgroundColor": "#17324d", "color": "white"}
                },
                map_style="light"
            )
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.map(
                latest.rename(columns={"Latitude": "lat", "Longitude": "lon"})[
                    ["lat", "lon"]
                ],
                use_container_width=True
            )
            st.caption("For coloured markers and hover details, install PyDeck: pip install pydeck")

    with right:
        st.markdown("**Latest measurement by location**")
        table = latest[["Country", "Location", "PM2.5", "Recorded_UTC"]].copy()
        table["PM2.5"] = table["PM2.5"].round(2)
        st.dataframe(table.sort_values("PM2.5", ascending=False),
                     use_container_width=True, hide_index=True, height=420)

    st.divider()
    section_heading("Selected location analysis", "Inspect the latest reading and historical PM2.5 trend.")
    places = sorted(latest["Location"].unique().tolist())
    selected_place = st.selectbox("Select monitoring location", places, key="map_selected_place")
    history = map_df[map_df["Location"] == selected_place].sort_values("Datetime")

    if not history.empty:
        last = history.iloc[-1]
        a, b, c, d = st.columns(4)
        a.metric("Latest PM2.5", f"{last['PM2.5']:.2f} µg/m³")
        b.metric("Average PM2.5", f"{history['PM2.5'].mean():.2f} µg/m³")
        c.metric("Maximum PM2.5", f"{history['PM2.5'].max():.2f} µg/m³")
        d.metric("Records", f"{len(history):,}")

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(history["Datetime"], history["PM2.5"], color="#168b80", linewidth=1.5)
        ax.set_xlabel("Date and time (UTC)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"Historical PM2.5 — {selected_place}",
                     loc="left", fontsize=13, fontweight="bold",
                     color="#17324d", pad=15)
        fig.autofmt_xdate()
        style_axis(ax)
        show_figure(fig)

    st.divider()
    section_heading("Compare locations", "Compare daily average PM2.5 for selected locations.")
    options = sorted(map_df["Location"].unique().tolist())
    defaults = options[:min(5, len(options))]
    compare_places = st.multiselect(
        "Locations to compare", options, default=defaults, key="map_compare_places"
    )
    if compare_places:
        comp = map_df[map_df["Location"].isin(compare_places)].copy()
        comp["Date"] = comp["Datetime"].dt.floor("D")
        comp = comp.groupby(["Date", "Location"])["PM2.5"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 4.8))
        for place in compare_places:
            part = comp[comp["Location"] == place].sort_values("Date")
            if not part.empty:
                ax.plot(part["Date"], part["PM2.5"], linewidth=2, label=place)
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("Daily mean PM2.5 (µg/m³)")
        ax.set_title("Daily PM2.5 comparison",
                     loc="left", fontsize=13, fontweight="bold",
                     color="#17324d", pad=15)
        fig.autofmt_xdate()
        style_axis(ax)
        ax.legend(frameon=False, ncol=2)
        show_figure(fig)
    else:
        st.info("Select at least one location to display the comparison chart.")

    st.caption(
        "Map coordinates are approximate city-centre positions for visual orientation. "
        "They do not represent exact monitoring-station positions or live sensor readings."
    )


# =========================================================
# 11. DATA EXPLORER PAGE
# =========================================================

# =========================================================
# 10. DATA EXPLORER PAGE
# =========================================================

elif page == "📊 Data Explorer":

    hero(
        "Explore the Air Quality Dataset",
        "Filter the processed 2024 measurements by country and monitoring location.",
        "DATA EXPLORATION"
    )

    if dashboard_df is None or dashboard_df.empty:
        st.error("The processed dataset is unavailable.")
        st.stop()

    filter_col1, filter_col2 = st.columns(2)
    available_countries = sorted(dashboard_df["Country"].unique().tolist())

    with filter_col1:
        selected_country = st.selectbox(
            "Filter by country",
            ["All countries"] + available_countries
        )

    filtered_df = dashboard_df.copy()
    if selected_country != "All countries":
        filtered_df = filtered_df[
            filtered_df["Country"] == selected_country
        ]

    available_places = sorted(filtered_df["Location"].unique().tolist())
    with filter_col2:
        selected_location = st.selectbox(
            "Filter by monitoring location",
            ["All locations"] + available_places
        )

    if selected_location != "All locations":
        filtered_df = filtered_df[
            filtered_df["Location"] == selected_location
        ]

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Filtered records", f"{len(filtered_df):,}")
    b.metric("Mean PM2.5", f"{filtered_df['PM2.5'].mean():.2f} µg/m³")
    c.metric("Maximum PM2.5", f"{filtered_df['PM2.5'].max():.2f} µg/m³")

    st.divider()
    section_heading(
        "PM2.5 distribution",
        "Histogram of the selected measurements. Extreme observations are retained."
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(
        filtered_df["PM2.5"],
        bins=50,
        color="#2f8f83",
        edgecolor="white",
        alpha=0.9
    )
    ax.set_xlabel("PM2.5 (µg/m³)")
    ax.set_ylabel("Number of records")
    ax.set_title("Distribution of PM2.5 measurements",
                 loc="left", fontsize=13, fontweight="bold",
                 color="#17324d", pad=15)
    style_axis(ax)
    show_figure(fig)

    section_heading("Filtered data preview")
    preview = filtered_df.sort_values("Datetime", ascending=False).head(100)
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.caption("Showing the latest 100 rows from the selected filter.")

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_data,
        file_name="filtered_air_quality_data.csv",
        mime="text/csv"
    )


# =========================================================
# 11. ABOUT PROJECT PAGE
# =========================================================

elif page == "ℹ️ About Project":

    hero(
        "About the Project",
        "A university Machine Learning project for short-term PM2.5 forecasting.",
        "PROJECT INFORMATION"
    )

    section_heading("Project overview")
    st.write(
        "The system analyzes historical PM2.5 measurements from selected "
        "monitoring locations in Germany, Japan and Pakistan, then predicts "
        "the next-hour PM2.5 concentration using a trained regression model."
    )

    st.divider()
    section_heading("Project objectives")
    st.markdown("""
    - Analyze historical air pollution measurements.
    - Explore temporal and location-based PM2.5 patterns.
    - Prepare time-series features for next-hour forecasting.
    - Train and compare regression models.
    - Evaluate model performance on a later-date test period.
    - Deploy the selected model through a Streamlit application.
    """)

    st.divider()
    section_heading("Technologies used")
    tech1, tech2, tech3, tech4 = st.columns(4)
    tech1.metric("Language", "Python")
    tech2.metric("Data analysis", "Pandas / NumPy")
    tech3.metric("Visualization", "Matplotlib")
    tech4.metric("Deployment", "Streamlit")

    st.write(
        "Additional tools: Scikit-learn for machine learning and Joblib "
        "for saving and loading trained model artifacts."
    )

    st.divider()
    section_heading(
        "Selected model performance",
        "Corrected Linear Regression results on the held-out October–December 2024 test period."
    )

    st.info(
        "Open the Model Testing page to calculate the current MAE, RMSE, "
        "R² score and test-record count directly from the saved model and "
        "chronological holdout dataset. This avoids displaying outdated hard-coded metrics."
    )

    st.divider()
    section_heading("Methodology")
    methodology = [
        "1. Data Loading",
        "2. Data Understanding",
        "3. Data Preprocessing",
        "4. Exploratory Data Analysis",
        "5. X and Y Split",
        "6. Feature Engineering and Selection",
        "7. Training and Testing Split",
        "8. Model Selection and Training",
        "9. Model Evaluation",
        "10. Prediction"
    ]
    st.markdown("\n".join(f"- {item}" for item in methodology))

    st.divider()
    section_heading("Scope and limitations")
    st.markdown("""
    - The dataset contains 15 selected monitoring locations across Germany,
      Japan and Pakistan during 2024; it does not cover every country.
    - Country comparisons summarize the selected monitoring locations and
      are not national averages.
    - The model predicts next-hour PM2.5 concentration in µg/m³. The separate AQI-equivalent display is indicative only, not an official daily AQI/NowCast.
    - The prediction page uses manually entered measurements and does not
      retrieve live sensor readings.
    - Model testing used later dates from the same selected monitoring
      locations; performance on unseen locations has not been established.
    - Prediction quality depends on the quality and representativeness of
      the input measurements.
    """)

    st.divider()
    st.markdown(
        """
        <div class="info-panel">
        <b>Project takeaway:</b> This application demonstrates a complete
        Machine Learning workflow—from historical data analysis and feature
        engineering to model evaluation and an interactive prediction interface.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Air Quality Prediction System &nbsp;•&nbsp;
        University Machine Learning Project &nbsp;•&nbsp;
        Python + Scikit-learn + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)