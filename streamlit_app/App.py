# -------------------------------
# auckland_aq_dashboard_v27_full_upgraded_filtered.py
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# TensorFlow LSTM included in ensemble
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# ---------------------------
# Forecasting Abbreviation Mapping
# ---------------------------
forecast_abbr = {
    'Temp': 'Temperature',
    'TEMP': 'Temperature',
    'RH': 'Relative Humidity',
    'WS': 'Wind Speed',
    'WD': 'Wind Direction'
}

traffic_abbr = {
    'TrafficV': 'Traffic Volume'
}

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(
    page_title="Auckland Air Quality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌏 Auckland Council — Air Quality Dashboard")
st.markdown("_Pre-cleaned or raw uploads → improved forecasting → daily-aggregated pollutant-weather/traffic correlations._")

# ---------------------------
# NESAQ THRESHOLDS
# ---------------------------
nesaq_thresholds = {
    'PM2.5': {'Annual': 10, '24-hour': 25},
    'PM10': {'Annual': 20, '24-hour': 50},
    'O3': {'8-hour': 100, '1-hour': 150},
    'NO2': {'Annual': 40, '24-hour': 100, '1-hour': 200},
    'SO2': {'24-hour': 350},
    'CO': {'1-hour': 30, '8-hour': 10, '24-hour': 4}
}
nesaq_pollutants_in_data = list(nesaq_thresholds.keys())

# ---------------------------
# Helpers / Data Loading
# ---------------------------
@st.cache_data
def load_combined_dataset(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Site','Parameter','Value','Datetime']).sort_values('Datetime')
    return df

@st.cache_data
def load_raw_dataset(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    return df

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def clean_raw_dataset(df, nesaq_pollutants_in_data=None, ffill_limit=2, knn_neighbors=3):
    """
    Clean and preprocess raw air quality monitoring data.

    Parameters:
        df (pd.DataFrame): Raw dataset with columns like 'Date', 'Time', 'Datetime', 'Site', 'Parameter', 'Value'.
        nesaq_pollutants_in_data (list): List of key pollutants to check for negative values.
        ffill_limit (int): Maximum consecutive missing values to forward/backward fill.
        knn_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: Cleaned dataset with correct types and imputed missing values.
    """

    # -------------------------------
    # 1. Datetime creation and validation
    # -------------------------------
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    else:
        raise ValueError("Dataset must have 'Datetime' or both 'Date' + 'Time' columns.")

    # -------------------------------
    # 2. Column existence check
    # -------------------------------
    for col in ['Site', 'Parameter', 'Value']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # -------------------------------
    # 3. Enforce correct data types
    # -------------------------------
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Site'] = df['Site'].astype('category')
    df['Parameter'] = df['Parameter'].astype('category')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Warn if too many values could not be converted
    if df['Value'].isna().mean() > 0.1:
        print("Warning: More than 10% of 'Value' could not be converted to numeric.")

    # -------------------------------
    # 4. Correct negative values for key pollutants
    # -------------------------------
    if nesaq_pollutants_in_data:
        mask = df['Parameter'].isin(nesaq_pollutants_in_data) & (df['Value'] < 0)
        df.loc[mask, 'Value'] = df.loc[mask, 'Value'].abs()

    # -------------------------------
    # 5. Fill missing categorical columns
    # -------------------------------
    cat_cols = ['Site', 'Parameter']
    df = df.sort_values('Datetime')
    for col in cat_cols:
        df[col] = df[col].ffill().bfill()
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value[0])

    # -------------------------------
    # 6. Numeric imputation
    # -------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col == 'Value':
            df_grouped = []
            for (site, param), group in df.groupby(['Site', 'Parameter']):
                group = group.sort_values('Datetime').copy()
                group['Value'] = group['Value'].ffill(limit=ffill_limit).bfill(limit=ffill_limit)
                if group['Value'].isna().sum() > 0 and group['Value'].notna().sum() >= 5:
                    imputer = KNNImputer(n_neighbors=knn_neighbors)
                    group[['Value']] = imputer.fit_transform(group[['Value']])
                df_grouped.append(group)
            df = pd.concat(df_grouped, ignore_index=True)
        else:
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df[[col]] = imputer.fit_transform(df[[col]])

    # -------------------------------
    # 7. Final clean-up
    # -------------------------------
    df = df.dropna(subset=['Site', 'Parameter', 'Value', 'Datetime']).sort_values('Datetime')
    df.reset_index(drop=True, inplace=True)

    return df


@st.cache_data
def pivot_long_to_wide(df):
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    wide = df.pivot_table(index='Datetime', columns='Parameter', values='Value', aggfunc='mean')
    wide = wide.sort_index()
    return wide

# ---------------------------
# Sidebar - Uploads + Controls
# ---------------------------
st.sidebar.header("Dataset Upload Options")
dataset_mode = st.sidebar.radio("Choose dataset type:", ['Pre-cleaned combined dataset', 'Raw dataset(s)'])

uploaded_file = None
uploaded_files = None
df = None

if dataset_mode == 'Pre-cleaned combined dataset':
    uploaded_file = st.file_uploader("Upload Combined Dataset (CSV/Excel)", type=["csv","xlsx"])
    if uploaded_file:
        try:
            df = load_combined_dataset(uploaded_file)
            st.success(f"✅ Combined dataset loaded: {len(df)} records, {df['Site'].nunique()} sites")
        except Exception as e:
            st.error(f"Error loading combined dataset: {e}")
else:
    uploaded_files = st.file_uploader("Upload One or More Raw Datasets (CSV/Excel)",
                                      type=["csv","xlsx"], accept_multiple_files=True)
    if uploaded_files:
        all_cleaned = []
        for f in uploaded_files:
            try:
                tmp = load_raw_dataset(f)
                tmp_clean = clean_raw_dataset(tmp)
                all_cleaned.append(tmp_clean)
                st.success(f"✅ {f.name} cleaned: {len(tmp_clean)} records, {tmp_clean['Site'].nunique()} sites")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
        if all_cleaned:
            df = pd.concat(all_cleaned, ignore_index=True)
            st.subheader("📊 Comparison Dataset Summary")
            st.dataframe(df.groupby(['Site','Parameter']).agg({'Value':'mean'}).reset_index())

# ---------------------------
# Global Detected Variables (Sidebar)
# ---------------------------
if df is not None:
    wide_df = pivot_long_to_wide(df)
    all_columns = wide_df.columns.tolist()

    nesaq_pollutants_in_data_detected = [c for c in all_columns if c in nesaq_pollutants_in_data]
    aux_pollutants = ['BC(370)','BC(880)','NO','NOx','AQI','TEMP']
    non_pollutants = [c for c in all_columns if c not in nesaq_pollutants_in_data + aux_pollutants]

    traffic_vars_in_data = [c for c in non_pollutants if 'traffic' in c.lower()]
    weather_vars_in_data = [c for c in non_pollutants if c not in traffic_vars_in_data]

    st.sidebar.header("Detected Variables")
    st.sidebar.write(f"✅ Key Pollutants Detected: {nesaq_pollutants_in_data_detected}")
    st.sidebar.write(f"🌦 Weather Variables Detected: {weather_vars_in_data}")
    st.sidebar.write(f"🛣 Traffic Variables Detected: {traffic_vars_in_data}")
    st.sidebar.write(f"ℹ️ Other Pollutants/Variables: {aux_pollutants}")  

    st.sidebar.markdown("---")
    st.sidebar.header("Forecasting Controls")
    do_forecast = st.sidebar.checkbox("Enable forecasting computations", value=True)
    log_adjust = st.sidebar.selectbox("Lag days (temporal feature)", list(range(1,15)), index=0)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=7, value=3)

# ---------------------------
# Site selector
# ---------------------------
if df is not None:
    available_sites = sorted(df['Site'].unique())
    selected_site = st.sidebar.selectbox("Select Site (Global Filter)", ["All Sites"] + available_sites)
    if selected_site != "All Sites":
        df_filtered = df[df['Site'] == selected_site].copy()
    else:
        df_filtered = df.copy()

# ---------------------------
# Pivot wide for weather/traffic correlation
# ---------------------------
if df is not None:
    wide_df = pivot_long_to_wide(df_filtered)

# ---------------------------
# Dashboard Tabs
# ---------------------------
if df is not None:
    tabs = st.tabs(["Overview","Temporal","Spatial","Weather & Traffic","Forecasting","Insights & Policy"])

    # ---------------------------
    # Overview Tab
    # ---------------------------
    with tabs[0]:
        st.subheader("📄 Project Overview & NESAQ Thresholds")
        threshold_rows = []
        for pollutant, periods in nesaq_thresholds.items():
            for period, value in periods.items():
                threshold_rows.append([pollutant, period, value])
        st.dataframe(pd.DataFrame(threshold_rows, columns=['Pollutant', 'Period', 'NESAQ Threshold']))

        st.subheader("📊 Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sites", df['Site'].nunique())
        col2.metric("Parameters", df['Parameter'].nunique())
        col3.metric("Records", len(df))

    # ---------------------------
    # Temporal Tab
    # ---------------------------
    with tabs[1]:
        st.subheader("⏱ Exceedances & Compliance")
        for pollutant in nesaq_pollutants_in_data_detected:
            df_param = df_filtered[df_filtered['Parameter']==pollutant].copy()
            thresholds = nesaq_thresholds.get(pollutant, {})
            st.markdown(f"### {pollutant} Exceedances")
            for period, threshold_value in thresholds.items():
                if period.lower() == 'annual':
                    annual_avg = df_param.resample('Y', on='Datetime')['Value'].mean().reset_index()
                    # filter out years with no records
                    annual_avg = annual_avg[~annual_avg['Value'].isna()]
                    annual_avg['Exceedance'] = annual_avg['Value'] > threshold_value
                    fig = px.bar(annual_avg, x='Datetime', y='Value', color='Exceedance',
                                 color_discrete_map={True:'red', False:'green'},
                                 title=f"{pollutant} Annual Exceedances")
                    fig.add_hline(y=threshold_value, line_dash='dash', annotation_text=f"Threshold: {threshold_value}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    df_param = df_param.sort_values('Datetime').copy()
                    if '24-hour' in period.lower():
                        agg = df_param.resample('D', on='Datetime')['Value'].mean().reset_index()
                        agg.rename(columns={'Value':'Value','Datetime':'Date'}, inplace=True)
                    elif '8-hour' in period.lower():
                        df_param['8hr_mean'] = df_param['Value'].rolling(window=8, min_periods=1).mean()
                        agg = df_param.groupby(df_param['Datetime'].dt.date)['8hr_mean'].max().reset_index()
                        agg.rename(columns={'8hr_mean':'Value','Datetime':'Date'}, inplace=True)
                    else:
                        agg = df_param.groupby(df_param['Datetime'].dt.date)['Value'].max().reset_index()
                        agg.rename(columns={'Datetime':'Date'}, inplace=True)

                    # filter out dates with no recorded data
                    agg = agg[~agg['Value'].isna()]
                    agg['Exceedance'] = agg['Value'] > threshold_value
                    fig = px.bar(agg, x='Date', y='Value', color='Exceedance',
                                 color_discrete_map={True:'red', False:'green'},
                                 title=f"{pollutant} {period} Exceedances")
                    fig.add_hline(y=threshold_value, line_dash='dash', annotation_text=f"Threshold: {threshold_value}")
                    st.plotly_chart(fig, use_container_width=True)


    # ---------------------------
    # Spatial Tab
    # ---------------------------
    with tabs[2]:
        st.subheader("📍 Spatial Comparison Across Sites")
        spatial_param = st.selectbox("Select Pollutant for Spatial Comparison", nesaq_pollutants_in_data_detected)
        df_spatial = df[df['Parameter']==spatial_param]
        site_avg = df_spatial.groupby('Site')['Value'].mean().reset_index()
        fig_bar = px.bar(site_avg, x='Site', y='Value', color='Site',
                         title=f"Average {spatial_param} by Site")
        thr = nesaq_thresholds.get(spatial_param, {}).get('Annual') or nesaq_thresholds.get(spatial_param, {}).get('24-hour')
        if thr is not None:
            fig_bar.add_hline(y=thr, line_dash='dash', annotation_text=f"Threshold: {thr}")
        st.plotly_chart(fig_bar, use_container_width=True)
	
    # ---------------------------
    # Weather & Traffic Tab
    # ---------------------------
    with tabs[3]:
        st.subheader("🌦 Weather & 🛣 Traffic Insights (Daily Aggregated)")
    
        all_columns = wide_df.columns.tolist()
        pollutants_in_data = [c for c in all_columns if c in nesaq_pollutants_in_data_detected]
        non_pollutants = [c for c in all_columns if c not in nesaq_pollutants_in_data_detected + aux_pollutants]
    
        traffic_vars_in_data = [c for c in non_pollutants if 'traffic' in c.lower()]
        weather_vars_in_data = [c for c in non_pollutants if c not in traffic_vars_in_data]
    
        friendly_map = {**forecast_abbr, **traffic_abbr}
        pollutants_friendly = [friendly_map.get(p, p) for p in pollutants_in_data]
        weather_vars_friendly = [friendly_map.get(w, w) for w in weather_vars_in_data]
        traffic_vars_friendly = [friendly_map.get(t, t) for t in traffic_vars_in_data]
    
        st.write(f"✅ Key Pollutants: {pollutants_friendly}")
        st.write(f"🌦 Weather Variables: {weather_vars_friendly}")
        st.write(f"🛣 Traffic Variables: {traffic_vars_friendly}")
    
        if not pollutants_in_data:
            st.info("No key pollutants detected in dataset.")
        elif not weather_vars_in_data and not traffic_vars_in_data:
            st.info("No weather or traffic variables detected in dataset.")
        else:
            wide_daily = df_filtered.pivot_table(index='Datetime', columns='Parameter', values='Value')
            wide_daily = wide_daily.apply(pd.to_numeric, errors='coerce').resample('D').mean()
            wide_daily = wide_daily.dropna(how='all')
    
            corr_columns = weather_vars_in_data + traffic_vars_in_data
            corr_data = pd.DataFrame(index=pollutants_in_data, columns=corr_columns, dtype=float)
    
            for pollutant in pollutants_in_data:
                for var in corr_columns:
                    if pollutant in wide_daily.columns and var in wide_daily.columns:
                        merged = pd.concat([wide_daily[pollutant], wide_daily[var]], axis=1).dropna()
                        corr_val = merged.iloc[:, 0].corr(merged.iloc[:, 1]) if len(merged) > 1 else np.nan
                        corr_data.loc[pollutant, var] = corr_val
                    else:
                        corr_data.loc[pollutant, var] = np.nan
    
            corr_data = corr_data.astype(float)
    
            # ---------------------------
            # Highlight function for dataframe
            # ---------------------------
            def highlight_corr(val):
                if pd.isna(val):
                    return 'background-color: #f0f0f0;'
                if val >= 0.6:
                    return 'background-color: green; color: white; font-weight:bold;'
                if val <= -0.6:
                    return 'background-color: red; color: white; font-weight:bold;'
                return 'background-color: yellow; color: black;'
    
            st.write("Correlation matrix (daily-aggregated) between pollutants and weather/traffic variables:")
            st.dataframe(corr_data.style.applymap(highlight_corr))
    
            # ---------------------------
            # Plotly Heatmap with data points (fixed colorscale)
            # ---------------------------
            heatmap_z = corr_data.values
            annot_text = np.empty_like(heatmap_z, dtype=object)
            for i in range(heatmap_z.shape[0]):
                for j in range(heatmap_z.shape[1]):
                    val = heatmap_z[i, j]
                    annot_text[i, j] = f"{val:.2f}" if not np.isnan(val) else ""
    
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_z,
                x=corr_data.columns.tolist(),
                y=corr_data.index.tolist(),
                text=annot_text,
                texttemplate="%{text}",
                textfont={"size":12, "color":"black"},
                hovertemplate="Pollutant: %{y}<br>Variable: %{x}<br>Correlation: %{text}<extra></extra>",
                colorscale=[
                    [0.0, 'red'],      #  -1
                    [0.5, 'yellow'],   #   0
                    [1, 'green'],      #  +1
                ],
                zmin=-1, zmax=1,
                colorbar=dict(title="Correlation")
            ))
    
            fig_heatmap.update_layout(
                title="🌡️ Pollutant vs Weather & Traffic Correlation (Daily Aggregated)",
                xaxis_title="Weather & Traffic Variables",
                yaxis_title="Pollutants",
                autosize=True,
                height=520
            )
    
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
            # ---------------------------
            # Strong correlations table
            # ---------------------------
            strong_corrs = corr_data.stack(dropna=True).reset_index()
            strong_corrs.columns = ['Pollutant', 'Variable', 'Correlation']
            strong_corrs = strong_corrs[(strong_corrs['Correlation'] >= 0.6) | (strong_corrs['Correlation'] <= -0.6)]
            if not strong_corrs.empty:
                st.write("Strong correlations (>=0.6 or <=-0.6):")
                st.dataframe(strong_corrs.sort_values('Correlation', ascending=False))
            else:
                st.info("No strong correlations detected (>=0.6 or <=-0.6).")

import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import r2_score
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# Add to your imports at the beginning of the file
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler

if df is not None:
    with tabs[4]:
        st.subheader("📅 Forecasting — PM2.5 & NO₂ Adaptive Ensemble Forecast")
        forecast_pollutants = ['PM2.5', 'NO2']

        if not do_forecast:
            st.info("Forecasting computations are disabled. Enable in sidebar to run models.")
        else:
            for selected_pollutant in forecast_pollutants:
                
                # -----------------------------
                # GET ALL DATA FOR THIS POLLUTANT
                # -----------------------------
                df_all = df_filtered[df_filtered['Parameter'] == selected_pollutant].copy()
                if df_all.empty or len(df_all) < 50:
                    continue

                df_all = df_all.sort_values('Datetime')
                df_all['Datetime'] = pd.to_datetime(df_all['Datetime'])
                
                # -----------------------------
                # SEPARATE STRATEGY FOR EACH MODEL TYPE
                # -----------------------------
                
                # 1. FIND CONTINUOUS SEGMENT FOR LSTM
                df_all['date_diff'] = df_all['Datetime'].diff().dt.days
                break_points = df_all[df_all['date_diff'] > 1].index.tolist()
                
                segments = []
                start_idx = 0
                for break_idx in break_points:
                    segments.append(df_all.iloc[start_idx:break_idx])
                    start_idx = break_idx
                segments.append(df_all.iloc[start_idx:])
                
                longest_segment = max(segments, key=lambda x: len(x))
                
                # Check if LSTM is viable
                lstm_viable = len(longest_segment) >= log_adjust * 3
                df_lstm = longest_segment.copy() if lstm_viable else None
                
                # 2. RF/GB USE ALL DATA
                df_rfgb = df_all.copy()
                
                # -----------------------------
                # TRAIN RF AND GB ON ALL DATA
                # -----------------------------
                df_rfgb = df_rfgb.set_index('Datetime')
                df_rfgb = df_rfgb[~df_rfgb.index.duplicated(keep='first')]
                
                features = [f for f in weather_vars_in_data + traffic_vars_in_data if f in df_rfgb.columns]
                df_model_rfgb = df_rfgb[['Value'] + features].copy()
                
                if features:
                    df_model_rfgb[features] = df_model_rfgb[features].fillna(method='ffill').fillna(method='bfill')
                
                df_model_rfgb['Value'] = df_model_rfgb['Value'].interpolate(method='linear', limit=7)
                
                for lag in range(1, log_adjust + 1):
                    df_model_rfgb[f'Lag{lag}'] = df_model_rfgb['Value'].shift(lag)
                    df_model_rfgb[f'Lag{lag}'] = df_model_rfgb[f'Lag{lag}'].fillna(method='ffill')
                
                df_model_rfgb = df_model_rfgb.dropna().reset_index()
                
                if len(df_model_rfgb) < log_adjust + 20:
                    continue
                
                # Split for RF/GB
                split_idx_rfgb = int(0.8 * len(df_model_rfgb))
                X_rfgb = df_model_rfgb[[f'Lag{i}' for i in range(1, log_adjust + 1)] + features]
                y_rfgb = df_model_rfgb['Value']
                X_train_rfgb, X_val_rfgb = X_rfgb[:split_idx_rfgb], X_rfgb[split_idx_rfgb:]
                y_train_rfgb, y_val_rfgb = y_rfgb[:split_idx_rfgb], y_rfgb[split_idx_rfgb:]
                
                # Train RF
                rf_param_grid = {
                    'n_estimators': [100, 150],
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5]
                }
                rf = RandomForestRegressor(random_state=42)
                rf_random = RandomizedSearchCV(rf, rf_param_grid, n_iter=3, cv=KFold(3, shuffle=True, random_state=42),
                                               scoring='neg_mean_squared_error', n_jobs=-1)
                rf_random.fit(X_train_rfgb, y_train_rfgb)
                rf_final = rf_random.best_estimator_
                y_pred_rf = rf_final.predict(X_rfgb)
                y_pred_rf_val = rf_final.predict(X_val_rfgb)
                
                # Train GB
                gb_param_grid = {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
                gb = GradientBoostingRegressor(random_state=42)
                gb_random = RandomizedSearchCV(gb, gb_param_grid, n_iter=3, cv=KFold(3, shuffle=True, random_state=42),
                                               scoring='neg_mean_squared_error', n_jobs=-1)
                
                if X_train_rfgb.isna().any().any():
                    X_train_rfgb_filled = X_train_rfgb.fillna(X_train_rfgb.mean())
                    X_val_rfgb_filled = X_val_rfgb.fillna(X_train_rfgb.mean())
                    gb_random.fit(X_train_rfgb_filled, y_train_rfgb)
                    gb_final = gb_random.best_estimator_
                    y_pred_gb = gb_final.predict(X_rfgb)
                    y_pred_gb_val = gb_final.predict(X_val_rfgb_filled)
                else:
                    gb_random.fit(X_train_rfgb, y_train_rfgb)
                    gb_final = gb_random.best_estimator_
                    y_pred_gb = gb_final.predict(X_rfgb)
                    y_pred_gb_val = gb_final.predict(X_val_rfgb)
                
                # Calculate RF/GB metrics
                rf_mae = mean_absolute_error(y_val_rfgb, y_pred_rf_val)
                rf_rmse = mean_squared_error(y_val_rfgb, y_pred_rf_val, squared=False)
                rf_r2 = r2_score(y_val_rfgb, y_pred_rf_val)
                
                gb_mae = mean_absolute_error(y_val_rfgb, y_pred_gb_val)
                gb_rmse = mean_squared_error(y_val_rfgb, y_pred_gb_val, squared=False)
                gb_r2 = r2_score(y_val_rfgb, y_pred_gb_val)
                
                # Store predictions
                df_model_rfgb['RF_Pred'] = y_pred_rf
                df_model_rfgb['GB_Pred'] = y_pred_gb
                
                # -----------------------------
                # TRAIN LSTM
                # -----------------------------
                lstm_mae, lstm_rmse, lstm_r2 = np.nan, np.nan, np.nan
                lstm_predictions = {}
                lstm_performance_acceptable = False
                
                if lstm_viable and df_lstm is not None:
                    try:
                        df_lstm = df_lstm.set_index('Datetime')
                        df_lstm = df_lstm[~df_lstm.index.duplicated(keep='first')]
                        
                        lstm_features = [f for f in features if f in df_lstm.columns]
                        
                        if lstm_features:
                            df_model_lstm = df_lstm[['Value'] + lstm_features].copy()
                            df_model_lstm[lstm_features] = df_model_lstm[lstm_features].fillna(method='ffill').fillna(method='bfill')
                            use_features_in_lstm = True
                        else:
                            df_model_lstm = df_lstm[['Value']].copy()
                            use_features_in_lstm = False
                        
                        for lag in range(1, log_adjust + 1):
                            df_model_lstm[f'Lag{lag}'] = df_model_lstm['Value'].shift(lag)
                        
                        df_model_lstm = df_model_lstm.dropna()
                        
                        if len(df_model_lstm) > log_adjust + 10:
                            if use_features_in_lstm:
                                scaler = StandardScaler()
                                all_data = df_model_lstm.values
                                all_data_scaled = scaler.fit_transform(all_data)
                                
                                X_seq, y_seq = [], []
                                for i in range(log_adjust, len(all_data_scaled)):
                                    X_seq.append(all_data_scaled[i-log_adjust:i, :])
                                    y_seq.append(all_data_scaled[i, 0])
                                
                                X_seq = np.array(X_seq)
                                y_seq = np.array(y_seq)
                                input_shape = (log_adjust, df_model_lstm.shape[1])
                            else:
                                y_values = df_model_lstm['Value'].values
                                scaler = StandardScaler()
                                y_scaled = scaler.fit_transform(y_values.reshape(-1, 1))
                                
                                X_seq, y_seq = [], []
                                for i in range(log_adjust, len(y_scaled)):
                                    X_seq.append(y_scaled[i-log_adjust:i, 0])
                                    y_seq.append(y_scaled[i, 0])
                                
                                X_seq = np.array(X_seq).reshape(-1, log_adjust, 1)
                                y_seq = np.array(y_seq)
                                input_shape = (log_adjust, 1)
                            
                            split_idx_lstm = int(0.8 * len(X_seq))
                            X_train_lstm, X_val_lstm = X_seq[:split_idx_lstm], X_seq[split_idx_lstm:]
                            y_train_lstm, y_val_lstm = y_seq[:split_idx_lstm], y_seq[split_idx_lstm:]
                            
                            # Simple LSTM
                            lstm_model = Sequential()
                            lstm_model.add(LSTM(32, activation='relu', input_shape=input_shape))
                            lstm_model.add(Dense(1))
                            lstm_model.compile(optimizer='adam', loss='mse')
                            
                            lstm_model.fit(X_train_lstm, y_train_lstm, epochs=30, batch_size=16, verbose=0)
                            
                            y_pred_scaled = lstm_model.predict(X_seq, verbose=0)
                            
                            if use_features_in_lstm:
                                dummy_pred = np.zeros((len(y_pred_scaled), df_model_lstm.shape[1]))
                                dummy_pred[:, 0] = y_pred_scaled.flatten()
                                y_pred = scaler.inverse_transform(dummy_pred)[:, 0]
                                
                                dummy_actual = np.zeros((len(y_seq), df_model_lstm.shape[1]))
                                dummy_actual[:, 0] = y_seq
                                y_actual = scaler.inverse_transform(dummy_actual)[:, 0]
                            else:
                                y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
                                y_actual = scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()
                            
                            val_start = split_idx_lstm
                            y_val_actual = y_actual[val_start:]
                            y_val_pred = y_pred[val_start:]
                            
                            lstm_mae = mean_absolute_error(y_val_actual, y_val_pred)
                            lstm_rmse = mean_squared_error(y_val_actual, y_val_pred, squared=False)
                            lstm_r2 = r2_score(y_val_actual, y_val_pred)
                            
                            # Performance check
                            avg_rfgb_mae = (rf_mae + gb_mae) / 2
                            avg_rfgb_rmse = (rf_rmse + gb_rmse) / 2
                            
                            lstm_performance_acceptable = (
                                not np.isnan(lstm_mae) and 
                                not np.isnan(lstm_r2) and
                                lstm_mae < avg_rfgb_mae * 1.5 and
                                lstm_rmse < avg_rfgb_rmse * 1.5 and
                                lstm_r2 > 0
                            )
                            
                            if lstm_performance_acceptable:
                                pred_dates = df_model_lstm.index[log_adjust:log_adjust+len(y_pred)]
                                for date, pred in zip(pred_dates, y_pred):
                                    lstm_predictions[date] = pred
                                    
                    except:
                        pass
                
                # -----------------------------
                # CREATE ENSEMBLE
                # -----------------------------
                df_model_rfgb['LSTM_Pred'] = np.nan
                lstm_match_count = 0
                
                if lstm_performance_acceptable and lstm_predictions:
                    for idx, row in df_model_rfgb.iterrows():
                        date = row['Datetime']
                        if date in lstm_predictions:
                            df_model_rfgb.at[idx, 'LSTM_Pred'] = lstm_predictions[date]
                            lstm_match_count += 1
                
                # Simple ensemble: RF+GB average, add LSTM if available
                if lstm_performance_acceptable and lstm_match_count > 0:
                    df_model_rfgb['Ensemble'] = df_model_rfgb[['RF_Pred', 'GB_Pred']].mean(axis=1)
                    lstm_mask = df_model_rfgb['LSTM_Pred'].notna()
                    df_model_rfgb.loc[lstm_mask, 'Ensemble'] = df_model_rfgb.loc[lstm_mask, ['RF_Pred', 'GB_Pred', 'LSTM_Pred']].mean(axis=1)
                else:
                    df_model_rfgb['Ensemble'] = df_model_rfgb[['RF_Pred', 'GB_Pred']].mean(axis=1)
                
                # Calculate ensemble metrics
                ens_pred_val = df_model_rfgb.iloc[split_idx_rfgb:]['Ensemble'].values
                ens_mae = mean_absolute_error(y_val_rfgb, ens_pred_val)
                ens_rmse = mean_squared_error(y_val_rfgb, ens_pred_val, squared=False)
                ens_r2 = r2_score(y_val_rfgb, ens_pred_val)

		# -----------------------------
                # ⭐ FUTURE FORECAST SECTION ⭐
                # -----------------------------
                # First, ensure we have the necessary data
                if 'df_model_rfgb' in locals() and len(df_model_rfgb) > log_adjust:
                    try:
                        # Get the last 'log_adjust' rows for forecasting
                        last_features = df_model_rfgb.iloc[-log_adjust:][[f'Lag{i}' for i in range(1, log_adjust + 1)] + features].values
                        
                        future_forecast_vals = []
                        
                        # Calculate sigma values for confidence intervals
                        sigma_rf = np.std(y_rfgb - df_model_rfgb['RF_Pred'])
                        sigma_gb = np.std(y_rfgb - df_model_rfgb['GB_Pred'])
                        
                        # Get LSTM future predictions if available
                        lstm_future_series = []
                        sigma_lstm = 0
                        
                        if lstm_performance_acceptable and lstm_model is not None and 'df_model_lstm' in locals():
                            try:
                                # Get the last values from LSTM training data
                                if 'y_values' not in locals():
                                    # If y_values not available, use the last values from df_model_lstm
                                    y_values = df_model_lstm['Value'].values[-log_adjust*2:]  # Get enough data
                                
                                scaler = StandardScaler()
                                y_scaled = scaler.fit_transform(y_values.reshape(-1, 1))
                                
                                last_input = y_scaled[-log_adjust:].reshape(1, log_adjust, 1)
                                for _ in range(forecast_horizon):
                                    pred_scaled = lstm_model.predict(last_input, verbose=0)
                                    pred_val = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                                    lstm_future_series.append(pred_val)
                                    last_input = np.append(last_input[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
                                
                                # Calculate LSTM sigma
                                if lstm_predictions:
                                    actual_lstm_values = list(lstm_predictions.values())
                                    predicted_lstm_values = [df_model_rfgb.loc[df_model_rfgb['Datetime'] == date, 'Value'].values[0] 
                                                           for date in lstm_predictions.keys() 
                                                           if date in df_model_rfgb['Datetime'].values]
                                    if len(actual_lstm_values) == len(predicted_lstm_values):
                                        sigma_lstm = np.std(np.array(predicted_lstm_values) - np.array(actual_lstm_values))
                            except:
                                lstm_future_series = []
                                sigma_lstm = 0
                        
                        # Generate future forecasts
                        for i in range(forecast_horizon):
                            rf_pred = rf_final.predict(last_features)[0]
                            gb_pred = gb_final.predict(last_features)[0]
                            
                            # Include LSTM prediction if available
                            if lstm_future_series and i < len(lstm_future_series):
                                lstm_pred = lstm_future_series[i]
                                ensemble_pred = (rf_pred + gb_pred + lstm_pred) / 3
                            else:
                                ensemble_pred = (rf_pred + gb_pred) / 2
                            
                            future_forecast_vals.append(ensemble_pred)
                            
                            # Update features for next prediction
                            new_lag_row = np.append(last_features[0, 1:log_adjust], ensemble_pred)
                            if features:
                                new_lag_row = np.append(new_lag_row, last_features[0, log_adjust:])
                            last_features = new_lag_row.reshape(1, -1)
                        
                        # Calculate confidence intervals
                        if lstm_future_series and sigma_lstm > 0:
                            sigma_ensemble = np.sqrt((sigma_rf**2 + sigma_gb**2 + sigma_lstm**2) / 3)
                        else:
                            sigma_ensemble = np.sqrt((sigma_rf**2 + sigma_gb**2) / 2)
                        
                        ci_mult = stats.norm.ppf(0.975)
                        ci_upper = [v + ci_mult * sigma_ensemble for v in future_forecast_vals]
                        ci_lower = [v - ci_mult * sigma_ensemble for v in future_forecast_vals]
                        
                        # Create forecast dates
                        future_dates = pd.date_range(df_model_rfgb['Datetime'].max() + pd.Timedelta(days=1), 
                                                    periods=forecast_horizon, freq='D')
                        
                        # Get threshold
                        threshold_24h = nesaq_thresholds[selected_pollutant].get('24-hour', np.inf)
                        
                        # Create forecast dataframe
                        df_forecast = pd.DataFrame({
                            'Date': future_dates,
                            'Forecast': future_forecast_vals,
                            'CI_upper': ci_upper,
                            'CI_lower': ci_lower,
                            'Exceedance': [v > threshold_24h for v in future_forecast_vals]
                        })
                        
                        # -----------------------------
                        # ⭐ SHOW FORECAST PLOT FIRST ⭐
                        # -----------------------------
                        fig_f = go.Figure()
                        
                        # Forecast line
                        fig_f.add_trace(go.Scatter(
                            x=df_forecast['Date'], 
                            y=df_forecast['Forecast'], 
                            mode='lines+markers', 
                            name='Forecast',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Confidence interval
                        fig_f.add_trace(go.Scatter(
                            x=df_forecast['Date'].tolist() + df_forecast['Date'].tolist()[::-1],
                            y=df_forecast['CI_upper'].tolist() + df_forecast['CI_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence Interval',
                            showlegend=True
                        ))
                        
                        # Exceedance markers
                        if any(df_forecast['Exceedance']):
                            fig_f.add_trace(go.Scatter(
                                x=df_forecast[df_forecast['Exceedance']]['Date'],
                                y=df_forecast[df_forecast['Exceedance']]['Forecast'],
                                mode='markers', 
                                marker=dict(color='red', size=10, symbol='triangle-up'),
                                name='Exceedance Warning'
                            ))
                        
                        # Threshold line
                        if threshold_24h < np.inf:
                            fig_f.add_hline(
                                y=threshold_24h, 
                                line_dash='dash', 
                                line_color='red', 
                                annotation_text=f"Health Threshold: {threshold_24h}",
                                annotation_position="bottom right"
                            )
                        
                        # Layout
                        fig_f.update_layout(
                            title=f"{selected_pollutant} — {forecast_horizon}-Day Ensemble Forecast",
                            xaxis_title="Date", 
                            yaxis_title=f"{selected_pollutant} Concentration",
                            hovermode='x unified',
                            showlegend=True
                        )
                        
                        # Display the forecast plot
                        st.plotly_chart(fig_f, use_container_width=True)
                        
                        # Show forecast summary
                        exceedance_count = df_forecast['Exceedance'].sum()
                        if exceedance_count > 0:
                            st.warning(f"⚠️ **Alert:** {exceedance_count} of {forecast_horizon} days exceed health threshold")
                        else:
                            st.success(f"✅ **Good news:** No exceedances forecasted for next {forecast_horizon} days")
                        
                    except Exception as e:
                        st.error(f"Forecast generation failed: {str(e)}")
                        # Still show metrics even if forecast fails
                
                # -----------------------------
                # ⭐ SHOW METRICS TABLE ⭐
                # -----------------------------
                # Format LSTM metrics
                lstm_mae_str = f"{lstm_mae:.2f}" if not np.isnan(lstm_mae) else "N/A"
                lstm_rmse_str = f"{lstm_rmse:.2f}" if not np.isnan(lstm_rmse) else "N/A"
                lstm_r2_str = f"{lstm_r2:.2f}" if not np.isnan(lstm_r2) else "N/A"
                
                st.markdown(f"""
### 📊 Model Performance Metrics — {selected_pollutant}

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Random Forest | {rf_mae:.2f} | {rf_rmse:.2f} | {rf_r2:.2f} |
| Gradient Boosting | {gb_mae:.2f} | {gb_rmse:.2f} | {gb_r2:.2f} |
| LSTM | {lstm_mae_str} | {lstm_rmse_str} | {lstm_r2_str} |
| **Ensemble** | **{ens_mae:.2f}** | **{ens_rmse:.2f}** | **{ens_r2:.2f}** |
""")