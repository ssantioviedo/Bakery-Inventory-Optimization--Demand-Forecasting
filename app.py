import streamlit as st

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Bakery Inventory Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom color scheme
PRIMARY_COLOR = "#D21E54"
SECONDARY_COLORS = {
    'Prophet': '#D21E54',
    'SARIMAX': '#6A0DAD',
    'ETS': '#FF1493',
    'Seasonal Naive': '#9370DB'
}

# ============================================================================
# DATA LOADING & CACHING
# ============================================================================
@st.cache_data
def load_data():
    """Load and preprocess the bakery dataset"""
    df = pd.read_csv('bread basket.csv')
    
    # Data preprocessing
    df_grouped = df.copy()
    df_grouped['date_time'] = pd.to_datetime(df_grouped['date_time'], dayfirst=True)
    df_grouped['date_time'] = df_grouped['date_time'].dt.strftime('%d-%m-%Y')
    df_grouped = df_grouped.groupby(['date_time', 'Item']).size().reset_index(name='sales')
    
    # Feature engineering: flour equivalents
    flour_equivalents = {
        "Bread": 0.45, "Baguette": 0.35, "Cake": 0.30, "Muffin": 0.12,
        "Brownie": 0.10, "Pastry": 0.20, "Medialuna": 0.15, "Cookies": 0.08,
        "Scone": 0.10, "Empanadas": 0.18, "Crepes": 0.05
    }
    
    df_grouped['flour_kg'] = df_grouped['Item'].map(flour_equivalents).fillna(0)
    df_grouped['flour_used'] = np.round(df_grouped['sales'] * df_grouped['flour_kg'], 2)
    df_grouped = df_grouped[df_grouped['flour_used'] > 0]
    
    # Aggregate to daily flour demand
    df_fc = df_grouped.groupby('date_time')['flour_used'].sum().reset_index()
    df_fc['flour_used'] = np.ceil(df_fc['flour_used'])
    df_fc = df_fc.rename(columns={'date_time': 'ds', 'flour_used': 'y'})
    df_fc['ds'] = pd.to_datetime(df_fc['ds'], dayfirst=True)
    df_fc = df_fc.sort_values('ds').set_index('ds').asfreq('D', fill_value=0).reset_index()
    
    return df, df_grouped, df_fc

# ============================================================================
# FORECASTING MODELS
# ============================================================================
def fit_predict_prophet(train_df, h):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False, 
                growth='linear', changepoint_prior_scale=0.05)
    m.fit(train_df)
    future_dates = pd.date_range(start=train_df['ds'].max() + pd.Timedelta(days=1), periods=h, freq='D')
    future = m.make_future_dataframe(periods=h, freq='D')
    fc = m.predict(future)
    fc_test = fc[['ds', 'yhat']].set_index('ds').reindex(future_dates).reset_index(drop=True)
    return fc_test['yhat'].reset_index(drop=True)

def fit_predict_sarimax(train_df, h):
    model = SARIMAX(train_df['y'], order=(1,1,1), seasonal_order=(1,1,1,7), 
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=h).predicted_mean
    preds = np.maximum(np.asarray(preds), 0)
    return pd.Series(preds).reset_index(drop=True)

def fit_predict_ets(train_df, h):
    model = ExponentialSmoothing(train_df['y'], trend=None, seasonal='add', 
                                 seasonal_periods=7, initialization_method='estimated')
    res = model.fit(optimized=True, use_brute=True)
    preds = np.maximum(res.forecast(h).values, 0)
    return pd.Series(preds).reset_index(drop=True)

def fit_predict_seasonal_naive(train_df, h, period=7):
    last_season = train_df['y'].iloc[-period:].values
    repeats = int(np.ceil(h / period))
    fc = np.tile(last_season, repeats)[:h]
    return pd.Series(fc).reset_index(drop=True)

# ============================================================================
# METRICS FUNCTIONS
# ============================================================================
def _smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    with np.errstate(divide='ignore', invalid='ignore'):
        smape_vals = np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / denom)
    return np.nanmean(smape_vals) * 100

def rolling_backtest(df, model_funcs, horizon=30, n_splits=5):
    total_days = len(df)
    if total_days < horizon * (n_splits + 1):
        raise ValueError("Not enough data for requested n_splits and horizon")

    records = []
    last_preds = {}
    last_test = None
    EPS = 1e-8

    for i in range(n_splits):
        train_end = total_days - horizon * (n_splits - i)
        train = df.iloc[:train_end].reset_index(drop=True)
        test = df.iloc[train_end:train_end + horizon].reset_index(drop=True)
        y_true = test['y'].reset_index(drop=True)
        denom = np.sum(np.abs(y_true))

        for name, func in model_funcs.items():
            y_pred = func(train, horizon)
            y_pred = pd.Series(y_pred).reset_index(drop=True)[:len(y_true)].clip(lower=0)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_vals = np.abs((y_true - y_pred) / y_true)
                mape = np.nanmean(np.where(np.isfinite(mape_vals), mape_vals, np.nan)) * 100
            smape = _smape(y_true.values, y_pred.values)
            wape = (np.sum(np.abs(y_true - y_pred)) / max(denom, EPS) * 100)

            records.append({
                'model': name,
                'fold': i + 1,
                'mae': mae,
                'rmse': rmse,
                'mape_%': mape,
                'wape_%': wape,
                'smape_%': smape
            })

            if i == (n_splits - 1):
                last_preds[name] = y_pred
                last_test = test

    metrics_df = pd.DataFrame.from_records(records)
    summary = metrics_df.groupby('model')[['mae', 'rmse', 'mape_%', 'wape_%', 'smape_%']].mean().reset_index()
    return metrics_df, summary, last_preds, last_test

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================
def ets_bootstrap_mc(fit_model, residuals, lead_time, n_sims=10000, seed=42):
    np.random.seed(seed)
    base_forecast = fit_model.forecast(steps=lead_time).values
    cum_forecasts = np.zeros(n_sims)
    day_forecasts = np.zeros((n_sims, lead_time))
    
    for sim in range(n_sims):
        sampled_resids = np.random.choice(residuals.values, size=lead_time, replace=True)
        sim_forecast = base_forecast + sampled_resids
        sim_forecast = np.maximum(sim_forecast, 0)
        day_forecasts[sim, :] = sim_forecast
        cum_forecasts[sim] = sim_forecast.sum()
        
    return cum_forecasts, day_forecasts

# ============================================================================
# UI FUNCTIONS (PLOTLY - INTERACTIVE)
# ============================================================================
def plot_historical_demand(df_fc):
    """Historical flour demand - Interactive Plotly version"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_fc['ds'],
        y=df_fc['y'],
        mode='lines',
        name='Flour demand',
        line=dict(color=PRIMARY_COLOR, width=2.5),
        fill='tozeroy',
        fillcolor='rgba(210, 30, 84, 0.15)'
    ))
    
    fig.update_layout(
        title=dict(text='Historical Flour Demand', font=dict(size=16, color='black')),
        xaxis_title='Date',
        yaxis_title='Flour Used (kg)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400
    )
    
    return fig

def plot_dow_demand(df_fc):
    """Average demand by day of week - Interactive Plotly version"""
    df_dow = df_fc.copy()
    df_dow['day_of_week'] = df_dow['ds'].dt.day_name()
    df_dow['dow_num'] = df_dow['ds'].dt.dayofweek
    
    dow_demand = df_dow.groupby(['dow_num', 'day_of_week'])['y'].agg(['mean', 'std']).reset_index()
    dow_demand = dow_demand.sort_values('dow_num')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dow_demand['day_of_week'],
        y=dow_demand['mean'],
        name='Average Demand',
        marker_color=PRIMARY_COLOR,
        error_y=dict(type='data', array=dow_demand['std'], visible=True, color='black', thickness=1.5)
    ))
    
    fig.update_layout(
        title=dict(text='Average Daily Flour Demand by Day of Week', font=dict(size=14)),
        xaxis_title='Day of Week',
        yaxis_title='Average Demand (kg)',
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    return fig, dow_demand

def plot_model_comparison(last_test, last_preds):
    """Model comparison for last test period - Interactive Plotly version"""
    fig = go.Figure()
    
    # Actual demand
    fig.add_trace(go.Scatter(
        x=last_test['ds'],
        y=last_test['y'],
        mode='lines+markers',
        name='Actual demand',
        line=dict(color='black', width=3),
        marker=dict(size=8, color='white', line=dict(color='black', width=2))
    ))
    
    # Model predictions
    for name, preds in last_preds.items():
        color = SECONDARY_COLORS.get(name, '#888888')
        y_vals = preds.values if hasattr(preds, 'values') else preds
        fig.add_trace(go.Scatter(
            x=last_test['ds'],
            y=y_vals,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color)
        ))
    
    fig.update_layout(
        title=dict(text=f'Model Comparison: Last {len(last_test)} Days', font=dict(size=16)),
        xaxis_title='Date',
        yaxis_title='Flour Used (kg)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450
    )
    
    return fig

def plot_probabilistic_forecast(day_forecasts, cum_forecasts, lead_time, custom_percentiles):
    """Probabilistic forecast with fan chart and cumulative distribution - Two separate figures"""
    
    # ============== FIGURE 1: Fan Chart (Left) ==============
    fig1 = go.Figure()
    
    daily_quantiles = {
        'q05': np.quantile(day_forecasts, 0.05, axis=0),
        'q25': np.quantile(day_forecasts, 0.25, axis=0),
        'q50': np.quantile(day_forecasts, 0.50, axis=0),
        'q75': np.quantile(day_forecasts, 0.75, axis=0),
        'q95': np.quantile(day_forecasts, 0.95, axis=0),
    }
    days = list(range(1, lead_time + 1))
    
    # 90% band
    fig1.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(daily_quantiles['q95']) + list(daily_quantiles['q05'][::-1]),
        fill='toself',
        fillcolor='rgba(210, 30, 84, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% of scenarios'
    ))
    
    # 50% band
    fig1.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(daily_quantiles['q75']) + list(daily_quantiles['q25'][::-1]),
        fill='toself',
        fillcolor='rgba(210, 30, 84, 0.35)',
        line=dict(color='rgba(255,255,255,0)'),
        name='50% of scenarios'
    ))
    
    # Median line
    fig1.add_trace(go.Scatter(
        x=days,
        y=daily_quantiles['q50'],
        mode='lines+markers',
        name='Median',
        line=dict(color=PRIMARY_COLOR, width=2.5),
        marker=dict(size=8, color=PRIMARY_COLOR)
    ))
    
    fig1.update_layout(
        template='plotly_white',
        title=dict(text='Daily Demand: Probabilistic Forecast', font=dict(size=14, weight='bold')),
        height=500,
        margin=dict(t=60, b=100, l=60, r=40),
        xaxis_title='Day',
        yaxis_title='Daily Flour Demand (kg)',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='lightgray',
            borderwidth=1
        )
    )
    
    # ============== FIGURE 2: Histogram (Right) ==============
    fig2 = go.Figure()
    
    # Histogram
    fig2.add_trace(go.Histogram(
        x=cum_forecasts,
        nbinsx=60,
        name='Simulations',
        marker_color=PRIMARY_COLOR,
        marker_line_color='white',
        marker_line_width=0.5,
        opacity=0.7
    ))
    
    # Default service levels
    default_levels = {95: '#1f77b4', 98: '#ff7f0e', 99: '#2ca02c'}
    y_positions = {95: 0.95, 98: 0.80, 99: 0.65}
    ax_offsets = {95: 25, 98: 35, 99: 45}
    
    for level, color in default_levels.items():
        q_val = np.quantile(cum_forecasts, level / 100.0)
        
        fig2.add_vline(
            x=q_val,
            line=dict(color=color, width=2.5, dash='dash'),
        )
        
        fig2.add_annotation(
            x=q_val,
            y=y_positions[level],
            yref='paper',
            text=f'<b>{level}%</b><br>{q_val:.0f} kg',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=color,
            ax=ax_offsets[level],
            ay=-15,
            font=dict(size=10, color=color),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=color,
            borderwidth=1,
            borderpad=3
        )
        
        fig2.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name=f'SL {level}%',
            line=dict(color=color, width=2.5, dash='dash')
        ))
    
    # Mean line
    mean_val = cum_forecasts.mean()
    fig2.add_vline(
        x=mean_val,
        line=dict(color='black', width=2.5),
    )
    
    fig2.add_annotation(
        x=mean_val,
        y=0.50,
        yref='paper',
        text=f'<b>Mean</b><br>{mean_val:.0f} kg',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor='black',
        ax=-35,
        ay=-20,
        font=dict(size=10, color='black'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=1,
        borderpad=3
    )
    
    fig2.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name=f'Mean',
        line=dict(color='black', width=2.5)
    ))
    
    # Custom service levels with contrasting colors
    custom_colors = [
        '#7B241C',  # Dark Red
        '#1A5276',  # Dark Blue
        '#145A32',  # Dark Green
        '#6C3483',  # Dark Purple
        '#B9770E',  # Dark Orange
    ]
    
    non_default = [p for p in custom_percentiles if p not in [50, 95, 98, 99]]
    
    for i, level in enumerate(non_default):
        q_val = np.quantile(cum_forecasts, level / 100.0)
        color = custom_colors[i % len(custom_colors)]
        
        fig2.add_vline(
            x=q_val,
            line=dict(color=color, width=3, dash='dot'),
        )
        
        # Annotation for custom levels
        fig2.add_annotation(
            x=q_val,
            y=0.12 + (i * 0.12),
            yref='paper',
            text=f'<b>★ {level}%</b><br>{q_val:.0f} kg',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=color,
            ax=-40,
            ay=20,
            font=dict(size=11, color=color, weight='bold'),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor=color,
            borderwidth=2,
            borderpad=4
        )
        
        fig2.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name=f'★ SL {level}%',
            line=dict(color=color, width=3, dash='dot')
        ))
    
    fig2.update_layout(
        template='plotly_white',
        title=dict(text=f'{lead_time}-Day Cumulative Demand Distribution', font=dict(size=14, weight='bold')),
        height=500,
        margin=dict(t=60, b=60, l=60, r=120),
        xaxis_title='Cumulative Flour Demand (kg)',
        yaxis_title='Frequency (# of simulations)',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='lightgray',
            borderwidth=1,
            title=dict(text='<b>Service Levels</b>')
        )
    )
    
    return fig1, fig2

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown("""
        <style>
        .main-header {
            font-size: 3em;
            font-weight: bold;
            color: #D21E54;
            margin-bottom: 0.5rem;
        }
        .subheader {
            font-size: 1.3em;
            color: #666;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">Bakery Inventory Optimization</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Demand Forecasting & Reorder Point Recommendations</p>', unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Select a section:",
            ["Overview", "EDA", "Historical Demand", "Model Evaluation", "Inventory Recommendations"],
            label_visibility="collapsed"
        )

    # Load data
    df, df_grouped, df_fc = load_data()

    # ========================================================================
    # PAGE: OVERVIEW
    # ========================================================================
    if page == "Overview":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(" Total Days of Data", f"{(df_fc['ds'].max() - df_fc['ds'].min()).days}")
        with col2:
            st.metric(" Unique Products", f"{df_grouped['Item'].nunique()}")
        with col3:
            st.metric(" Total Flour Used (kg)", f"{df_fc['y'].sum():.0f}")

        st.markdown("---")
        st.markdown("""
        ### Project Overview
        
        This interactive dashboard transforms raw bakery sales data into actionable inventory management insights.
        
        **Key Features:**
        -  **Data Pipeline**: From raw transactions to daily flour demand
        -  **Exploratory Analysis**: Identify top-selling products and demand patterns
        -  **Multi-Model Forecasting**: Compare ETS, SARIMAX, Prophet, and Seasonal Naive
        -  **Probabilistic Recommendations**: Monte Carlo simulation for safety stock optimization
        
        **Workflow:**
        1. Aggregate product sales to flour demand
        2. Evaluate multiple forecasting models
        3. Generate probabilistic demand scenarios
        4. Recommend reorder points for different service levels
        """)

    # ========================================================================
    # PAGE: EDA
    # ========================================================================
    elif page == "EDA":
        st.markdown("### Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top-Selling Products (by count)")
            sku_sales = df_grouped.groupby('Item')['sales'].sum().sort_values(ascending=False).head(10)
            
            # Interactive Plotly horizontal bar chart
            fig_sku = go.Figure(go.Bar(
                x=sku_sales.values,
                y=sku_sales.index,
                orientation='h',
                marker_color=PRIMARY_COLOR
            ))
            fig_sku.update_layout(
                template='plotly_white',
                yaxis=dict(autorange="reversed"),
                xaxis_title='Total Sales',
                height=400
            )
            st.plotly_chart(fig_sku, use_container_width=True)
        
        with col2:
            st.markdown("#### Average Demand by Day of Week")
            fig_dow, dow_demand = plot_dow_demand(df_fc)
            st.plotly_chart(fig_dow, use_container_width=True)
        
        # Demand pattern metrics
        st.markdown("---")
        st.markdown("#### Demand Pattern Insights")
        
        # Calculate metrics
        df_dow = df_fc.copy()
        df_dow['dow_num'] = df_dow['ds'].dt.dayofweek
        weekday_avg = df_dow[df_dow['dow_num'].isin([0, 1, 2, 3, 4])]['y'].mean()
        weekend_avg = df_dow[df_dow['dow_num'].isin([5, 6])]['y'].mean()
        peak_day = dow_demand.loc[dow_demand['mean'].idxmax()]
        low_day = dow_demand.loc[dow_demand['mean'].idxmin()]
        pct_difference = ((peak_day['mean'] - low_day['mean']) / low_day['mean']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Weekday Avg", f"{weekday_avg:.1f} kg")
        with col2:
            st.metric("Weekend Avg", f"{weekend_avg:.1f} kg")
        with col3:
            st.metric("Peak Day", f"{peak_day['day_of_week']} ({peak_day['mean']:.1f} kg)")
        with col4:
            st.metric("Peak % Higher (vs Lower)", f"{pct_difference:.1f}%")

        st.markdown("---")
        st.markdown("#### Product Details")
        df_summary = df_grouped.groupby('Item').agg({
            'sales': 'sum',
            'flour_used': 'sum'
        }).sort_values('flour_used', ascending=False).round(2)
        st.dataframe(df_summary, use_container_width=True)

    # ========================================================================
    # PAGE: HISTORICAL DEMAND
    # ========================================================================
    elif page == "Historical Demand":
        st.markdown("### Historical Flour Demand")
        st.write("Time series of daily flour usage from raw bakery transactions.")
        
        fig = plot_historical_demand(df_fc)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Daily Demand", f"{df_fc['y'].mean():.1f} kg")
        with col2:
            st.metric("Std Dev", f"{df_fc['y'].std():.1f} kg")
        with col3:
            st.metric("Max Daily Demand", f"{df_fc['y'].max():.0f} kg")

    # ========================================================================
    # PAGE: MODEL EVALUATION
    # ========================================================================
    elif page == "Model Evaluation":
        st.markdown("###  Time Series Model Comparison")
        st.write("Expanding-window backtesting across multiple forecasting models.")
        
        # Model selection
        col1, col2 = st.columns(2)
        with col1:
            test_days = st.slider("Test horizon (days):", 7, 30, 14)
        with col2:
            n_splits = st.slider("Number of test folds:", 3, 8, 5)
        
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Training models and running backtest..."):
                models = {
                    'Prophet': fit_predict_prophet,
                    'SARIMAX': fit_predict_sarimax,
                    'ETS': fit_predict_ets,
                    'Seasonal Naive': fit_predict_seasonal_naive
                }
                
                metrics_df, summary_means, last_preds, last_test = rolling_backtest(
                    df_fc, models, horizon=test_days, n_splits=n_splits
                )
                
                st.markdown("#### Model Performance Summary")
                st.dataframe(summary_means.round(2), use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### Model Predictions vs Actual (Last Test Period)")
                fig = plot_model_comparison(last_test, last_preds)
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # PAGE: INVENTORY RECOMMENDATIONS
    # ========================================================================
    elif page == "Inventory Recommendations":
        st.markdown("###  Probabilistic Reorder Point (ROP) Recommendations")
        
        # Explanatory section
        with st.expander("ℹ️ How Reorder Points Work", expanded=False):
            st.markdown("""
            **Reorder Point (ROP)** is the inventory level at which you should place a new order.
            
            The ROP depends on:
            - **Lead time**: Days between placing an order and receiving it
            - **Service level**: Your acceptable stockout risk (higher % = less stockout risk but more inventory)
            
            **What the numbers mean:**
            - **50% (Mean)**: Expected average demand - use if you accept frequent stockouts
            - **95%**: Common choice - covers 95% of scenarios, balances cost vs. stockout risk
            - **98%**: High service level - recommended for critical products
            - **99%**: Very high safety - minimal stockout risk, but higher holding costs
            
            **Safety Stock** = Recommended Stock - Mean
            This extra inventory protects against demand variability during lead time.
            """)
        
        st.write("Customize your Monte Carlo simulation parameters below.")
        
        col1, col2 = st.columns(2)
        with col1:
            lead_time = st.slider("Lead time (days):", 1, 10, 4)
        with col2:
            n_sims = st.slider("Number of simulations:", 1000, 20000, 10000, step=1000)
        
        # Custom service levels
        st.markdown("---")
        st.markdown("#### Service Levels to Display")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            custom_percentiles = st.multiselect(
                "Select custom service levels (%):",
                options=list(range(50, 100)),
                default=[50, 95, 98, 99],
                help="Choose any percentile from 50% to 99%"
            )
        with col2:
            st.markdown("")  # Spacing
            if st.button("🔄 Reset to Defaults", help="Reset to 50%, 95%, 98%, 99%"):
                custom_percentiles = [50, 95, 98, 99]
        
        # Ensure custom_percentiles are sorted and contain at least 50
        if not custom_percentiles:
            st.warning("⚠️ Please select at least one service level")
            custom_percentiles = [50]
        custom_percentiles = sorted(set(custom_percentiles))  # Remove duplicates and sort
        
        if st.button(" Generate Recommendations", type="primary"):
            with st.spinner("Running Monte Carlo simulation..."):
                # Fit ETS model
                model = ExponentialSmoothing(
                    df_fc['y'],
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method='estimated'
                )
                fit = model.fit(optimized=True, use_brute=True)
                residuals = fit.resid.dropna()
                
                # Run MC simulation
                cum_forecasts, day_forecasts = ets_bootstrap_mc(
                    fit, residuals, lead_time=lead_time, n_sims=n_sims
                )
                
                # Display results
                st.markdown("---")
                st.markdown("#### Probabilistic Forecast Visualization")
                fig1, fig2 = plot_probabilistic_forecast(day_forecasts, cum_forecasts, lead_time, custom_percentiles)

                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("---")
                st.markdown(f"#### Recommended Reorder Point (ROP) — {lead_time}-Day Lead Time")
                
                # Calculate quantiles for selected service levels
                percentiles_normalized = [p / 100.0 for p in custom_percentiles]
                quantiles = np.quantile(cum_forecasts, percentiles_normalized)
                
                # Create table with service level labels
                service_level_labels = []
                for p in custom_percentiles:
                    if p == 50:
                        service_level_labels.append("Mean (50%)")
                    else:
                        service_level_labels.append(f"{p}%")
                
                mean_forecast = cum_forecasts.mean()
                
                summary_table = pd.DataFrame({
                    'Service Level': service_level_labels,
                    'Recommended Stock (kg)': [f"{q:.1f}" for q in quantiles],
                    'Safety Stock (vs Mean)': [
                        f"{0:.1f}" if p == 50 else f"{q - mean_forecast:.1f}"
                        for p, q in zip(custom_percentiles, quantiles)
                    ]
                })
                
                st.dataframe(summary_table, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Forecast", f"{mean_forecast:.1f} kg")
                with col2:
                    st.metric("Std Dev", f"{cum_forecasts.std():.1f} kg")
                with col3:
                    max_recommended = quantiles[-1]
                    st.metric("Max Recommended", f"{max_recommended:.1f} kg")
                
                
                st.markdown("---")
                st.markdown("### How to Choose Your Service Level")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Lower Service Levels (50-80%)**
                    - ⬆️ Higher stockout risk
                    - ⬇️ Lowest holding costs
                    - Use: Non-critical items, low demand variability
                    """)
                
                with col2:
                    st.markdown("""
                    **Higher Service Levels (95-99%)**
                    - ⬇️ Lower stockout risk
                    - ⬆️ Higher holding costs
                    - Use: Critical products, seasonal items
                    """)
                
                st.markdown("""
                **Industry Benchmarks:**
                - **95%**: Standard for most retail (popular choice)
                - **98%**: Critical items, perishables
                - **99%**: High-value inventory, customer-facing items
                """)


if __name__ == "__main__":
    main()