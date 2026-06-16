"""
Tesla (TSLA) Stock Price Prediction — Streamlit App
Author : Gopi Mahith
Deploy : streamlit run app.py
"""

import os, warnings, pickle
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 20px; border-radius: 12px; color: white;
    text-align: center; margin: 5px;
}
.metric-val { font-size: 2rem; font-weight: 700; }
.metric-lbl { font-size: 0.85rem; opacity: 0.85; }
.winner-badge {
    background: #27ae60; color: white; padding: 4px 12px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 600;
}
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_engineer(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True); df.reset_index(drop=True, inplace=True)
    price_cols = ['Open','High','Low','Close','Adj Close']
    df[price_cols] = df[price_cols].ffill()
    df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(5,min_periods=1).median())
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Range']  = df['High'] - df['Low']
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Volume_MA_5']  = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+gain/(loss+1e-10)))
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2*df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2*df['BB_Std']
    df['BB_Width'] = (df['BB_Upper']-df['BB_Lower'])/df['BB_Mid']
    ema12 = df['Close'].ewm(span=12,adjust=False).mean()
    ema26 = df['Close'].ewm(span=26,adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df.dropna(inplace=True); df.reset_index(drop=True, inplace=True)
    return df

FEATS = ['Close','Open','High','Low','Volume','MA_20','MA_50','RSI','MACD','BB_Width','Price_Range','Daily_Return','Volume_Ratio']
CI    = FEATS.index('Close')
NF    = len(FEATS)
LB    = 60

def make_sequences(data, lb=60, ci=0):
    X, y = [], []
    for i in range(lb, len(data)):
        X.append(data[i-lb:i]); y.append(data[i,ci])
    return np.array(X), np.array(y)

def forecast_n(model, X_seed, n_days, cs, ci=CI):
    preds, cur = [], X_seed.copy()
    for _ in range(n_days):
        p = float(model.predict(cur, verbose=0)[0,0])
        preds.append(p)
        ns = cur[0,-1,:].copy(); ns[ci]=p
        cur = np.concatenate([cur[:,1:,:], ns.reshape(1,1,-1)], axis=1)
    return cs.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=80)
    st.title("🚗 TSLA Predictor")
    st.markdown("---")

    uploaded = st.file_uploader("📁 Upload TSLA.csv", type=['csv'])
    if uploaded:
        with open("TSLA_upload.csv","wb") as f: f.write(uploaded.read())
        CSV_PATH = "TSLA_upload.csv"
    else:
        CSV_PATH = "TSLA.csv"

    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    horizon = st.selectbox("Forecast Horizon", [1, 5, 10], index=0,
                           help="Number of days to forecast ahead")
    lookback_disp = st.info(f"📅 Look-back window: **{LB} days**")

    st.markdown("---")
    st.subheader("🎨 Chart Options")
    show_ma    = st.checkbox("Show Moving Averages", value=True)
    show_bb    = st.checkbox("Show Bollinger Bands", value=True)
    show_vol   = st.checkbox("Show Volume Chart",    value=True)

    st.markdown("---")
    st.caption("Built by **Gopi Mahith** | Parul University 2026")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🚗 Tesla (TSLA) Stock Price Prediction")
st.markdown("**Deep Learning Forecasting** using SimpleRNN & LSTM | Domain: Financial Services")
st.markdown("---")

# Load data
try:
    df = load_and_engineer(CSV_PATH)
except Exception as e:
    st.error(f"❌ Could not load data: {e}\nPlease upload TSLA.csv via the sidebar.")
    st.stop()

# Load models & scalers
@st.cache_resource
def load_models_and_scalers(df):
    try:
        import tensorflow as tf
        scaler = MinMaxScaler()
        cs     = MinMaxScaler()
        ts     = int(len(df) * 0.80)
        data   = df[FEATS].values
        scaler.fit(data[:ts])
        cs.fit(df[['Close']].values[:ts])
        tr_s = scaler.transform(data[:ts])
        te_s = scaler.transform(data[ts:])
        Xtr,ytr = make_sequences(tr_s,LB,CI)
        Xte,yte = make_sequences(te_s,LB,CI)
        rnn  = tf.keras.models.load_model('simple_rnn_tsla.keras')
        lstm = tf.keras.models.load_model('best_lstm_tsla.keras')
        rnn_p  = rnn.predict(Xte,  verbose=0).flatten()
        lstm_p = lstm.predict(Xte, verbose=0).flatten()
        ytrue  = cs.inverse_transform(yte.reshape(-1,1)).flatten()
        ypR    = cs.inverse_transform(rnn_p.reshape(-1,1)).flatten()
        ypL    = cs.inverse_transform(lstm_p.reshape(-1,1)).flatten()
        seed   = Xte[[-1]]
        return scaler, cs, rnn, lstm, ytrue, ypR, ypL, seed, ts
    except Exception as ex:
        return None, None, None, None, None, None, None, None, None

scaler, cs, rnn, lstm, ytrue, ypR, ypL, seed, ts = load_models_and_scalers(df)
models_ok = rnn is not None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Overview",
    "🤖 Model Results",
    "📅 Forecast",
    "📊 Comparison",
    "📋 Data Explorer"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Market Overview
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📈 TSLA Market Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    cur_price = df['Close'].iloc[-1]
    prev_price= df['Close'].iloc[-2]
    chg = cur_price - prev_price
    pct_chg = chg / prev_price * 100

    col1.metric("Latest Close",  f"${cur_price:,.2f}", f"{chg:+.2f} ({pct_chg:+.2f}%)")
    col2.metric("All-Time High", f"${df['Close'].max():,.2f}")
    col3.metric("All-Time Low",  f"${df['Close'].min():,.2f}")
    col4.metric("Mean Daily Ret", f"{df['Daily_Return'].mean()*100:.4f}%")
    col5.metric("Daily Volatility", f"{df['Daily_Return'].std()*100:.4f}%")

    # Price chart
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Date'], df['Close'], color='#1f77b4', linewidth=1.2, label='Close')
    if show_ma:
        ax.plot(df['Date'], df['MA_20'], color='#ff7f0e', linewidth=1.2, label='MA-20', alpha=0.9)
        ax.plot(df['Date'], df['MA_50'], color='#2ca02c', linewidth=1.2, label='MA-50', alpha=0.9)
    if show_bb:
        ax.fill_between(df['Date'], df['BB_Lower'], df['BB_Upper'], alpha=0.1, color='purple', label='Bollinger Band')
    ax.set_title('TSLA Historical Close Price', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (USD)'); ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if show_vol:
        fig2, ax2 = plt.subplots(figsize=(14, 3))
        ax2.bar(df['Date'], df['Volume']/1e6, color='#ff7f0e', alpha=0.7, width=1)
        ax2.set_title('Trading Volume (Millions)', fontweight='bold')
        ax2.set_ylabel('Volume (M)'); ax2.set_xlabel('Date')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    # RSI
    st.subheader("📉 RSI — Momentum Indicator")
    fig3, ax3 = plt.subplots(figsize=(14, 3))
    ax3.plot(df['Date'], df['RSI'], color='#9467bd', linewidth=1)
    ax3.axhline(70, color='red',   linestyle='--', linewidth=1, label='Overbought (70)')
    ax3.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    ax3.fill_between(df['Date'], 30, 70, alpha=0.05, color='gray')
    ax3.set_ylabel('RSI'); ax3.set_ylim(0, 100); ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout(); st.pyplot(fig3); plt.close()

# ════════════════════════════════════════════════════════════════
# TAB 2 — Model Results
# ════════════════════════════════════════════════════════════════
with tab2:
    if not models_ok:
        st.warning("⚠️ Models not found. Run the notebook first to train and save models.")
    else:
        st.subheader("🤖 Model Predictions on Test Set")

        def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
        def mae(a,b):  return float(mean_absolute_error(a,b))
        def mape(a,b): return float(np.mean(np.abs((a-b)/(a+1e-8)))*100)
        def r2(a,b):   return float(r2_score(a,b))

        mR = {'RMSE':rmse(ytrue,ypR),'MAE':mae(ytrue,ypR),'MAPE':mape(ytrue,ypR),'R2':r2(ytrue,ypR)}
        mL = {'RMSE':rmse(ytrue,ypL),'MAE':mae(ytrue,ypL),'MAPE':mape(ytrue,ypL),'R2':r2(ytrue,ypL)}

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### SimpleRNN Predictions")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("RMSE",  f"${mR['RMSE']:,.0f}")
            c2.metric("MAE",   f"${mR['MAE']:,.0f}")
            c3.metric("MAPE",  f"{mR['MAPE']:.1f}%")
            c4.metric("R²",    f"{mR['R2']:.4f}")
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot(ytrue, label='Actual',    color='#1f77b4', linewidth=1.5)
            ax.plot(ypR,   label='SimpleRNN', color='#e74c3c', linewidth=1.3, alpha=0.85, linestyle='--')
            ax.set_title('SimpleRNN: Actual vs Predicted', fontweight='bold')
            ax.set_xlabel('Test Step'); ax.set_ylabel('Price (USD)'); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### LSTM Predictions")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("RMSE",  f"${mL['RMSE']:,.0f}")
            c2.metric("MAE",   f"${mL['MAE']:,.0f}")
            c3.metric("MAPE",  f"{mL['MAPE']:.1f}%")
            c4.metric("R²",    f"{mL['R2']:.4f}")
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot(ytrue, label='Actual', color='#1f77b4', linewidth=1.5)
            ax.plot(ypL,   label='LSTM',   color='#2ca02c', linewidth=1.3, alpha=0.85, linestyle='-.')
            ax.set_title('LSTM: Actual vs Predicted', fontweight='bold')
            ax.set_xlabel('Test Step'); ax.set_ylabel('Price (USD)'); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### 🔀 Overlay — All Models vs Actual")
        fig,ax=plt.subplots(figsize=(14,5))
        ax.plot(ytrue, label='Actual',    color='#1f77b4', linewidth=2)
        ax.plot(ypR,   label='SimpleRNN', color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.85)
        ax.plot(ypL,   label='LSTM',      color='#2ca02c', linewidth=1.5, linestyle='-.', alpha=0.85)
        ax.set_title('All Models — Test Set Overlay', fontsize=13, fontweight='bold')
        ax.set_xlabel('Test Timestep'); ax.set_ylabel('Price (USD)'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════
# TAB 3 — Forecast
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"📅 {horizon}-Day Ahead Forecast")
    if not models_ok:
        st.warning("⚠️ Models not loaded. Train models first via the notebook.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🔮 Generate {horizon}-Day Forecast", type="primary", use_container_width=True):
                with st.spinner("Running iterative forecast..."):
                    rnn_fc  = forecast_n(rnn,  seed, horizon, cs)
                    lstm_fc = forecast_n(lstm, seed, horizon, cs)
                st.session_state['rnn_fc']  = rnn_fc
                st.session_state['lstm_fc'] = lstm_fc
                st.session_state['h']       = horizon

        if 'rnn_fc' in st.session_state:
            rnn_fc  = st.session_state['rnn_fc']
            lstm_fc = st.session_state['lstm_fc']
            h_used  = st.session_state['h']
            days    = list(range(1, h_used+1))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📍 SimpleRNN Forecast")
                for d, p in zip(days, rnn_fc):
                    delta_pct = (p - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100
                    st.metric(f"Day +{d}", f"${p:,.2f}", f"{delta_pct:+.2f}% vs last close")
            with col2:
                st.markdown("#### 📍 LSTM Forecast")
                for d, p in zip(days, lstm_fc):
                    delta_pct = (p - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100
                    st.metric(f"Day +{d}", f"${p:,.2f}", f"{delta_pct:+.2f}% vs last close")

            fig, ax = plt.subplots(figsize=(12, 5))
            hist_n = min(60, len(df))
            ax.plot(range(-hist_n, 0), df['Close'].iloc[-hist_n:].values,
                    color='#1f77b4', linewidth=2, label='Historical Close')
            ax.plot(days, rnn_fc,  marker='o', color='#e74c3c', linewidth=2,
                    linestyle='--', label='SimpleRNN Forecast', markersize=7)
            ax.plot(days, lstm_fc, marker='s', color='#2ca02c', linewidth=2,
                    linestyle='-.', label='LSTM Forecast', markersize=7)
            ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, label='Today')
            ax.set_title(f'TSLA — {h_used}-Day Price Forecast', fontsize=13, fontweight='bold')
            ax.set_xlabel('Days (negative = past, positive = future)')
            ax.set_ylabel('Price (USD)'); ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.info("⚠️ **Disclaimer:** These are model predictions for educational purposes only. "
                    "Do NOT use for actual investment decisions.")

# ════════════════════════════════════════════════════════════════
# TAB 4 — Comparison
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 SimpleRNN vs LSTM — Full Comparison")
    if not models_ok:
        st.warning("⚠️ Models not loaded.")
    else:
        def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
        def mae(a,b):  return float(mean_absolute_error(a,b))
        def mape(a,b): return float(np.mean(np.abs((a-b)/(a+1e-8)))*100)
        def r2(a,b):   return float(r2_score(a,b))

        comp = pd.DataFrame([
            {'Model':'SimpleRNN','RMSE':rmse(ytrue,ypR),'MAE':mae(ytrue,ypR),'MAPE':mape(ytrue,ypR),'R2':r2(ytrue,ypR)},
            {'Model':'LSTM',     'RMSE':rmse(ytrue,ypL),'MAE':mae(ytrue,ypL),'MAPE':mape(ytrue,ypL),'R2':r2(ytrue,ypL)},
        ]).set_index('Model')

        best_model = comp['RMSE'].idxmin()
        st.markdown(f"**🏆 Winner: `{best_model}`** (lowest RMSE)")
        st.dataframe(comp.style.format({'RMSE':'${:,.2f}','MAE':'${:,.2f}','MAPE':'{:.2f}%','R2':'{:.4f}'})\
                     .highlight_min(subset=['RMSE','MAE','MAPE'], color='#d4edda')\
                     .highlight_max(subset=['R2'], color='#d4edda'), use_container_width=True)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        clrs = ['#e74c3c','#2ca02c']
        for ax, metric in zip(axes, ['RMSE','MAE','MAPE','R2']):
            vals = comp[metric].values
            bars = ax.bar(comp.index, vals, color=clrs, edgecolor='black', alpha=0.85)
            ax.set_title(f'{metric}', fontweight='bold')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.tick_params(axis='x', rotation=10)
        plt.suptitle('SimpleRNN vs LSTM — Metric Comparison', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("""
        ### 🔬 Why LSTM Wins
        | Aspect | SimpleRNN | LSTM |
        |---|---|---|
        | **Memory** | Short-term only | Short + Long-term via cell state |
        | **Gates** | None | Forget, Input, Output gates |
        | **Gradient problem** | Vanishing gradients | Mitigated by gating |
        | **Sequence length** | Struggles >20 steps | Handles 60+ steps |
        | **Parameters** | Fewer | More (better capacity) |
        | **Training speed** | Faster | Slightly slower |
        """)

# ════════════════════════════════════════════════════════════════
# TAB 5 — Data Explorer
# ════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📋 Dataset Explorer")
    st.markdown(f"**{len(df):,} trading days** | {df['Date'].min().date()} → {df['Date'].max().date()}")

    col1, col2 = st.columns([3,1])
    with col2:
        n_rows = st.slider("Rows to show", 5, 100, 20)
    with col1:
        view = st.radio("View", ["Latest rows","Earliest rows","All stats"], horizontal=True)

    if view == "Latest rows":
        st.dataframe(df[['Date','Open','High','Low','Close','Volume','RSI','MACD','BB_Width']]\
                     .tail(n_rows).round(4), use_container_width=True)
    elif view == "Earliest rows":
        st.dataframe(df[['Date','Open','High','Low','Close','Volume','RSI','MACD','BB_Width']]\
                     .head(n_rows).round(4), use_container_width=True)
    else:
        st.dataframe(df.describe().round(4), use_container_width=True)

    st.markdown("#### 🔥 Feature Correlation")
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[['Close','RSI','MACD','BB_Width','Daily_Return','Volume_Ratio','Price_Range']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>🚗 <b>Tesla Stock Price Predictor</b> | Built with TensorFlow + Streamlit | "
    "Gopi Mahith · Parul University 2026</center>",
    unsafe_allow_html=True
)
