import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from FinMind.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import date

# ==========================================
# 0. 頁面與 UI 控制面板設定
# ==========================================
st.set_page_config(page_title="台股量化交易系統", layout="wide")
st.title("📈 台股量化交易分析系統 (ML Optimized)")

# 側邊欄：參數設定
st.sidebar.header("系統參數設定")
stock_id = st.sidebar.text_input("股票代號", value="2330")
start_date = st.sidebar.date_input("開始日期", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("結束日期", value=date.today())

st.sidebar.markdown("---")
st.sidebar.subheader("策略參數")
sma_fast = st.sidebar.slider("短均線天數 (SMA Fast)", 5, 60, 20)
sma_slow = st.sidebar.slider("長均線天數 (SMA Slow)", 20, 240, 60)

st.sidebar.markdown("---")
st.sidebar.subheader("機器學習參數")
N_days = st.sidebar.number_input("預測未來 N 天", value=5)
target_threshold = st.sidebar.slider("目標報酬率閾值 (%)", 0.5, 5.0, 1.5) / 100
upper_conf = st.sidebar.slider("買進信心閾值 (Upper)", 0.5, 0.9, 0.6)
lower_conf = st.sidebar.slider("賣出信心閾值 (Lower)", 0.1, 0.5, 0.4)

cost_rate = st.sidebar.slider("單邊交易成本 (%)", 0.0, 1.0, 0.2) / 100

# ==========================================
# 1 & 2. 資料處理函式 (加入 @st.cache_data 避免重複抓取)
# ==========================================
@st.cache_data
def load_and_preprocess_data(stock_id, start, end, fast, slow):
    dl = DataLoader()
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=str(start), end_date=str(end))
    
    if df.empty:
        return df

    df = df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.ffill(inplace=True)

    # 計算指標
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.roc(length=5, append=True)
    df.ta.stoch(append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.obv(append=True)
    
    # 使用 UI 傳入的均線參數
    df.ta.sma(length=fast, append=True)
    df.ta.sma(length=slow, append=True)
    df['Bias_20'] = (df['Close'] - df[f'SMA_{fast}']) / df[f'SMA_{fast}'] 
    df['Return_1d'] = df['Close'].pct_change()
    df['Return_3d'] = df['Close'].pct_change(3)
    
    df.dropna(inplace=True)
    return df

# ==========================================
# 執行主邏輯
# ==========================================
try:
    with st.spinner('正在計算與回測中...'):
        # 載入資料
        df = load_and_preprocess_data(stock_id, start_date, end_date, sma_fast, sma_slow)
        
        if df.empty:
            st.warning("查無資料，請確認股票代號或日期區間。")
            st.stop()

        # 自動抓取特徵名稱 (避免名稱寫死造成的 KeyError)
        features = [col for col in df.columns if any(x in col for x in ['MACD', 'ADX', 'RSI', 'ROC', 'STOCH', 'ATR', 'BBL', 'BBU', 'OBV', 'Bias', 'Return_1d', 'Return_3d'])]

        # --- 3. 策略與訊號生成 ---
        df['Signal_Rule'] = np.where(df[f'SMA_{sma_fast}'] > df[f'SMA_{sma_slow}'], 1, 0)

        df['Future_Return'] = (df['Close'].shift(-N_days) - df['Close']) / df['Close']
        df['Target'] = np.where(df['Future_Return'] > target_threshold, 1, 0)
        
        df_ml = df.dropna().copy()
        X = df_ml[features]
        y = df_ml['Target']

        split_idx = int(len(df_ml) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_train, y_train)

        prob_up = model.predict_proba(X)[:, 1]
        conditions = [(prob_up > upper_conf), (prob_up < lower_conf)]
        df_ml['Signal_ML_raw'] = np.select(conditions, [1, 0], default=np.nan)
        df_ml['Signal_ML'] = df_ml['Signal_ML_raw'].ffill().fillna(0)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # --- 4. 回測計算 ---
        df_ml['Market_Return'] = df_ml['Close'].pct_change()
        df_ml['Pos_Rule'] = df_ml['Signal_Rule'].shift(1)
        df_ml['Pos_ML'] = df_ml['Signal_ML'].shift(1)
        df_ml.fillna(0, inplace=True)

        df_ml['Ret_Rule'] = df_ml['Pos_Rule'] * df_ml['Market_Return']
        df_ml['Ret_ML'] = df_ml['Pos_ML'] * df_ml['Market_Return']

        df_ml['Cost_Rule'] = df_ml['Pos_Rule'].diff().abs() * cost_rate
        df_ml['Cost_ML'] = df_ml['Pos_ML'].diff().abs() * cost_rate

        df_ml['Net_Ret_Rule'] = df_ml['Ret_Rule'] - df_ml['Cost_Rule'].fillna(0)
        df_ml['Net_Ret_ML'] = df_ml['Ret_ML'] - df_ml['Cost_ML'].fillna(0)

        # 擷取測試集並計算累積報酬
        test_start_date = X_test.index[0]
        df_test = df_ml.loc[test_start_date:].copy()

        df_test['Cum_Market'] = (1 + df_test['Market_Return']).cumprod()
        df_test['Cum_Rule'] = (1 + df_test['Net_Ret_Rule']).cumprod()
        df_test['Cum_ML'] = (1 + df_test['Net_Ret_ML']).cumprod()

        # ==========================================
        # 5. UI 輸出展示
        # ==========================================
        st.subheader(f"回測績效 (測試集區間: {test_start_date.date()} 至今)")
        
        # 建立三個並排的指標卡片
        col1, col2, col3 = st.columns(3)
        col1.metric("Buy & Hold (大盤基準)", f"{(df_test['Cum_Market'].iloc[-1] - 1)*100:.2f}%")
        col2.metric("雙均線策略 (Rule-based)", f"{(df_test['Cum_Rule'].iloc[-1] - 1)*100:.2f}%", 
                    delta=f"{(df_test['Cum_Rule'].iloc[-1] - df_test['Cum_Market'].iloc[-1])*100:.2f}% (vs 大盤)")
        col3.metric("機器學習策略 (ML)", f"{(df_test['Cum_ML'].iloc[-1] - 1)*100:.2f}%", 
                    delta=f"{(df_test['Cum_ML'].iloc[-1] - df_test['Cum_Market'].iloc[-1])*100:.2f}% (vs 大盤)")
        
        st.write(f"✅ ML 模型測試集 Accuracy: **{acc:.2f}**")

        # 使用 Plotly 畫圖
        st.markdown("### 資產淨值曲線 (Equity Curve)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Cum_Market'], name='Buy & Hold', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Cum_Rule'], name='SMA Strategy', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Cum_ML'], name='ML Optimized', line=dict(color='cyan', width=2)))
        fig.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("檢視原始數據與訊號"):
            st.dataframe(df_ml[['Close', f'SMA_{sma_fast}', 'Signal_Rule', 'Signal_ML', 'Target']].tail(10))

except Exception as e:
    st.error(f"發生錯誤：{e}")