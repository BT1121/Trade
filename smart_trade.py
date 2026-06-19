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
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="台股量化交易系統", layout="wide")
st.title("台股量化交易分析系統 (ML Optimized)")


# ==========================================
# 1. 資料獲取與特徵工程
# ==========================================
@st.cache_data
def load_and_preprocess_data(stock_id, start, end, fast, slow):
    """從 FinMind 取得股價資料，並計算技術指標與衍生特徵。

    快取取決於: stock_id, 日期區間, SMA 參數
    """
    dl = DataLoader()
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=str(start), end_date=str(end))

    if df.empty:
        return df

    df = df.rename(columns={
        'date': 'Date', 'open': 'Open', 'max': 'High',
        'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.ffill(inplace=True)
    df = df[df['Close'] > 0]
    
    # 技術指標
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.roc(length=5, append=True)
    df.ta.stoch(append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.obv(append=True)

    # 使用者自訂均線 & 衍生特徵
    df.ta.sma(length=fast, append=True)
    df.ta.sma(length=slow, append=True)
    df['Bias_20'] = (df['Close'] - df[f'SMA_{fast}']) / df[f'SMA_{fast}']
    df['Return_1d'] = df['Close'].pct_change()
    df['Return_3d'] = df['Close'].pct_change(3)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# ==========================================
# 1b. 股票資訊查詢 (代號 ↔ 名稱雙向解析)
# ==========================================
@st.cache_data
def get_taiwan_stock_info():
    """取得台股代號與名稱對照表。

    快取取決於: FinMind 回傳內容（session 期間不變）。
    """
    dl = DataLoader()
    info = dl.taiwan_stock_info()
    # 每檔股票可能有多筆記錄 (不同產業別)，取最新一筆
    latest = info.sort_values('date').groupby('stock_id').last().reset_index()
    return latest[['stock_id', 'stock_name']]


def resolve_stock_input(user_input, stock_info):
    """解析使用者輸入，回傳 (stock_id, stock_name)。

    支援三種比對模式：
      1. 精確比對股票代號 (ex: '2330')
      2. 精確比對股票名稱 (ex: '台積電')
      3. 模糊比對股票名稱 (ex: '台積')
    """
    user_input = user_input.strip()
    if not user_input:
        raise ValueError("請輸入股票代號或名稱")

    # 1. 精確比對股票代號
    match = stock_info[stock_info['stock_id'] == user_input]
    if not match.empty:
        row = match.iloc[0]
        return row['stock_id'], row['stock_name']

    # 2. 精確比對股票名稱
    match = stock_info[stock_info['stock_name'] == user_input]
    if len(match) == 1:
        row = match.iloc[0]
        return row['stock_id'], row['stock_name']
    elif len(match) > 1:
        ids = match['stock_id'].tolist()
        raise ValueError(
            f"「{user_input}」有多筆結果：{', '.join(ids)}，請改用股票代號"
        )

    # 3. 模糊比對股票名稱 (部分字串)
    match = stock_info[stock_info['stock_name'].str.contains(user_input, na=False)]
    if len(match) == 1:
        row = match.iloc[0]
        return row['stock_id'], row['stock_name']
    elif len(match) > 1:
        samples = [
            f"{r['stock_id']} {r['stock_name']}"
            for _, r in match.head(5).iterrows()
        ]
        suffix = f"…等 {len(match)} 筆" if len(match) > 5 else ""
        raise ValueError(
            f"「{user_input}」模糊比對到多筆：{', '.join(samples)}{suffix}，"
            "請輸入更精確的名稱或股票代號"
        )

    raise ValueError(f"查無股票資料：{user_input}")


# ==========================================
# 2. 機器學習模型訓練
# ==========================================
@st.cache_data
def prepare_ml_data(df, features, N_days, target_threshold):
    """建立標籤（Target）、分割訓練/測試集、訓練 RandomForest 分類器。

    快取取決於: df 內容（含 TA 指標）、特徵列表、N_days、目標報酬率閾值
    """
    df = df.copy()

    # 建立未來 N 日報酬與二元標籤
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

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, df_ml, split_idx, acc


# ==========================================
# 3. 回測引擎
# ==========================================
def run_backtest(df_ml, model, features, sma_fast, sma_slow,
                 upper_conf, lower_conf, cost_rate, split_idx):
    """根據 ML 預測信心度與 SMA 規則產生交易訊號，回測並計算累積報酬。

    無快取 — 運算輕量，且相依於多個頻繁調整的滑桿參數。
    """
    df = df_ml.copy()

    # ---- 規則型訊號 (SMA 黃金/死亡交叉) ----
    df['Signal_Rule'] = np.where(
        df[f'SMA_{sma_fast}'] > df[f'SMA_{sma_slow}'], 1, 0
    )

    # ---- ML 機率型訊號 (使用信心閾值) ----
    prob_up = model.predict_proba(df[features])[:, 1]
    conditions = [(prob_up > upper_conf), (prob_up < lower_conf)]
    df['Signal_ML_raw'] = np.select(conditions, [1, 0], default=np.nan)
    df['Signal_ML'] = df['Signal_ML_raw'].ffill().fillna(0)

    # ---- 計算各策略每日報酬 ----
    df['Market_Return'] = df['Close'].pct_change()
    df['Pos_Rule'] = df['Signal_Rule'].shift(1)
    df['Pos_ML'] = df['Signal_ML'].shift(1)
    df.fillna(0, inplace=True)

    df['Ret_Rule'] = df['Pos_Rule'] * df['Market_Return']
    df['Ret_ML'] = df['Pos_ML'] * df['Market_Return']

    df['Cost_Rule'] = df['Pos_Rule'].diff().abs() * cost_rate
    df['Cost_ML'] = df['Pos_ML'].diff().abs() * cost_rate

    df['Net_Ret_Rule'] = df['Ret_Rule'] - df['Cost_Rule'].fillna(0)
    df['Net_Ret_ML'] = df['Ret_ML'] - df['Cost_ML'].fillna(0)

    # ---- 測試集期間累積報酬曲線 ----
    test_start_date = df.index[split_idx]
    df_test = df.loc[test_start_date:].copy()

    df_test['Cum_Market'] = (1 + df_test['Market_Return']).cumprod()
    df_test['Cum_Rule'] = (1 + df_test['Net_Ret_Rule']).cumprod()
    df_test['Cum_ML'] = (1 + df_test['Net_Ret_ML']).cumprod()

    return df_test, test_start_date, df


# ==========================================
# 4. 主程式區塊 (Streamlit UI)
# ==========================================

# --- 側邊欄參數設定 ---
st.sidebar.header("系統參數設定")
stock_input = st.sidebar.text_input("股票代號或名稱 (支援代碼/名稱)", value="2330")
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

# --- 核心邏輯執行 ---
try:
    # --- Step 0: 股票代號 / 名稱解析 ---
    stock_info = get_taiwan_stock_info()
    stock_id, stock_name = resolve_stock_input(stock_input, stock_info)

    with st.spinner('正在計算與回測中...'):
        # Step 1: 資料獲取與特徵工程
        #  快取命中條件: stock_id, date range, SMA 參數皆未變
        df = load_and_preprocess_data(stock_id, start_date, end_date, sma_fast, sma_slow)

        if df.empty:
            st.warning(f"「{stock_name} ({stock_id})」查無交易資料，請確認日期區間。")
            st.stop()

        # 自動抓取特徵名稱 (避免名稱寫死造成的 KeyError)
        features = [
            col for col in df.columns
            if any(x in col for x in [
                'MACD', 'ADX', 'RSI', 'ROC', 'STOCH', 'ATR',
                'BBL', 'BBU', 'OBV', 'Bias', 'Return_1d', 'Return_3d'
            ])
        ]

        # Step 2: ML 模型訓練
        #  快取命中條件: df 內容、N_days、target_threshold 皆未變
        #  → 調整 confidence / cost 滑桿不觸發重新訓練
        model, df_ml, split_idx, acc = prepare_ml_data(
            df, features, N_days, target_threshold
        )

        # Step 3: 回測引擎 (無快取，輕量運算)
        df_test, test_start_date, df_full = run_backtest(
            df_ml, model, features,
            sma_fast, sma_slow,
            upper_conf, lower_conf,
            cost_rate, split_idx
        )

        # ==========================================
        # 5. UI 輸出展示
        # ==========================================
        # 股票資訊卡片 (置於結果最上方)
        scol1, scol2 = st.columns([1, 3])
        scol1.metric("股票代號", stock_id)
        scol2.metric("股票名稱", stock_name)
        st.markdown("---")

        st.subheader(f"回測績效 (測試集區間: {test_start_date.date()} 至今)")

        # 三個並排指標卡片
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Buy & Hold (大盤基準)",
            f"{(df_test['Cum_Market'].iloc[-1] - 1) * 100:.2f}%"
        )
        col2.metric(
            "雙均線策略 (Rule-based)",
            f"{(df_test['Cum_Rule'].iloc[-1] - 1) * 100:.2f}%",
            delta=f"{(df_test['Cum_Rule'].iloc[-1] - df_test['Cum_Market'].iloc[-1]) * 100:.2f}% (vs 大盤)"
        )
        col3.metric(
            "機器學習策略 (ML)",
            f"{(df_test['Cum_ML'].iloc[-1] - 1) * 100:.2f}%",
            delta=f"{(df_test['Cum_ML'].iloc[-1] - df_test['Cum_Market'].iloc[-1]) * 100:.2f}% (vs 大盤)"
        )

        st.write(f"✅ ML 模型測試集 Accuracy: **{acc:.2f}**")

        # Plotly 圖表
        st.markdown("### 資產淨值曲線 (Equity Curve)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_test.index, y=df_test['Cum_Market'],
            name='Buy & Hold', line=dict(color='gray', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df_test.index, y=df_test['Cum_Rule'],
            name='SMA Strategy', line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=df_test.index, y=df_test['Cum_ML'],
            name='ML Optimized', line=dict(color='cyan', width=2)
        ))
        fig.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("檢視原始數據與訊號"):
            st.dataframe(
                df_full[['Close', f'SMA_{sma_fast}', 'Signal_Rule', 'Signal_ML', 'Target']].tail(10)
            )

except Exception as e:
    st.error(f"發生錯誤：{e}")
