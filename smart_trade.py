import pandas as pd
import numpy as np
import pandas_ta as ta
from FinMind.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 資料獲取層 (Data Acquisition)
# ==========================================
print("📥 正在從 FinMind 獲取資料...")
dl = DataLoader()

# 獲取台積電 (2330) 日 K 線資料
df = dl.taiwan_stock_daily(
    stock_id='2330',
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# ==========================================
# 2. 資料預處理與特徵工程 (Preprocessing & Feature Engineering)
# ==========================================
print("⚙️ 進行資料清洗與進階特徵萃取...")

# 1. 基礎清洗與欄位重命名 (必須保留這段)
df = df.rename(columns={
    'date': 'Date', 
    'open': 'Open', 
    'max': 'High', 
    'min': 'Low', 
    'close': 'Close', 
    'Trading_Volume': 'Volume'
})
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.ffill(inplace=True)

# 2. 趨勢特徵 (Trend)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.adx(length=14, append=True)

# 3. 動能特徵 (Momentum)
df.ta.rsi(length=14, append=True)
df.ta.roc(length=5, append=True)
df.ta.stoch(append=True)

# 4. 波動率特徵 (Volatility)
df.ta.atr(length=14, append=True)
df.ta.bbands(length=20, std=2, append=True)

# 5. 成交量特徵 (Volume)
df.ta.obv(append=True)

# 6. 自定義進階特徵
df.ta.sma(length=20, append=True)
df.ta.sma(length=60, append=True)
df['Bias_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] 
df['Return_1d'] = df['Close'].pct_change()
df['Return_3d'] = df['Close'].pct_change(3)

# 移除因計算指標產生的 NaN
df.dropna(inplace=True)

# ---------------------------------------------------------
# 更新第三部分給 ML 模型的 Features 列表 (對應上述產生的新欄位)
features = [
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
    'ADX_14', 'RSI_14', 'ROC_5', 
    'STOCHk_14_3_3', 'STOCHd_14_3_3', 
    'ATRr_14', 
    'BBL_20_2.0_2.0', 'BBU_20_2.0_2.0',
    'OBV', 'Bias_20', 'Return_1d', 'Return_3d'
]
# ==========================================
# 3. 策略與訊號生成層 (Strategy & Signal Generation)
# ==========================================
print("📈 產生交易訊號 (Rule-based & ML-based)...")

# --- 方法一：邏輯判斷 (Rule-based) ---
# 策略：雙均線交叉 (SMA_20 > SMA_60 為 1，否則為 0)
df['Signal_Rule'] = np.where(df['SMA_20'] > df['SMA_60'], 1, 0)


# --- 方法二：機器學習模型 (ML-based) ---
# 1. 重新定義 Target (Y): 預測「未來 5 天的累積報酬率」是否大於 1.5%
N_days = 5
threshold = 0.015

# 計算未來 N 天的報酬率 (使用 shift 將未來價格對齊到今天)
df['Future_Return_5d'] = (df['Close'].shift(-N_days) - df['Close']) / df['Close']

# 若未來 5 天漲幅大於 1.5%，則今天的訊號標記為 1 (看多)
df['Target'] = np.where(df['Future_Return_5d'] > threshold, 1, 0)

# 移除因 shift(-5) 導致最後 5 天產生的 NaN
df_ml = df.dropna().copy()

# 2. 選擇特徵 (Features/X) - 維持原本豐富的特徵矩陣
features = [
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
    'ADX_14', 'RSI_14', 'ROC_5', 
    'STOCHk_14_3_3', 'STOCHd_14_3_3', 
    'ATRr_14', 
    'BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 
    'OBV', 'Bias_20', 'Return_1d', 'Return_3d'
]
X = df_ml[features]
y = df_ml['Target']


# 3. 切割訓練集 (Training) 與測試集 (Testing)
# 時間序列「絕對不可以」打亂 (shuffle=False)，必須按時間順序切割
split_idx = int(len(df_ml) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 4. 建立與訓練 Random Forest 分類器
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# 5. 輸出模型預測「機率」([:, 1] 代表預測為 1 (上漲) 的機率)
prob_up = model.predict_proba(X)[:, 1]

# 實作信心水準與磁滯效應 (Hysteresis) 以降低頻繁交易成本
# 設定閾值
upper_threshold = 0.60
lower_threshold = 0.40

# 建立條件陣列
conditions = [
    (prob_up > upper_threshold),
    (prob_up < lower_threshold)
]
choices = [1, 0]

# 當機率大於 60% 給 1，小於 40% 給 0，介於中間給 NaN
df_ml['Signal_ML_raw'] = np.select(conditions, choices, default=np.nan)

# 神奇的 ffill (Forward Fill)：遇到 NaN 時，自動填寫前一天的數值 (維持部位)
# 最後若最前面還有 NaN，則填補 0 (空手)
df_ml['Signal_ML'] = df_ml['Signal_ML_raw'].ffill().fillna(0)

# (這行保持不變，用於計算測試集準確率，但改用原本的 predict 來算)
y_pred = model.predict(X_test)
print(f"✅ ML 模型在測試集上的預測準確率 (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print("\n最終資料表預覽:")
print(df_ml[['Close', 'SMA_20', 'Signal_Rule', 'Signal_ML', 'Target']].tail())

# ==========================================
# 4. 回測引擎與績效評估 (Backtesting & Evaluation)
# ==========================================
print("\n💰 執行向量化回測 (包含交易成本)...")

# 1. 計算市場每日基準報酬率 (Buy and Hold)
df_ml['Market_Return'] = df_ml['Close'].pct_change()

# 2. 避免未來函數 (Look-ahead Bias)
# 今天的訊號，決定明天的部位 (今天收盤看到訊號，明天開盤才能買賣)
df_ml['Pos_Rule'] = df_ml['Signal_Rule'].shift(1)
df_ml['Pos_ML'] = df_ml['Signal_ML'].shift(1)
df_ml.fillna(0, inplace=True)

# 3. 計算策略每日報酬率 (部位 * 市場報酬)
df_ml['Ret_Rule'] = df_ml['Pos_Rule'] * df_ml['Market_Return']
df_ml['Ret_ML'] = df_ml['Pos_ML'] * df_ml['Market_Return']

# 4. 模擬台股交易成本 (摩擦成本)
# 假設每次「換倉」(0變1 或 1變0) 產生的單邊成本約為 0.2% (手續費+滑價)
cost_rate = 0.002
# diff().abs() 可以抓出部位變動的時間點 (變動為1，沒變動為0)
df_ml['Cost_Rule'] = df_ml['Pos_Rule'].diff().abs() * cost_rate
df_ml['Cost_ML'] = df_ml['Pos_ML'].diff().abs() * cost_rate

# 5. 扣除成本後的淨報酬
df_ml['Net_Ret_Rule'] = df_ml['Ret_Rule'] - df_ml['Cost_Rule'].fillna(0)
df_ml['Net_Ret_ML'] = df_ml['Ret_ML'] - df_ml['Cost_ML'].fillna(0)

# ==========================================
# 5. 結算測試集績效
# ==========================================
# 為了公平，我們只從「測試集 (Testing Set)」開始計算累積報酬
test_start_date = X_test.index[0]
df_test = df_ml.loc[test_start_date:].copy()

# 計算累積報酬 (Cumulative Product)
df_test['Cum_Market'] = (1 + df_test['Market_Return']).cumprod()
df_test['Cum_Rule'] = (1 + df_test['Net_Ret_Rule']).cumprod()
df_test['Cum_ML'] = (1 + df_test['Net_Ret_ML']).cumprod()

print(f"\n=== 測試集回測結果 (從 {test_start_date.date()} 開始) ===")
print(f"📈 Buy & Hold (大盤) 累積報酬: {(df_test['Cum_Market'].iloc[-1] - 1)*100:>6.2f}%")
print(f"📊 雙均線策略 (Rule)  累積報酬: {(df_test['Cum_Rule'].iloc[-1] - 1)*100:>6.2f}%")
print(f"🤖 機器學習策略 (ML) 累積報酬: {(df_test['Cum_ML'].iloc[-1] - 1)*100:>6.2f}%")