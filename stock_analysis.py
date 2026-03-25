import logging
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取指定标的的复权历史数据。"""
    logging.info(f"Fetching market data for {ticker} from {start_date} to {end_date}.")
    try:
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if df.empty:
            logging.warning("Dataframe is empty. Check the ticker or date range.")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        raise

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程与标签构建。"""
    logging.info("Building technical features and target labels.")
    
    # 构建基础特征
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    
    # 构建目标标签：1 表示次日收盘价上涨，0 表示下跌或平盘
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    
    # 清理因计算移动平均和时移产生的空值
    return df.dropna()

def train_and_evaluate_model(df: pd.DataFrame, feature_cols: list) -> float:
    """训练随机森林模型并评估准确率。"""
    logging.info("Training Random Forest classifier.")
    
    X = df[feature_cols]
    y = df['Target']
    
    # 时间序列数据切分，保持时间先后顺序 (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,        # 限制树的深度，防止过拟合
        random_state=42, 
        n_jobs=-1           # 调用所有 CPU 核心并行计算
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    logging.info(f"Model evaluation completed. Baseline Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # 全局参数配置
    TICKER = "SPY"
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-01"
    FEATURE_COLUMNS = ['Daily_Return', 'MA_5', 'MA_10']
    
    # 执行流水线
    raw_data = fetch_market_data(TICKER, START_DATE, END_DATE)
    processed_data = build_features(raw_data)
    
    if not processed_data.empty:
        acc = train_and_evaluate_model(processed_data, FEATURE_COLUMNS)
