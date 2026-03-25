import logging
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取复权历史数据"""
    logging.info(f"Fetching market data for {ticker} from {start_date} to {end_date}.")
    try:
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if df.empty:
            logging.warning("Dataframe is empty.")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        raise

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程：构建 Alpha 因子"""
    logging.info("Building advanced technical features (MACD, RSI, Volatility).")
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
    
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (4 * bb_std) / df['MA_20']
    
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    
    return df.dropna()

def train_and_evaluate_model(df: pd.DataFrame, feature_cols: list) -> tuple:
    """训练模型并返回准确率和训练好的模型实例"""
    logging.info("Training Random Forest classifier.")
    
    X = df[feature_cols]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    logging.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
    # 相比之前，这里多返回了一个 model 给下一步画图用
    return accuracy, model

def plot_feature_importance(model, feature_names: list):
    """绘制并保存特征重要性条形图"""
    logging.info("Generating feature importance chart...")
    importances = model.feature_importances_
    
    # 将特征名和重要性组合成表格，并按重要性从小到大排序
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # 设置高颜值的图表样式
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='#1f77b4', edgecolor='black')
    plt.title('Random Forest - Feature Importance (Alpha Factors)', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance (Contribution to Model)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 自动保存为高清图片
    plt.savefig('feature_importance.png', dpi=300)
    logging.info("Chart saved successfully as 'feature_importance.png'.")
    plt.show()

if __name__ == "__main__":
    TICKER = "SPY"
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-01"
    
    FEATURE_COLUMNS = ['Daily_Return', 'MA_5', 'MA_20', 'MACD', 'RSI_14', 'BB_Width']
    
    raw_data = fetch_market_data(TICKER, START_DATE, END_DATE)
    processed_data = build_features(raw_data)
    
    if not processed_data.empty:
        # 获取准确率和训练好的模型
        acc, trained_model = train_and_evaluate_model(processed_data, FEATURE_COLUMNS)
        # 将模型和特征名字传给画图函数
        plot_feature_importance(trained_model, FEATURE_COLUMNS)
