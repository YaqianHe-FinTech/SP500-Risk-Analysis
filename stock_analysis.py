import logging
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch Chinese A-share market data using AKShare."""
    logging.info(f"Fetching A-share data for {ticker} from {start_date} to {end_date}.")
    try:
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")
        
        df = ak.stock_zh_index_daily_em(symbol=ticker, start_date=start_str, end_date=end_str)
        
        if df.empty:
            logging.warning("Dataframe is empty. Check the ticker or network.")
            return df
            
        # Standardize column names for the ML pipeline
        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'close': 'Close', 
            'high': 'High', 'low': 'Low', 'volume': 'Volume'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        return df
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        raise

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering: Building Alpha Factors."""
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
    """Train Random Forest and evaluate baseline accuracy."""
    logging.info("Training Random Forest classifier on A-share data.")
    
    X = df[feature_cols]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    logging.info(f"Model evaluation completed. CSI 300 Accuracy: {accuracy:.4f}")
    return accuracy, model

def plot_feature_importance(model, feature_names: list):
    """Plot and save feature importance chart."""
    logging.info("Generating feature importance chart...")
    importances = model.feature_importances_
    
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='#d62728', edgecolor='black')
    plt.title('Random Forest Feature Importance - CSI 300 (A-Shares)', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('feature_importance_A_share.png', dpi=300)
    logging.info("Chart saved as 'feature_importance_A_share.png'.")
    plt.show()

if __name__ == "__main__":
    # Target: CSI 300 Index (沪深300)
    TICKER = "sh000300"
    START_DATE = "2018-01-01"
    END_DATE = "2024-01-01"
    
    FEATURE_COLUMNS = ['Daily_Return', 'MA_5', 'MA_20', 'MACD', 'RSI_14', 'BB_Width']
    
    raw_data = fetch_market_data(TICKER, START_DATE, END_DATE)
    processed_data = build_features(raw_data)
    
    if not processed_data.empty:
        acc, trained_model = train_and_evaluate_model(processed_data, FEATURE_COLUMNS)
        plot_feature_importance(trained_model, FEATURE_COLUMNS)
