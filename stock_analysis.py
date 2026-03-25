import logging
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch adjusted historical market data."""
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
    """Feature Engineering: Building advanced Alpha Factors."""
    logging.info("Building advanced technical features (MACD, RSI, Volatility).")
    
    # 1. Basic Returns & Moving Averages
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # 2. MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # 3. RSI (Relative Strength Index) - 14 Days
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
    
    # 4. Bollinger Bands Width (Volatility)
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (4 * bb_std) / df['MA_20']
    
    # 5. Target Label (1 if tomorrow goes up, else 0)
    df['Tomorrow_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    
    return df.dropna()

def train_and_evaluate_model(df: pd.DataFrame, feature_cols: list) -> float:
    """Train Random Forest and evaluate baseline accuracy."""
    logging.info("Training Random Forest classifier with new Alpha factors.")
    
    X = df[feature_cols]
    y = df['Target']
    
    # Chronological split to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    model = RandomForestClassifier(
        n_estimators=150,   # Increased number of trees
        max_depth=6,        # Slightly deeper trees to understand complex features
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    logging.info(f"Model evaluation completed. New Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # Global Configuration
    TICKER = "SPY"
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-01"
    
    # We now feed the model 6 advanced financial indicators instead of 3
    FEATURE_COLUMNS = [
        'Daily_Return', 'MA_5', 'MA_20', 
        'MACD', 'RSI_14', 'BB_Width'
    ]
    
    # Execute Pipeline
    raw_data = fetch_market_data(TICKER, START_DATE, END_DATE)
    processed_data = build_features(raw_data)
    
    if not processed_data.empty:
        acc = train_and_evaluate_model(processed_data, FEATURE_COLUMNS)
