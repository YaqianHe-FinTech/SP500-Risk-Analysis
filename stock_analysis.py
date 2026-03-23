import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. 获取数据 (以标普500指数 ETF 'SPY' 为例)
ticker = "SPY"
print(f"正在获取 {ticker} 的历史数据...")

# 【关键修复】使用更稳定的 history() 方法来获取数据
data = yf.Ticker(ticker).history(start="2023-01-01", end="2024-01-01")

# 2. 计算每日收益率 (金融分析的核心指标)
# 公式: (今日价格 - 昨日价格) / 昨日价格
# 【关键修复】history() 方法返回的 'Close' 已经是复权价格，直接使用即可
data['Daily_Return'] = data['Close'].pct_change()

# 3. 计算风险指标：最大回撤 (Max Drawdown)
# 衡量资产从最高点跌落的幅度
rolling_max = data['Close'].cummax()
drawdown = (data['Close'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# 4. 可视化展示
plt.figure(figsize=(12, 6))

# 子图1: 价格走势
plt.subplot(2, 1, 1)
plt.plot(data['Close'], label='Close Price', color='blue')
plt.title(f'{ticker} Price Analysis')
plt.legend()

# 子图2: 每日收益率分布
plt.subplot(2, 1, 2)
plt.hist(data['Daily_Return'].dropna(), bins=50, color='green', alpha=0.7)
plt.title('Daily Returns Distribution')

plt.tight_layout()
plt.show()

print("-" * 30)
print(f"分析完成！")
print(f"该时段内最大回撤为: {max_drawdown:.2%}")