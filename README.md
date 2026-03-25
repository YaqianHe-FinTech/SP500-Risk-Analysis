# Quantitative Analysis & ML Trend Prediction (S&P 500)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Quantitative_Finance-success?style=for-the-badge" alt="Finance" />
</div>

## 📌 Project Overview
This repository contains an end-to-end quantitative financial analysis and machine learning pipeline for the S&P 500 Index (SPY). It moves beyond basic Exploratory Data Analysis (EDA) and risk metric evaluation by implementing a Random Forest binary classification model to predict the next-day price trend (up/down).

The codebase has been refactored to meet production-level engineering standards, featuring modular design, type hinting, and standard logging practices.

## ⚙️ Core Engineering Modules

### 1. Data Pipeline & Risk Analysis
* **Automated Data Ingestion:** Utilizes `yfinance` to reliably fetch adjusted historical market data with built-in exception handling.
* **Risk Assessment:** Calculates **Maximum Drawdown** and visualizes daily return volatility to quantify downside risk exposure.

## 2. Machine Learning Pipeline (Alpha Research)
* **Advanced Feature Engineering:** Moving beyond simple moving averages, the pipeline now incorporates institutional-grade technical indicators:
  * **MACD (Moving Average Convergence Divergence)** for trend following.
  * **RSI (Relative Strength Index)** for momentum and mean-reversion signals.
  * **Bollinger Bands Width** for modeling realized volatility.
* **Strict Time-Series Validation:** Data is split chronologically (`shuffle=False`) to strictly prevent look-ahead bias (Data Leakage).
* **Model Performance:** By integrating these Alpha factors, the Random Forest classifier's directional prediction accuracy **improved from the 49.50% baseline to 52.00%**, demonstrating the predictive power of domain-specific feature engineering in noisy financial time series.

## 🚀 Quick Start
Gemini 说
Got it! Let's switch to English.

As I mentioned, 49.50% is a very realistic and honest baseline. In quantitative finance, if a simple moving average model gives you 60%+ accuracy, it almost always means there is a "Data Leakage" bug (the model accidentally peeked at future data). Your result proves your code is rigorous and your testing methodology is sound.

Let's update your GitHub repository with this professional, English-language documentation and your newly refactored production-ready code.

Step 1: Update Your README.md
Go to your GitHub repository, click the pencil icon to edit your README.md, and replace everything with this professional English template:

Markdown
# Quantitative Analysis & ML Trend Prediction (S&P 500)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Quantitative_Finance-success?style=for-the-badge" alt="Finance" />
</div>

## 📌 Project Overview
This repository contains an end-to-end quantitative financial analysis and machine learning pipeline for the S&P 500 Index (SPY). It moves beyond basic Exploratory Data Analysis (EDA) and risk metric evaluation by implementing a Random Forest binary classification model to predict the next-day price trend (up/down).

The codebase has been refactored to meet production-level engineering standards, featuring modular design, type hinting, and standard logging practices.

## ⚙️ Core Engineering Modules

### 1. Data Pipeline & Risk Analysis
* **Automated Data Ingestion:** Utilizes `yfinance` to reliably fetch adjusted historical market data with built-in exception handling.
* **Risk Assessment:** Calculates **Maximum Drawdown** and visualizes daily return volatility to quantify downside risk exposure.

### 2. Machine Learning Pipeline
* **Feature Engineering:** Constructs rolling technical indicators (e.g., MA_5, MA_10) as model input features.
* **Time-Series Split:** Strictly splits training and testing sets chronologically (`shuffle=False`) to prevent data leakage (look-ahead bias).
* **Classification Model:** Deploys a `RandomForestClassifier` with depth constraints (`max_depth=5`) to mitigate overfitting.
* **Baseline Performance:** The current baseline model achieves a directional prediction accuracy of approximately **49.50%**, providing a realistic and robust benchmark for future Alpha factor research.

## 🚀 Quick Start

### Prerequisites
```bash
pip install yfinance pandas scikit-learn matplotlib
Run the Pipeline
Execute the main script to trigger data ingestion, feature building, model training, and log output:
python stock_analysis.py
🔮 Roadmap
With the baseline pipeline established, future iterations will focus on:

Multi-Factor Expansion: Introducing momentum indicators (RSI, MACD) and volatility metrics (Bollinger Bands).

Hyperparameter Tuning: Utilizing GridSearchCV or Optuna to optimize model parameters.

Model Ensembling: Exploring Gradient Boosting frameworks (XGBoost, LightGBM) to capture complex, non-linear market patterns.
### Prerequisites
```bash
pip install yfinance pandas scikit-learn matplotlib
