# 量化金融分析与机器学习趋势预测 (S&P 500)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Quantitative_Finance-success?style=for-the-badge" alt="Finance" />
</div>

## 📌 项目简介
本项目针对标普500指数（SPY）构建了一个端到端的量化金融分析与机器学习预测流水线 (Pipeline)。项目不仅包含基础的探索性数据分析 (EDA) 与风险指标评估，还引入了基于随机森林 (Random Forest) 的二分类模型，用于预测资产的次日涨跌趋势。

代码采用生产级规范重构，包含类型提示 (Type Hinting)、模块化设计及标准日志记录 (Logging)，具备良好的可扩展性。

## ⚙️ 核心工程模块

### 1. 数据管道与风控分析 (Data Pipeline & EDA)
* **自动化数据拉取：** 基于 `yfinance` 稳定获取复权历史数据，包含异常处理机制。
* **风险评估：** 实现了**最大回撤 (Max Drawdown)** 与日收益率波动计算，量化市场下行风险敞口。

### 2. 机器学习预测流水线 (ML Pipeline)
* **特征工程 (Feature Engineering)：** 构建了基于时间序列的滚动技术指标（如 MA_5, MA_10）作为模型输入特征。
* **时间序列防穿越切分：** 严格按照时间先后顺序进行 `train_test_split` (shuffle=False)，杜绝数据泄露。
* **分类模型构建：** 部署了 `RandomForestClassifier`，通过控制树的深度 (`max_depth`) 缓解过拟合问题。
* **基准测试 (Baseline)：** 当前基础特征集下的模型方向预测准确率稳定在 **49.50%** 左右，为后续的 Alpha 因子挖掘提供了可靠的对照基准。

## 🚀 快速启动

### 环境依赖
```bash
pip install yfinance pandas scikit-learn matplotlib
