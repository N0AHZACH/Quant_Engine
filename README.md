# 🚀 Quant_Engine: NSE Cross-Sectional Swing Alpha

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Market](https://img.shields.io/badge/Market-NSE%20India-orange.svg)](https://www.nseindia.com/)

A high-performance, Machine Learning-driven quantitative trading engine meticulously engineered for the **National Stock Exchange (NSE)**. This system leverages advanced ensemble learning and cross-sectional analysis to capture swing alpha over 3-day horizons.

---

## 💎 Core Methodology

The system implements a **selective, cross-sectional strategy** that focuses on capital survival and consistent compounding through strict drawdown controls.

-   **Asset Universe**: NSE Equities (selected from a survivorship-bias aware universe).
-   **Trading Style**: Cross-sectional & Market-neutral components.
-   **Holding Period**: Fixed **3 trading days**.
-   **Prediction Target**: `log(Close[t+3] / Close[t])` (3-day log-returns).
-   **Objective**: 30% CAGR with a maximum drawdown limit of 15–20%.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Data_Infrastructure [1. Data Infrastructure & ETL]
        A1[NSE Data Fetcher] -->|Daily/Live| A2[Raw CSV/Parquet Lake]
        A2 -->|Cleaning| A3[Silver Data: Cleaned & Adjusted]
        A3 -->|Feature Engine| A4[Gold Data: ML-Ready Tensors]
        
        subgraph Features [Feature Engineering]
            A4a[Technical: RSI, MACD, EMAs]
            A4b[Cross-Sectional: Z-Scores, Rankings]
            A4c[Volatility: ATR, Rolling Std]
        end
        A4 --- A4a & A4b & A4c
    end

    subgraph Intelligence_Engine [2. AI/ML Ensemble Engine]
        A4 --> B1{Ensemble Predictor}
        subgraph Models [Diverse Model Universe]
            B1a[XGBoost: Gradient Boosting]
            B1b[Random Forest: Tree-Based]
            B1c[CNN/GRU: Temporal Sequence]
            B1d[LSTM: Long-term Dependencies]
        end
        B1 --- B1a & B1b & B1c & B1d
        B1a & B1b & B1c & B1d --> B2[Model Weight Optimization]
        B2 --> B3[Aggregated Return Forecasts]
    end

    subgraph Risk_Decision_Center [3. Risk & Decision Center]
        B3 --> C1[Signal Generation Filter]
        C1 --> C2{Regime Filter}
        C2 -->|Bull/Bear/Flat| C3[Adaptive Risk Manager]
        
        subgraph Risk_Controls [Strict Risk Controls]
            C3a[Active Drawdown Guard]
            C3b[Position Sizing: Vol-Adjusted]
            C3c[Diversification Limits]
        end
        C3 --- C3a & C3b & C3c
        C3 --> C4[Final Allocation Vector]
    end

    subgraph Execution_Validation [4. Execution & Validation]
        C4 --> D1{Trade Gate}
        D1 -->|Live Execution| D2[NSE API / Broker]
        D1 -->|Simulated| D3[Paper Trading Engine]
        
        D2 & D3 --> E1[Performance Metrics Logger]
        E1 -->|Feedback Loop| E2[Backtester / Walk-Forward]
        E2 -->|Hyperparameter Tune| B1
    end

    %% Styles
    style Data_Infrastructure fill:#f0f7ff,stroke:#0056b3,stroke-width:2px
    style Intelligence_Engine fill:#fff3f0,stroke:#d4380d,stroke-width:2px
    style Risk_Decision_Center fill:#f6ffed,stroke:#389e0d,stroke-width:2px
    style Execution_Validation fill:#fffbe6,stroke:#d4b106,stroke-width:2px
    style Features fill:#ffffff,stroke-dasharray: 5 5
    style Models fill:#ffffff,stroke-dasharray: 5 5
    style Risk_Controls fill:#ffffff,stroke-dasharray: 5 5
```

---

The engine is modularized into specialized layers for high maintainability and performance:

| Component | Description |
| :--- | :--- |
| **`src/`** | Core library containing model training, feature engineering, and data preprocessing. |
| **`live/`** | Deployment scripts for live trading and portfolio monitoring. |
| **`production/`** | The "Production-Ready" engine including risk management, ensemble predictors, and decision engines. |
| **`engine/`** | Specialized logic for the backtester and core loading mechanisms. |
| **`docs/`** | Comprehensive architectural and development documentation. |

---

## 🛠️ Tech Stack & Key Features

-   **Machine Learning**: Ensemble of **XGBoost**, **Random Forest**, **LSTM**, **GRU**, and **CNN** for robust signal generation.
-   **Adaptive Risk Manager**: Dynamic position sizing, regime detection, and volatility-adjusted entry thresholds.
-   **Automated Pipeline**: End-to-end data fetching and processing through the `lake/` data management system.
-   **Backtesting Suite**: Walk-forward validation and Monte Carlo simulations to prevent overfitting.

---

## 🚀 Quick Start

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/N0AHZACH/Quant_Engine.git
    cd Quant_Engine
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Execution
-   **Train the Pipeline**:
    ```bash
    python src/train_full_pipeline.py
    ```
-   **Run Backtests**:
    ```bash
    python backtest/run_backtest.py
    ```
-   **Live Monitoring**:
    ```bash
    python live/live_trader_v2.py
    ```

---

## ✅ Performance Targets Checklist

-   [ ] **CAGR**: Target $\geq$ 30%
-   [ ] **Max Drawdown**: Target $\leq$ 20%
-   [ ] **Win Rate**: Target 40% – 55%
-   [ ] **Sharpe Ratio**: Target $\geq$ 2.0
-   [ ] **Profit Factor**: Target 1.6 – 2.3

---

## ⚠️ Disclaimer
This system is for research purposes only. Quantitative trading involves significant risk of loss. Always perform rigorous walk-forward backtesting and paper trading before committing real capital.

---

Developed by [N0AHZACH](https://github.com/N0AHZACH)
