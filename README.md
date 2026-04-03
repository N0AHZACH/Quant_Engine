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
    %% Global Style Definitions
    classDef mainNode fill:#fdfdfd,stroke:#000,stroke-width:2px,color:#000;
    classDef subNode fill:#fff,stroke:#666,stroke-width:1px,color:#333,stroke-dasharray: 5 5;

    %% 1. Data Pipeline
    subgraph Data_Pipeline [1. Data Infrastructure & ETL]
        A[<b>NSE Data Fetcher</b><br/>Daily/Live API Ingestion] --> B[<b>Raw Data Lake</b><br/>CSV / Parquet Storage]
        B --> C[<b>Silver Data Layer</b><br/>Cleaning & Corporate Actions]
    end
    class A,B,C mainNode;

    %% 2. Feature Engineering
    subgraph Feature_Layer [2. Feature Engineering Engine]
        D[<b>Feature Extraction</b>]
        D1[Technical: RSI, MACD, EMAs]
        D2[Cross-Sectional: Z-Scores, Rankings]
        D3[Volatility: ATR, Rolling Std]
        D --- D1 & D2 & D3
    end
    class D mainNode;
    class D1,D2,D3 subNode;

    %% 3. ML Ensemble
    C & D1 & D2 & D3 --> E[<b>3. ML Ensemble Predictor</b>]
    subgraph ML_Models [Diverse Model Universe]
        E1[XGBoost & Random Forest]
        E2[CNN / GRU / LSTM]
    end
    E --- E1 & E2
    E1 & E2 --> F[<b>Multi-Model Aggregation</b><br/>Weighted Return Forecasts]
    class E,F mainNode;
    class E1,E2 subNode;

    %% 4. Risk & Decision
    F --> G[<b>4. Risk & Decision Center</b>]
    subgraph Risk_Controls [Adaptive Controls]
        H{Regime Filter}
        I[Adaptive Risk Manager]
        I1[Active Drawdown Guard]
        I2[Volatility Position Sizing]
    end
    G --> H --> I --- I1 & I2
    class G,I mainNode;
    class H,I1,I2 subNode;

    %% 5. Execution & Feedback
    I1 & I2 --> J[<b>5. Execution & Validation</b>]
    J --> K{Trade Gate}
    K --> L[Live NSE Execution]
    K --> M[Paper Trading]
    L & M --> N[<b>Performance Metrics</b><br/>Logging & Feedback Loop]
    N -->|Hyperparameter Tune| E
    class J,N mainNode;
    class K,L,M subNode;

    %% High-level connections
    linkStyle default stroke:#000,stroke-width:1.5px;
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
