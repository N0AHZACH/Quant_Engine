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
    classDef mainNode fill:#fff,stroke:#333,stroke-width:1.5px,color:#000;
    classDef internalNode fill:#fdfdfd,stroke:#999,stroke-width:1px,color:#444,stroke-dasharray: 4 4;

    %% 1. Data Infrastructure (Lake Architecture)
    subgraph Data_Lake [1. Data Infrastructure & ETL]
        A[<b>NSE Data Fetcher</b><br/>Multi-Source API Ingestion] --> B[<b>Raw Data Lake</b><br/>Immutable CSV/Parquet Store]
        B --> C[<b>Silver Data Layer</b><br/>Cleaning, Adjustments & Splits]
        C --> D[<b>Gold Data Layer</b><br/>ML-Ready Model Tensors & Targets]
    end
    class A,B,C,D mainNode;

    %% 2. Feature Engineering
    subgraph Feature_Layer [2. Feature Engineering Engine]
        E[<b>Feature Extraction Pipeline</b>]
        E1[Technical: RSI, MACD, EMAs]
        E2[Cross-Sectional: Rankings, Z-Scores]
        E3[Volatility: ATR, Rolling Std]
        E --- E1 & E2 & E3
    end
    class E mainNode;
    class E1,E2,E3 internalNode;

    %% 3. Intelligence Engine
    D & E1 & E2 & E3 --> F[<b>3. ML Ensemble Intelligence</b>]
    subgraph Model_Mix [Diverse Model Universe]
        F1[XGBoost & Random Forest]
        F2[CNN / GRU / LSTM]
    end
    F --- F1 & F2
    F1 & F2 --> G[<b>Multi-Model Aggregation</b><br/>Weighted Return Forecasts]
    class F,G mainNode;
    class F1,F2 internalNode;

    %% 4. Risk & Decision Center
    G --> H[<b>4. Risk & Decision Center</b>]
    subgraph Risk_Controls [Adaptive Risk Strategy]
        H1{Regime Filter}
        H2[Adaptive Risk Manager]
        H3[Active Drawdown Guard]
        H4[Volatility-Based Sizing]
    end
    H --> H1 --> H2 --- H3 & H4
    class H,H2 mainNode;
    class H1,H3,H4 internalNode;

    %% 5. Execution & Feedback
    H3 & H4 --> I[<b>5. Execution & Validation</b>]
    subgraph Execution_Path [Deployment]
        I1{Trade Gate}
        I2[Live NSE Execution]
        I3[Paper Trading Simulation]
    end
    I --> I1 --> I2 & I3
    I2 & I3 --> J[<b>Performance Metrics</b><br/>Logging & Meta-Analytics]
    J -->|Feedback Loop| F
    class I,J mainNode;
    class I1,I2,I3 internalNode;

    %% Connector Styling
    linkStyle default stroke:#444,stroke-width:1.2px;
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
