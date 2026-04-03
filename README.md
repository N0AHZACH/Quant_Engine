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
flowchart TD
    %% Global Style Definitions
    classDef mainNode fill:#FFFFFF,stroke:#000000,stroke-width:2px,color:#000000;
    classDef internalNode fill:#F9F9F9,stroke:#000000,stroke-width:1px,color:#000000;
    classDef labelNode fill:none,stroke:none,color:#000000,font-weight:bold;

    %% 1. Data Infrastructure (Lake Architecture)
    subgraph Data_Lake [1. Data Infrastructure & ETL]
        A[<b>NSE Data Fetcher</b><br/>Multi-Source API Ingestion]
        B[<b>Raw Data Lake</b><br/>Immutable CSV/Parquet Store]
        C[<b>Silver Data Layer</b><br/>Cleaning, Adjustments & Splits]
        D[<b>Gold Data Layer</b><br/>ML-Ready Model Tensors & Targets]
        
        A --- B
        B --- C
        C --- D
    end
    class A,B,C,D mainNode;

    %% 2. Feature Engineering
    subgraph Feature_Layer [2. Feature Engineering Engine]
        E[<b>Feature Extraction Pipeline</b>]
        E1[Technical: RSI, MACD, EMAs]
        E2[Cross-Sectional: Rankings, Z-Scores]
        E3[Volatility: ATR, Rolling Std]
        
        E --- E1
        E --- E2
        E --- E3
    end
    class E mainNode;
    class E1,E2,E3 internalNode;

    %% 3. Intelligence Engine
    D --- F
    E1 --- F
    E2 --- F
    E3 --- F
    subgraph Intelligence_Engine [3. ML Ensemble Intelligence]
        F[<b>Ensemble Predictor</b>]
        F1[XGBoost & Random Forest]
        F2[CNN / GRU / LSTM]
        G[<b>Multi-Model Aggregation</b><br/>Weighted Return Forecasts]
        
        F --- F1
        F --- F2
        F1 --- G
        F2 --- G
    end
    class F,G mainNode;
    class F1,F2 internalNode;

    %% 4. Risk & Decision Center
    G --- H
    subgraph Risk_Decision [4. Risk & Decision Center]
        H[<b>Signal Logic Filter</b>]
        H1{Regime Filter}
        H2[Adaptive Risk Manager]
        H3[Active Drawdown Guard]
        H4[Volatility-Based Sizing]
        
        H --- H1
        H1 --- H2
        H2 --- H3
        H2 --- H4
    end
    class H,H2 mainNode;
    class H1,H3,H4 internalNode;

    %% 5. Execution & Feedback
    H3 --- I
    H4 --- I
    subgraph Execution_Feedback [5. Execution & Validation]
        I[<b>Execution Gate</b>]
        I1{Trade Gate}
        I2[Live NSE Execution]
        I3[Paper Trading Simulation]
        J[<b>Performance Metrics</b><br/>Logging & Meta-Analytics]
        
        I --- I1
        I1 --- I2
        I1 --- I3
        I2 --- J
        I3 --- J
    end
    class I,J mainNode;
    class I1,I2,I3 internalNode;

    %% Global Feedback loop
    J -.->|Feedback Loop| F

    %% Connector Styling - High Visibility
    linkStyle default stroke:#000000,stroke-width:2px;
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
