#  Quant Engine: NSE Cross-Sectional Swing Alpha

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
    classDef mainNode fill:#FFFFFF,stroke:#000000,stroke-width:2.5px,color:#000000;
    classDef detailNode fill:#F9F9F9,stroke:#333333,stroke-width:1.5px,color:#111111,stroke-dasharray: 4 4;

    %% 1. Data Infrastructure & ETL
    subgraph Data_Pipe [1. Data Infrastructure & ETL]
        A[<b>NSE Data Fetcher</b><br/>Multi-Source API Ingestion]
        B[<b>Raw Data Lake</b><br/>Immutable CSV/Parquet Store]
        C[<b>Silver Data Layer</b><br/>Cleaning, Adjustments & Splits]
        D[<b>Gold Data Layer</b><br/>ML-Ready Model Tensors & Targets]
        
        A --> B
        B --> C
        C --> D
    end
    class A,B,C,D mainNode;

    %% 2. Feature Engineering
    D --> E[<b>Feature Extraction Engine</b>]
    subgraph Feature_Eng [Feature Library]
        E1[Technical: RSI, MACD, EMAs]
        E2[Cross-Sectional: Rankings, Z-Scores]
        E3[Volatility: ATR, Rolling Std]
        
        E --> E1
        E --> E2
        E --> E3
    end
    class E mainNode;
    class E1,E2,E3 detailNode;

    %% 3. Intelligence Engine
    E1 --> F
    E2 --> F
    E3 --> F
    subgraph Intelligence [ML Ensemble Strategy]
        F[<b>Ensemble Predictor Engine</b>]
        F1[XGBoost & Random Forest Ensembles]
        F2[CNN / GRU / LSTM Temporal Models]
        G[<b>Model Weight Optimization</b>]
        H[<b>Final Weighted Signal Generation</b>]
        
        F --> F1
        F --> F2
        F1 --> G
        F2 --> G
        G --> H
    end
    class F,G,H mainNode;
    class F1,F2 detailNode;

    %% 4. Risk Decision Center
    H --> I[<b>Decision Engine</b>]
    subgraph Risk_Center [Risk & Decision Center]
        I1[Regime Filter: Bull / Bear / Flat]
        I2[Adaptive Risk Manager]
        I3[Active Drawdown Guard]
        I4[Volatility-Based Position Sizing]
        
        I --> I1
        I1 --> I2
        I2 --> I3
        I2 --> I4
    end
    class I,I2 mainNode;
    class I1,I3,I4 detailNode;

    %% 5. Execution & Validation
    I3 --> J
    I4 --> J
    subgraph Execution [Execution & Validation]
        J[<b>Trade Gate Filter</b>]
        J1[Live NSE Execution: API/Broker]
        J2[Paper Trading Simulation]
        K[<b>Performance Analytics Logger</b>]
        L[<b>Backtester / Walk-forward Loop</b>]
        
        J --> J1
        J --> J2
        J1 --> K
        J2 --> K
        K --> L
    end
    class J,K,L mainNode;
    class J1,J2 detailNode;

    %% Global Feedback loop
    L -.->|Meta-Optimization| F

    %% Global Connector Styling
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
