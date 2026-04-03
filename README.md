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
    classDef internalNode fill:#FBFBFB,stroke:#000000,stroke-width:1.5px,color:#000000;

    %% 1. Data Infrastructure
    subgraph Data_Pipe [1. Data Infrastructure & ETL]
        A[<b>NSE Data Fetcher</b>] --> B[<b>Raw Data Lake</b>]
        B --> C[<b>Silver Data Layer</b>]
        C --> D[<b>Gold Data Layer</b><br/>(ML Tensors & Targets)]
    end
    class A,B,C,D mainNode;

    %% 2. Feature Engineering
    subgraph Feature_Eng [2. Feature Engineering Engine]
        E[<b>Feature Extraction Pipeline</b>]
        E --> E1[Technical Indicators]
        E --> E2[Cross-Sectional Rankings]
        E --> E3[Volatility Metrics]
    end
    class E mainNode;
    class E1,E2,E3 internalNode;

    %% Intelligence Engine - Vertical Connections to avoid crossover
    D --> F[<b>3. Intelligence Engine</b>]
    E1 & E2 & E3 --> F
    
    subgraph Ensemble_Section [ML Ensemble Strategy]
        F --> F1[XGBoost & Random Forest]
        F --> F2[CNN / GRU / LSTM]
        F1 & F2 --> G[<b>Final Model Aggregation</b>]
    end
    class F,G mainNode;
    class F1,F2 internalNode;

    %% 4. Risk & Decision Center
    G --> H[<b>4. Risk & Decision Center</b>]
    subgraph Risk_Strategy [Adaptive Risk Controls]
        H --> H1{Regime Filter}
        H1 --> H2[Adaptive Risk Manager]
        H2 --> H3[Active Drawdown Guard]
        H2 --> H4[Volatility-Based Sizing]
    end
    class H,H2 mainNode;
    class H1,H3,H4 internalNode;

    %% 5. Execution & Feedback
    H3 & H4 --> I[<b>5. Execution & Validation</b>]
    subgraph Execution_Loop [Deployment Path]
        I --> I1{Trade Gate}
        I1 --> I2[Live NSE Execution]
        I1 --> I3[Paper Trading Simulation]
        I2 & I3 --> J[<b>Performance Analytics</b>]
    end
    class I,J mainNode;
    class I1,I2,I3 internalNode;

    %% Feedback Loop - Single path to avoid clutter
    J -.->|Feedback Loop| F

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
