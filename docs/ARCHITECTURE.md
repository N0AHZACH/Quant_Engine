# System Architecture

## Overview

Quant_Engine implements a hierarchical ensemble learning architecture for quantitative trading on the NSE. The system consists of three main subsystems: **Data Pipeline**, **Model Stack**, and **Execution Engine**.

---

## High-Level System Architecture

```mermaid
graph TB
    subgraph External["External Systems"]
        NSE[NSE Servers<br/>Market Data Feed]
    end
    
    subgraph DataPipeline["Data Pipeline Layer"]
        Fetch[fetch_all.py<br/>Data Acquisition]
        Silver[(Silver Lake<br/>Raw + Basic TA)]
        Gold[(Gold Lake<br/>Enhanced Features)]
        Process[preprocess.py<br/>Normalization Engine]
    end
    
    subgraph ModelStack["Model Stack Layer"]
        Base1[LSTM<br/>Temporal Attention]
        Base2[GRU<br/>Gated Recurrence]
        Base3[CNN<br/>ResNet 1D]
        Base4[XGBoost<br/>Gradient Trees]
        Base5[Random Forest<br/>Ensemble Trees]
        Judge[Judge Network<br/>Meta-Learner]
    end
    
    subgraph Execution["Execution Layer"]
        Live[live_swing_system_judge.py<br/>Trading Engine]
        Risk[Risk Management<br/>Position Sizing]
        Log[trade_log.csv<br/>Audit Trail]
    end
    
    NSE -->|REST API| Fetch
    Fetch --> Silver
    Silver -->|make_gold.py| Gold
    Gold --> Process
    Process --> Base1 & Base2 & Base3 & Base4 & Base5
    Base1 & Base2 & Base3 & Base4 & Base5 --> Judge
    Judge --> Live
    Live --> Risk --> Log
    
    style NSE fill:#e3f2fd
    style Judge fill:#fff3e0
    style Live fill:#e8f5e9
```

---

## Data Pipeline Architecture

### The Lake Architecture (Medallion Pattern)

```mermaid
graph LR
    subgraph Bronze["Bronze Layer (Implicit)"]
        API[NSE API Response<br/>Raw JSON/CSV]
    end
    
    subgraph Silver["Silver Layer"]
        S1[Raw OHLCV]
        S2[Basic Indicators<br/>RSI, EMA, Log Returns]
        S3[Data Validation<br/>NaN Handling]
    end
    
    subgraph Gold["Gold Layer"]
        G1[18 Technical Features<br/>MFI, MACD, ADX, ATR, OBV, BB]
        G2[Time Embeddings<br/>Sin/Cos Cyclical]
        G3[Lag Features<br/>Velocity Signals]
    end
    
    subgraph Platinum["Platinum Layer (Alpha Factory)"]
        P1[20 Elite Features<br/>Yang-Zhang Vol, Z-Scores]
        P2[MinMax Normalization<br/>Per-Stock Scaling]
        P3[Sequence Generation<br/>60-Day Windows]
    end
    
    API --> S1 --> S2 --> S3
    S3 --> G1 --> G2 --> G3
    G3 --> P1 --> P2 --> P3
    
    style Bronze fill:#cd7f32,color:#fff
    style Silver fill:#c0c0c0
    style Gold fill:#ffd700
    style Platinum fill:#e5e4e2
```

**Storage Locations**:
- **Silver**: `lake/silver/*.csv` (1900+ files, ~500MB)
- **Gold**: `lake/gold/*.csv` (1900+ files, ~800MB)
- **Platinum**: `lake/processed_market.npz` (Compressed, ~200MB)

---

## Model Stack Architecture

### Hierarchical Ensemble Design

The system implements a **two-tier ensemble**:

1. **Tier 1 (Base Learners)**: 5 heterogeneous models
2. **Tier 2 (Meta-Learner)**: Judge Network for model fusion

```mermaid
graph TD
    Input[Input Tensor<br/>Shape: Batch x 60 x 20] --> Split{Input Routing}
    
    Split -->|Sequence| LSTM[LSTM Model<br/>2-layer Bidirectional<br/>Hidden: 64]
    Split -->|Sequence| GRU[GRU Model<br/>2-layer<br/>Hidden: 64]
    Split -->|Sequence| CNN[CNN Model<br/>ResNet Blocks<br/>Channels: 32→64]
    Split -->|Flattened| XGB[XGBoost<br/>n_estimators: 100<br/>max_depth: 6]
    Split -->|Flattened| RF[Random Forest<br/>n_estimators: 100<br/>max_features: sqrt]
    
    LSTM --> Attention[Attention Layer<br/>Weighted Sum]
    Attention --> P1[Prediction 1]
    GRU --> P2[Prediction 2]
    CNN --> Pool[Global Avg Pool] --> P3[Prediction 3]
    XGB --> P4[Prediction 4]
    RF --> P5[Prediction 5]
    
    P1 & P2 & P3 & P4 & P5 --> Meta[Meta-Feature Vector<br/>5 Predictions]
    
    Meta --> Judge[Judge Network<br/>32-32-1 MLP<br/>ReLU Activation]
    Judge --> Output[Final Prediction<br/>Expected Return]
    
    style Judge fill:#ff6b35
    style Output fill:#004e89
```

### Model Specifications

| Model | Type | Input Shape | Output | Parameters |
|:------|:-----|:------------|:-------|:-----------|
| **LSTM** | Recurrent NN | (B, 60, 20) | (B, 1) | ~100K |
| **GRU** | Recurrent NN | (B, 60, 20) | (B, 1) | ~80K |
| **CNN** | Convolutional NN | (B, 20, 60) | (B, 1) | ~50K |
| **XGBoost** | Gradient Boosting | (B, 1200) | (B,) | ~500K |
| **Random Forest** | Ensemble Trees | (B, 1200) | (B,) | ~300K |
| **Judge** | Meta-MLP | (B, 5) | (B, 1) | 1.1K |

**Total Model Size**: ~350MB (PyTorch) + ~200MB (XGBoost/RF)

---

## Execution Architecture

### Live Trading System Flow

```mermaid
sequenceDiagram
    participant Market
    participant DataLoader
    participant ModelStack
    participant RiskMgmt
    participant Broker
    participant AuditLog
    
    Market->>DataLoader: Fetch Daily Data
    DataLoader->>DataLoader: Apply Features.py
    DataLoader->>DataLoader: Normalize (Scaler)
    DataLoader->>ModelStack: Feed 60-Day Sequences
    
    ModelStack->>ModelStack: LSTM/GRU/CNN Inference
    ModelStack->>ModelStack: XGB/RF Inference
    ModelStack->>ModelStack: Judge Network Fusion
    ModelStack->>RiskMgmt: Ranked Predictions
    
    RiskMgmt->>RiskMgmt: Filter by Confidence
    RiskMgmt->>RiskMgmt: Check Market Regime
    RiskMgmt->>RiskMgmt: Calculate Position Size
    RiskMgmt->>RiskMgmt: Validate ATR Limits
    
    RiskMgmt->>Broker: Execute Orders
    Broker->>AuditLog: Log Trade Details
    AuditLog->>AuditLog: Append to trade_log.csv
```

### Risk Management Layer

```mermaid
flowchart TD
    Signal[Trading Signal<br/>from Judge] --> R1{Regime Filter}
    R1 -->|Index < SMA200| Reject1[❌ Reject: Bear Market]
    R1 -->|Index >= SMA200| R2{Volatility Check}
    
    R2 -->|ATR > 3%| Reject2[❌ Reject: Too Volatile]
    R2 -->|ATR <= 3%| R3{Position Count}
    
    R3 -->|Count >= 5| Reject3[❌ Reject: Max Positions]
    R3 -->|Count < 5| R4{Capital Check}
    
    R4 -->|Insufficient| Reject4[❌ Reject: No Capital]
    R4 -->|Sufficient| Calc[Calculate Position Size<br/>Qty = NAV × 0.006 / 1.4×ATR]
    
    Calc --> Execute[✅ Execute Trade]
    Execute --> Monitor[Continuous Monitoring]
    
    Monitor --> Exit1{Stop Loss?}
    Monitor --> Exit2{Profit Target?}
    Monitor --> Exit3{Time Limit?}
    
    Exit1 -->|Hit| Close[Close Position]
    Exit2 -->|Hit| Close
    Exit3 -->|Exceeded| Close
    
    style Execute fill:#4caf50
    style Close fill:#ff5722
    style Reject1 fill:#757575
    style Reject2 fill:#757575
    style Reject3 fill:#757575
    style Reject4 fill:#757575
```

---

## Technology Stack

### Core Dependencies

```mermaid
graph LR
    subgraph ML["Machine Learning"]
        PT[PyTorch 2.0+<br/>Neural Networks]
        XG[XGBoost 2.0+<br/>Gradient Boosting]
        SK[Scikit-learn<br/>Random Forest & Scaling]
    end
    
    subgraph Data["Data Processing"]
        PD[Pandas<br/>Time Series]
        NP[NumPy<br/>Numerical Compute]
        TA[pandas_ta<br/>Technical Analysis]
    end
    
    subgraph IO["I/O & Monitoring"]
        YF[yfinance<br/>Market Data]
        JL[joblib<br/>Model Serialization]
        TQ[tqdm<br/>Progress Tracking]
    end
    
    PT -.-> NP
    XG -.-> NP
    SK -.-> NP
    PD -.-> NP
    TA -.-> PD
    YF -.-> PD
```

### Hardware Requirements

| Resource | Minimum | Recommended |
|:---------|:--------|:------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16 GB |
| **GPU** | None | CUDA-compatible (3GB+ VRAM) |
| **Storage** | 5 GB | 10 GB SSD |
| **Network** | Broadband | Low-latency (<50ms to NSE) |

---

## Design Patterns & Principles

### 1. Separation of Concerns
- **Data Layer**: Pure ETL, no business logic
- **Model Layer**: Stateless prediction functions
- **Execution Layer**: Isolated risk management

### 2. Idempotency
- All data processing scripts support resume functionality
- Model training uses fixed random seeds
- Trade logs are append-only (immutable)

### 3. Walk-Forward Validation
- Training window: 756 days (3 years)
- Testing window: 63 days (3 months)
- Prevents look-ahead bias and overfitting

### 4. Defensive Programming
- Null checks on all DataFrame operations
- MinMaxScaler per-stock to handle different scales
- Try-catch blocks in data acquisition loops

---

## Scalability Considerations

### Current Limitations
- **Single-threaded execution**: Live trading runs on one CPU core
- **In-memory data loading**: Full dataset must fit in RAM
- **Sequential model inference**: No batch parallelization

### Future Architecture (Proposed)

```mermaid
graph TB
    subgraph Ingest["Async Data Ingestion"]
        Q1[Kafka Queue<br/>Real-time Prices]
    end
    
    subgraph Compute["Distributed Compute"]
        W1[Worker 1<br/>LSTM]
        W2[Worker 2<br/>GRU/CNN]
        W3[Worker 3<br/>XGB/RF]
    end
    
    subgraph Orchestration["Orchestration Layer"]
        Redis[(Redis Cache<br/>Hot Features)]
        Scheduler[Airflow DAG<br/>Training Pipeline]
    end
    
    Q1 --> Redis
    Redis --> W1 & W2 & W3
    W1 & W2 & W3 --> Judge[Judge Service<br/>FastAPI]
    Scheduler -.->|Daily Retrain| W1 & W2 & W3
```

---

## Security & Compliance

### Data Security
- No API keys stored in code (environment variables)
- Trade logs are local-only (not transmitted)
- Model weights are versioned with git-lfs

### Regulatory Considerations
> [!WARNING]
> This system is designed for **research and backtesting**. Live trading requires:
> - Broker API integration with proper authentication
> - Compliance with SEBI (Securities and Exchange Board of India) regulations
> - Real-time risk monitoring and circuit breakers

---

## Monitoring & Observability

### Key Metrics Tracked

| Metric | Location | Purpose |
|:-------|:---------|:--------|
| **Sharpe Ratio** | `run_metrics.py` | Risk-adjusted returns |
| **Max Drawdown** | `train_full_pipeline.py` | Capital preservation |
| **Win Rate** | `trade_log.csv` | Strategy effectiveness |
| **Execution Latency** | Logs (stdout) | Performance optimization |

### Logging Strategy
- **INFO**: Normal operations, successful trades
- **WARNING**: Failed data fetches, skipped stocks
- **ERROR**: Model loading failures, critical exceptions

---

## References

- **Data Source**: NSE EQUITY_L.csv (Official Archive)
- **Model Inspiration**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Risk Framework**: "Safe Haven Investing" (Marks, 2011)
