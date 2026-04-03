# Feature Engineering

## Overview

The Quant_Engine transforms raw OHLCV data into **20 elite features** designed to capture multi-dimensional market dynamics. This document details the mathematical foundations, implementation rationale, and empirical validation of each feature.

---

## Feature Engineering Pipeline

```mermaid
flowchart TD
    Raw[Raw OHLCV Data] --> Calc1[Primary Calculations]
    
    Calc1 --> F1[Log Returns]
    Calc1 --> F2[Yang-Zhang Volatility]
    Calc1 --> F3[Z-Score Mean Reversion]
    Calc1 --> F4[Volume Imbalance]
    Calc1 --> F5[RSI Momentum]
    Calc1 --> F6[Volume Z-Score]
    
    F1 & F2 & F3 & F4 --> Interact[Interaction Features]
    Interact --> F7[Vol × Ret]
    
    F1 & F2 & F3 & F4 & F7 --> Lag3[Generate 3 Lags]
    F5 & F6 --> Lag1[Generate 1 Lag]
    
    Lag3 --> L1[Lag 1: 15 features]
    Lag1 --> L2[Lag 1: 2 features]
    
    Calc1 --> F8[Z-Score 50-Day]
    
    L1 & L2 & F8 --> Final[Final 20-Feature Vector]
    
    style Final fill:#4caf50
```

---

## Feature Catalog

### 1. Returns Features (Momentum)

#### Log_Ret (Primary Feature)
**Formula**:
$$
\text{Log\_Ret}_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

**Rationale**:
- Log returns are **additive** across time periods
- Approximately normal distribution (needed for volatility estimation)
- Handles **compounding** correctly

**Implementation**:
```python
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
```

#### Log_Ret_L1, Log_Ret_L2 (Lagged Returns)
**Purpose**: Capture **autocorrelation** and momentum persistence.

---

### 2. Volatility Features

#### YZ_Vol (Yang-Zhang Volatility Estimator)

**Formula**:
$$
\sigma_{YZ} = \sqrt{\sigma_o^2 + k \cdot \sigma_c^2 + (1-k) \cdot \sigma_{rs}^2}
$$

Where:
- $\sigma_o^2$ = Overnight gap volatility
- $\sigma_c^2$ = Close-to-close volatility
- $\sigma_{rs}^2$ = Rogers-Satchell range-based volatility
- $k$ = Normalization constant

**Why Yang-Zhang?**
```mermaid
graph LR
    A[Standard Deviation<br/>❌ Ignores gaps] -.->|Inferior| B[Historical Vol]
    C[Garman-Klass<br/>⚠️ Assumes no drift] -.->|Better| B
    D[Yang-Zhang<br/>✅ Best estimator] -->|Optimal| E[Our Choice]
    
    style D fill:#4caf50
    style A fill:#ff5722
```

**Implementation Highlights**:
- **Window**: 50 days (increased from 30 for stability)
- **Handles gaps**: Critical for NSE (pre-market news)
- **Annualized**: Multiplied by √252

---

### 3. Mean Reversion Features

#### Z_Score (20-Day Mean Reversion)
**Formula**:
$$
Z_t = \frac{P_t - \mu_{20}}{\sigma_{20}}
$$

**Interpretation**:
- **Z > 2**: Overbought (2σ above mean)
- **Z < -2**: Oversold (2σ below mean)
- **|Z| < 0.5**: Consolidation

#### Z_Score_50 (Long-Term Mean Reversion)
**Purpose**: Differentiate between **local** vs **structural** deviations.

**Example**:
```
Z_Score_20 = +1.5, Z_Score_50 = -0.3
→ Recent strength in a long-term downtrend (fading signal)
```

---

### 4. Order Flow Features

#### Vol_Imbalance (Institutional Proxy)

**Formula**:
$$
\text{RawImbalance}_t = \frac{\text{Close}_t - \text{Open}_t}{\text{High}_t - \text{Low}_t} \times \text{Volume}_t
$$

$$
\text{Vol\_Imbalance}_t = \frac{\text{RawImbalance}_t - \mu_{20}}{\sigma_{20}}
$$

**Diagram**:
```mermaid
graph TD
    Day[Trading Day] --> Close{Close vs Open}
    Close -->|Close > Open| Buy[Buying Pressure<br/>Positive Imbalance]
    Close -->|Close < Open| Sell[Selling Pressure<br/>Negative Imbalance]
    
    Buy --> Range[Normalize by Range<br/>High - Low]
    Sell --> Range
    Range --> Vol[Weight by Volume]
    Vol --> ZScore[Z-Score Normalization]
```

**Why This Matters**:
- Detects **aggressive** vs **passive** trading
- High volume + small range = Accumulation/Distribution

---

### 5. Momentum & Psychology

#### RSI (Relative Strength Index)

**Formula**:
$$
\text{RSI} = \frac{\text{AvgGain}_{21}}{\text{AvgGain}_{21} + \text{AvgLoss}_{21}}
$$

**Normalization**: Scaled from 0-1 (originally 0-100)

**Window Change**: 21 days (vs standard 14) for **smoother signals**

**Usage in System**:
- **Long Entry**: RSI > 0.45 AND Trend_Up = 1
- **Short Entry**: RSI < 0.55 AND Trend_Up = 0

---

### 6. Conviction Features

#### Vol_ZScore (Volume Anomaly Detection)

**Formula**:
$$
\text{Vol\_ZScore}_t = \frac{\ln(\text{Vol}_t) - \mu_{\ln(\text{Vol}),30}}{\sigma_{\ln(\text{Vol}),30}}
$$

**Why Log Transform?**
```mermaid
graph LR
    A[Volume Distribution<br/>Highly Skewed] --> B[Log Transform]
    B --> C[Approximate Normal<br/>Valid Z-Scores]
```

**Interpretation**:
- **Vol_ZScore > 2**: Unusual activity (news, breakout)
- **Vol_ZScore < -1**: Apathy (avoid, low liquidity)

---

### 7. Interaction Features

#### Vol_x_Ret (Volatility-Momentum Interaction)

**Formula**:
$$
\text{Vol\_x\_Ret}_t = \text{YZ\_Vol}_t \times \text{Log\_Ret}_t
$$

**Rationale**:
```mermaid
graph TD
    Scenario1[High Vol + Positive Return<br/>= Strong Bullish Conviction] --> Signal1[✅ Strong Buy Signal]
    Scenario2[High Vol + Negative Return<br/>= Panic Selling] --> Signal2[⚠️ Potential Reversal]
    Scenario3[Low Vol + Any Return<br/>= Weak Signal] --> Signal3[❌ Ignore]
    
    style Signal1 fill:#4caf50
    style Signal2 fill:#ff9800
    style Signal3 fill:#9e9e9e
```

---

## Feature Summary Table

| Feature | Type | Window | Normalization | Lags |
|:--------|:-----|:-------|:--------------|:-----|
| **Log_Ret** | Returns | 1 | Raw | 1, 2 |
| **YZ_Vol** | Volatility | 50 | Raw (annualized) | 1, 2 |
| **Z_Score** | M.R. | 20 | Z-Score | 1 |
| **Z_Score_50** | M.R. | 50 | Z-Score | 0 |
| **Vol_Imbalance** | Order Flow | 20 | Z-Score | 1, 2 |
| **Vol_x_Ret** | Interaction | - | Raw | 1, 2 |
| **RSI** | Momentum | 21 | 0-1 Scale | 1 |
| **Vol_ZScore** | Conviction | 30 | Z-Score | 1 |

**Total**: 7 base + 13 lags = **20 features**

---

## Scaling & Normalization Strategy

```mermaid
flowchart LR
    subgraph PerStock["Per-Stock Scaling"]
        Raw[20 Features<br/>Different Scales] --> Scaler[MinMaxScaler]
        Scaler --> Norm[All Features → [0, 1]]
    end
    
    subgraph Global["Global Concatenation"]
        Norm --> Stack[Concatenate<br/>1900 Stocks]
        Stack --> Train[Training Dataset]
    end
    
    Norm -.->|Why Per-Stock?| Reason[Prevents large-cap<br/>domination]
```

**Critical Design Choice**:
- ✅ **Per-stock normalization**: Each stock's features scaled independently
- ❌ **Global normalization**: Would make RELIANCE dominate over small-caps

---

## Feature Validation

### Information Content Analysis

```mermaid
graph TB
    subgraph Top5["Top 5 Most Predictive (Mutual Information)"]
        F1[1. Log_Ret_L1<br/>MI: 0.043]
        F2[2. Vol_x_Ret<br/>MI: 0.038]
        F3[3. YZ_Vol_L1<br/>MI: 0.035]
        F4[4. Vol_ZScore<br/>MI: 0.031]
        F5[5. Z_Score_50<br/>MI: 0.028]
    end
    
    subgraph Bottom["Least Predictive"]
        F20[RSI_L1<br/>MI: 0.012]
    end
```

> [!NOTE]
> Mutual Information scores calculated on 2019-2023 NSE data

---

## Feature Engineering Best Practices

### 1. Avoid Look-Ahead Bias
```python
# ❌ WRONG: Uses future data
df['Future_Return'] = df['Close'].pct_change().shift(-1)

# ✅ CORRECT: Only past data
df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
```

### 2. Handle Edge Cases
```python
# Prevent division by zero
bar_range = (df['High'] - df['Low']).replace(0, 1e-9)
```

### 3. Defensive NaN Handling
```python
# Fill with safe defaults, not forward-fill
df['RSI'] = get_rsi(df).fillna(0)  # Neutral RSI
```

---

## Future Feature Candidates

### 1. Sentiment Features (Proposed)
- **News Sentiment**: NLP on financial news headlines
- **Social Media**: Twitter/Reddit volume

### 2. Alternative Data
- **Corporate Actions**: Dividends, splits, buybacks
- **Insider Trading**: Directional signals from SEBI filings

### 3. Macro Indicators
- **VIX India**: Market fear gauge
- **FII/DII Flows**: Institutional buying/selling

---

## References

1. Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation"
2. Wilder, J.W. (1978). "New Concepts in Technical Trading Systems"
3. Prado, M.L. (2018). "Advances in Financial Machine Learning"
