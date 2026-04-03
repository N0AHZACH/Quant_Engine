# Quant_Engine
A small ML based quantitative engine to trade the NSE

SYSTEM NAME: NSE Cross-Sectional Swing Alpha

ASSET UNIVERSE:
- NSE listed equities only
- Survivorship-bias aware universe

TRADING STYLE:
- Cross-sectional
- Market-neutral (long / short)

HOLDING PERIOD:
- Fixed 3 trading days

PREDICTION TARGET:
- Log return over 3 days
  target = log(Close[t+3] / Close[t])

OBJECTIVE:
- Capital compounding with strict drawdown control

RISK CONSTRAINTS:
- Max drawdown target: 15–20%
- Capital survival > return maximization

TRADE FREQUENCY:
- Selective
- Zero-trade days are valid and expected
