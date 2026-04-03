def apply_transaction_costs(
    positions: dict,
    prices: dict,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
):
    """
    Applies transaction costs & slippage.

    cost_bps     = broker + fees
    slippage_bps = market impact
    """

    total_cost = 0.0

    for symbol, size in positions.items():
        price = prices.get(symbol)
        if price is None:
            continue

        notional = abs(size)
        cost = notional * (cost_bps + slippage_bps) / 10_000
        total_cost += cost

    return total_cost
