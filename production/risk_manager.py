def enforce_portfolio_limits(
    positions: dict,
    capital: float,
    max_gross: float = 1.2,
    max_net: float = 0.2,
):
    """
    Enforces portfolio-level constraints.
    """

    gross = sum(abs(v) for v in positions.values())
    net = sum(positions.values())

    max_gross_dollars = capital * max_gross
    max_net_dollars = capital * max_net

    scale = 1.0

    if gross > max_gross_dollars:
        scale = min(scale, max_gross_dollars / gross)

    if abs(net) > max_net_dollars:
        scale = min(scale, max_net_dollars / abs(net))

    if scale < 1.0:
        for k in positions:
            positions[k] *= scale

    return positions
