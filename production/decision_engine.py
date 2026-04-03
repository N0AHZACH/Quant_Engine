from production.trade_gate import generate_positions


def decide_trades(
    edges: dict,
    vols: dict,
    capital: float,
):
    """
    Converts model outputs into executable positions.
    """

    positions = generate_positions(
        edges=edges,
        vols=vols,
        capital=capital,
    )

    return positions
