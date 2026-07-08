"""
strategy_tools.py
------------------
Pure calculations (no network, no LLM) about the math of a leveraged
position. This is NOT financial advice: these are the objective numbers
any futures position implies, so decisions are made looking at the full
picture rather than only the upside.

Documented simplifications (important to understand):
- Assumes isolated margin and does not subtract open/close fees or
  funding rates, which on positions held for days can erode a relevant
  part of the result.
- Each exchange's real liquidation happens slightly BEFORE the
  theoretical approximation (100%/leverage), because the exchange
  reserves maintenance margin. `approx_liquidation_move_pct` is an
  optimistic ceiling, not BingX's exact number.
"""
from __future__ import annotations


def simulate_bullet_math(
    collateral_usd: float,
    entry_price: float,
    leverage: float = 5.0,
    target_position_gain_pct: float = 15.0,
) -> dict:
    """Compute the objective math of a long futures "bullet".

    Args:
        collateral_usd: Margin posted as collateral, in USD.
        entry_price: Asset entry price.
        leverage: Leverage multiplier (e.g. 5 for x5).
        target_position_gain_pct: Target gain ON THE POSITION
            (not on price), e.g. 15 for +15%.
    """
    if collateral_usd <= 0 or entry_price <= 0 or leverage <= 0:
        raise ValueError("collateral_usd, entry_price and leverage must be > 0")

    position_size_usd = collateral_usd * leverage

    # To gain X% on the position at leverage L, price must move X/L %
    # in your favor.
    required_price_move_pct = target_position_gain_pct / leverage
    target_price = entry_price * (1 + required_price_move_pct / 100)

    # Theoretical liquidation approximation (no maintenance margin):
    # with isolated margin, the position is liquidated when the loss
    # equals the collateral, i.e. an adverse move of ~100/L %.
    approx_liquidation_move_pct = -100.0 / leverage
    approx_liquidation_price = entry_price * (1 + approx_liquidation_move_pct / 100)

    profit_at_target_usd = collateral_usd * (target_position_gain_pct / 100)

    return {
        "collateral_usd": collateral_usd,
        "leverage": leverage,
        "position_size_usd": round(position_size_usd, 2),
        "entry_price": entry_price,
        "required_price_move_pct": round(required_price_move_pct, 2),
        "target_price": round(target_price, 2),
        "profit_at_target_usd": round(profit_at_target_usd, 2),
        "approx_liquidation_move_pct": round(approx_liquidation_move_pct, 2),
        "approx_liquidation_price": round(approx_liquidation_price, 2),
        "warning": (
            "Approximation without fees, funding or maintenance margin; "
            "real liquidation happens before the theoretical one. A drop of "
            f"{abs(round(approx_liquidation_move_pct, 1))}% wipes out 100% of this bullet's collateral."
        ),
    }
