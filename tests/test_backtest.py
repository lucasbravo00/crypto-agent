"""
Tests for src/backtest.py. No network: every test passes synthetic
candles in via the `candles` argument.
"""
import pytest

from src import backtest

DAY_MS = 24 * 60 * 60 * 1000
START_MS = 1672531200000  # 2023-01-01T00:00:00Z


def _candle(day_index, open_price, high, low, close):
    return [START_MS + day_index * DAY_MS, open_price, high, low, close, 0.0]


def _flat_candles(n, price):
    return [_candle(i, price, price, price, price) for i in range(n)]


# --- pure helpers -------------------------------------------------------

def test_combined_pnl_scales_with_leverage():
    bullets = [{"entry_price": 100.0, "collateral_usd": 1000.0}]
    # +10% price move at 5x = +50% on the position = +500 on 1000 collateral
    assert backtest._combined_pnl(bullets, 110.0, 5.0) == pytest.approx(500.0)


def test_target_price_single_bullet():
    bullets = [{"entry_price": 100.0, "collateral_usd": 1000.0}]
    # +15% combined at 5x needs a +3% price move
    assert backtest._target_price(bullets, 5.0, 15.0) == pytest.approx(103.0)


def test_target_price_averages_across_bullets():
    # Two equal-size bullets at 100 and 200: the combined target sits
    # between each one's individual target, weighted by size.
    bullets = [
        {"entry_price": 100.0, "collateral_usd": 1000.0},
        {"entry_price": 200.0, "collateral_usd": 1000.0},
    ]
    price = backtest._target_price(bullets, 5.0, 15.0)
    assert backtest._combined_pnl(bullets, price, 5.0) == pytest.approx(0.15 * 2000.0)


def test_liquidation_price_uses_full_round_balance():
    # Cross margin: the whole balance backs the position, so a round with
    # only 1 of 30 bullets deployed liquidates far below that bullet's own
    # isolated liquidation price (which at 5x would be -20%, i.e. 80).
    bullets = [{"entry_price": 100.0, "collateral_usd": 1000.0}]
    liq_full = backtest._liquidation_price(bullets, balance=30000.0, leverage=5.0)
    liq_thin = backtest._liquidation_price(bullets, balance=1000.0, leverage=5.0)
    assert liq_thin == pytest.approx(80.0)   # matches isolated math when balance == collateral
    assert liq_full < liq_thin                # more headroom with the unused budget backing it


# --- full simulation ----------------------------------------------------

def test_flat_market_only_bleeds_fees():
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_usd=10000.0,
        candles=_flat_candles(10, 100.0),
    )
    # No price movement -> no TP, no liquidation, just entry fees paid.
    assert result["rounds_take_profit"] == 0
    assert result["rounds_liquidated"] == 0
    assert result["round_still_open"] is True
    assert result["final_balance_usd"] < 10000.0
    assert result["total_return_pct"] < 0


def test_take_profit_closes_round_and_compounds():
    # Day 0 opens a bullet at 100. Day 1 rallies well past the +3% price
    # move a lone bullet needs for +15% combined at 5x.
    candles = [
        _candle(0, 100.0, 100.0, 100.0, 100.0),
        _candle(1, 100.0, 120.0, 100.0, 120.0),
    ]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-02", initial_balance_usd=10000.0, candles=candles,
    )
    assert result["rounds_take_profit"] >= 1
    assert result["final_balance_usd"] > 10000.0
    first_round = result["rounds"][0]
    assert first_round["outcome"] == "take_profit"
    # Booked at the target price, not at the day's high (which overshot).
    assert first_round["exit_price"] == pytest.approx(103.0, abs=0.5)


def test_liquidation_wipes_balance_and_stops_trading():
    # A crash deep enough to take equity to zero even with the full
    # 10k balance backing a growing position.
    candles = [_candle(0, 100.0, 100.0, 100.0, 100.0)]
    candles += [_candle(i, 100.0, 100.0, 100.0, 100.0) for i in range(1, 30)]
    candles.append(_candle(30, 100.0, 100.0, 1.0, 1.0))  # catastrophic drop
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-31", initial_balance_usd=10000.0, candles=candles,
    )
    assert result["rounds_liquidated"] == 1
    assert result["final_balance_usd"] == 0.0
    assert result["total_return_pct"] == -100.0


def test_liquidation_takes_precedence_over_take_profit_same_day():
    # One day whose range spans both the liquidation price and the target.
    # The conservative reading (documented) is that liquidation hit first.
    #
    # max_bullets_per_round=1 on purpose: it takes a fully-deployed round
    # for liquidation to be reachable at all (see the test below).
    candles = [
        _candle(0, 100.0, 100.0, 100.0, 100.0),
        _candle(1, 100.0, 500.0, 1.0, 100.0),
    ]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-02", initial_balance_usd=10000.0, candles=candles,
        max_bullets_per_round=1,
    )
    assert result["rounds_liquidated"] == 1
    assert result["rounds_take_profit"] == 0


def test_cross_margin_makes_a_thinly_deployed_round_unliquidatable():
    # The whole point of the 30-bullet budget under CROSS margin: with
    # only a couple of bullets deployed, the untouched balance backs the
    # position so heavily that the liquidation price goes NEGATIVE --
    # i.e. price would have to go below zero. Under ISOLATED margin those
    # same bullets would each die at -20%.
    bullets = [
        {"entry_price": 100.0, "collateral_usd": 10000.0 / 30},
        {"entry_price": 100.0, "collateral_usd": 10000.0 / 30},
    ]
    assert backtest._liquidation_price(bullets, balance=10000.0, leverage=5.0) < 0


def test_round_cap_limits_bullets_per_round():
    result = backtest.run_backtest(
        "2023-01-01", "2023-02-20", initial_balance_usd=10000.0,
        candles=_flat_candles(45, 100.0),  # more days than the cap
        max_bullets_per_round=30,
    )
    assert result["rounds"][0]["bullets_used"] == 30


def test_one_bullet_per_day_even_across_a_round_boundary():
    # Day 1 closes round 1 in profit. The next round's first bullet must
    # wait for day 2 -- the one-new-bullet-per-day rule is global, it
    # isn't reset by a round ending (mirrors bullets._opened_today()).
    candles = [
        _candle(0, 100.0, 100.0, 100.0, 100.0),
        _candle(1, 100.0, 120.0, 100.0, 120.0),   # round 1 hits TP here
        _candle(2, 120.0, 120.0, 120.0, 120.0),   # round 2's first bullet
    ]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-03", initial_balance_usd=10000.0, candles=candles,
    )
    assert result["rounds"][0]["bullets_used"] == 2   # days 0 and 1
    assert result["rounds"][1]["opened_at"] == "2023-01-03"
    assert result["rounds"][1]["bullets_used"] == 1


def test_empty_candles_raises():
    with pytest.raises(ValueError):
        backtest.run_backtest("2023-01-01", "2023-01-10", candles=[])
