"""
Tests for src/backtest.py. No network: every test passes synthetic
candles in via the `candles` argument.

The strategy is COIN-MARGINED (BTC collateral, BTC P&L) -- see the
module docstring. The math below is hand-derived from
PnL_btc = collateral * leverage * (1 - entry/price), not copied from
the implementation.
"""
import pytest

from src import backtest

DAY_MS = 24 * 60 * 60 * 1000
START_MS = 1672531200000  # 2023-01-01T00:00:00Z


def _candle(day_index, open_price, high, low, close):
    return [START_MS + day_index * DAY_MS, open_price, high, low, close, 0.0]


def _flat_candles(n, price):
    return [_candle(i, price, price, price, price) for i in range(n)]


# --- inverse contract math ---------------------------------------------

def test_combined_pnl_btc_is_capped_not_linear():
    bullets = [{"entry_price": 100.0, "collateral_btc": 1.0}]
    # +100% price move at 5x: PnL = 1 * 5 * (1 - 100/200) = 2.5 BTC,
    # NOT the 5 BTC a linear/USDT-margined intuition would suggest.
    assert backtest._combined_pnl_btc(bullets, 200.0, 5.0) == pytest.approx(2.5)
    # As price runs away, BTC gains asymptote at collateral * leverage.
    assert backtest._combined_pnl_btc(bullets, 1e9, 5.0) == pytest.approx(5.0, rel=1e-6)


def test_combined_pnl_btc_zero_at_entry():
    bullets = [{"entry_price": 100.0, "collateral_btc": 1.0}]
    assert backtest._combined_pnl_btc(bullets, 100.0, 5.0) == pytest.approx(0.0)


def test_target_price_btc_needs_more_than_the_linear_move():
    bullets = [{"entry_price": 100.0, "collateral_btc": 1.0}]
    price = backtest._target_price_btc(bullets, 5.0, 15.0)
    # Linear intuition says +3%; inverse needs slightly more.
    assert price == pytest.approx(100.0 * 5 / 4.85)
    assert price > 103.0
    # Sanity: at that price the position really is +15% in BTC.
    assert backtest._combined_pnl_btc(bullets, price, 5.0) == pytest.approx(0.15)


def test_target_price_usd_triggers_earlier_than_btc():
    bullets = [{"entry_price": 100.0, "collateral_btc": 1.0}]
    btc_target = backtest._target_price_btc(bullets, 5.0, 15.0)
    usd_target = backtest._target_price_usd(bullets, 5.0, 15.0)
    # Part of a USD gain comes from the collateral itself appreciating,
    # so the USD target is reached at a lower price.
    assert usd_target < btc_target


def test_target_price_usd_really_is_15pct_in_usd():
    bullets = [{"entry_price": 100.0, "collateral_btc": 1.0}]
    price = backtest._target_price_usd(bullets, 5.0, 15.0)
    equity_btc = 1.0 + backtest._combined_pnl_btc(bullets, price, 5.0)
    usd_committed = 1.0 * 100.0
    assert equity_btc * price / usd_committed == pytest.approx(1.15)


def test_liquidation_at_16_67_pct_when_fully_deployed():
    # Coin-margined 5x liquidates at -16.67%, not the -20% a USDT-margined
    # position would: the collateral loses USD value on the way down too.
    bullets = [{"entry_price": 100.0, "collateral_btc": 1.0}]
    liq = backtest._liquidation_price(bullets, balance_btc=1.0, leverage=5.0)
    assert liq == pytest.approx(100.0 * 5 / 6)
    assert liq == pytest.approx(83.333, abs=0.01)


def test_cross_margin_makes_a_thinly_deployed_round_far_safer():
    # Only 2 of 30 bullets deployed: the untouched balance backs the
    # position, pushing liquidation far below the fully-deployed -16.67%.
    bullets = [
        {"entry_price": 100.0, "collateral_btc": 1.0 / 30},
        {"entry_price": 100.0, "collateral_btc": 1.0 / 30},
    ]
    liq = backtest._liquidation_price(bullets, balance_btc=1.0, leverage=5.0)
    assert liq < 30.0


# --- cycle-top position sizing ------------------------------------------

def test_size_factor_full_below_threshold():
    assert backtest._size_factor(1.0) == 1.0
    assert backtest._size_factor(backtest.MAYER_FULL_SIZE) == 1.0


def test_size_factor_zero_at_cycle_top():
    assert backtest._size_factor(backtest.MAYER_STOP) == 0.0
    assert backtest._size_factor(3.0) == 0.0


def test_size_factor_scales_linearly_in_between():
    midpoint = (backtest.MAYER_FULL_SIZE + backtest.MAYER_STOP) / 2
    assert backtest._size_factor(midpoint) == pytest.approx(0.5)


def test_size_factor_unknown_mayer_is_full_size():
    assert backtest._size_factor(None) == 1.0


# --- full simulation ----------------------------------------------------

def test_flat_market_only_bleeds_fees():
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_btc=1.0,
        candles=_flat_candles(10, 100.0),
    )
    assert result["rounds_take_profit"] == 0
    assert result["rounds_liquidated"] == 0
    assert result["round_still_open"] is True
    assert result["final_balance_btc"] < 1.0  # entry fees only


def test_take_profit_closes_round_and_compounds_btc():
    candles = [
        _candle(0, 100.0, 100.0, 100.0, 100.0),
        _candle(1, 100.0, 120.0, 100.0, 120.0),
    ]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-02", initial_balance_btc=1.0, candles=candles,
    )
    assert result["rounds_take_profit"] >= 1
    assert result["final_balance_btc"] > 1.0
    assert result["rounds"][0]["outcome"] == "take_profit"


def test_liquidation_wipes_balance():
    candles = _flat_candles(30, 100.0)
    candles.append(_candle(30, 100.0, 100.0, 1.0, 1.0))
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-31", initial_balance_btc=1.0, candles=candles,
    )
    assert result["rounds_liquidated"] == 1
    assert result["final_balance_btc"] == 0.0
    assert result["btc_return_pct"] == -100.0


def test_liquidation_takes_precedence_over_take_profit_same_day():
    # max_bullets_per_round=1 so the round is fully deployed and therefore
    # liquidatable at all (see test_cross_margin_makes_a_thinly_deployed...).
    candles = [
        _candle(0, 100.0, 100.0, 100.0, 100.0),
        _candle(1, 100.0, 500.0, 1.0, 100.0),
    ]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-02", initial_balance_btc=1.0, candles=candles,
        max_bullets_per_round=1,
    )
    assert result["rounds_liquidated"] == 1
    assert result["rounds_take_profit"] == 0


def test_round_cap_limits_bullets_per_round():
    result = backtest.run_backtest(
        "2023-01-01", "2023-02-20", initial_balance_btc=1.0,
        candles=_flat_candles(45, 100.0), max_bullets_per_round=30,
    )
    assert result["rounds"][0]["bullets_used"] == 30


def test_one_bullet_per_day_even_across_a_round_boundary():
    # Day 1 closes round 1 in profit. The next round's first bullet must
    # wait for day 2 -- the one-new-bullet-per-day rule is global and is
    # not reset by a round ending (mirrors bullets._opened_today()).
    candles = [
        _candle(0, 100.0, 100.0, 100.0, 100.0),
        _candle(1, 100.0, 120.0, 100.0, 120.0),
        _candle(2, 120.0, 120.0, 120.0, 120.0),
    ]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-03", initial_balance_btc=1.0, candles=candles,
    )
    assert result["rounds"][0]["bullets_used"] == 2
    assert result["rounds"][1]["opened_at"] == "2023-01-03"
    assert result["rounds"][1]["bullets_used"] == 1


def _hot_market():
    """Warmup at a low price, then a market trading at 3x it -> Mayer well
    above MAYER_STOP."""
    warmup = [_candle(-200 + i, 100.0, 100.0, 100.0, 100.0) for i in range(200)]
    return warmup, _flat_candles(10, 300.0)


def test_derisk_bullet_size_stops_opening_when_mayer_is_extreme():
    warmup, candles = _hot_market()
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_btc=1.0,
        candles=candles, warmup_candles=warmup, derisk_mode="bullet_size",
    )
    assert result["rounds_total"] == 0
    assert result["final_balance_btc"] == 1.0  # untouched, not even fees


def test_derisk_off_opens_normally_in_the_same_hot_market():
    warmup, candles = _hot_market()
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_btc=1.0,
        candles=candles, warmup_candles=warmup, derisk_mode="off",
    )
    assert result["rounds"][0]["bullets_used"] == 10


def test_derisk_withdraw_moves_btc_out_of_the_trading_account():
    warmup, candles = _hot_market()
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_btc=1.0,
        candles=candles, warmup_candles=warmup, derisk_mode="withdraw",
    )
    # Mayer past MAYER_STOP -> exposure target is 0, so everything is
    # banked and nothing is left in the trading account to lose.
    assert result["banked_btc"] == pytest.approx(1.0)
    assert result["trading_balance_btc"] == pytest.approx(0.0)
    assert result["final_balance_btc"] == pytest.approx(1.0)


def test_withdraw_protects_profit_from_a_liquidation_but_bullet_size_does_not():
    # Warmup low so Mayer starts hot, then a catastrophic drop. Under
    # cross margin a liquidation takes the whole TRADING balance -- only
    # BTC already moved out survives. This is the core finding.
    warmup = [_candle(-200 + i, 100.0, 100.0, 100.0, 100.0) for i in range(200)]
    # Mayer ~1.95 (midpoint) -> factor 0.5: half is banked, half at risk.
    candles = [_candle(i, 195.0, 195.0, 195.0, 195.0) for i in range(30)]
    candles.append(_candle(30, 195.0, 195.0, 1.0, 1.0))  # wipeout

    withdrawn = backtest.run_backtest(
        "2023-01-01", "2023-01-31", initial_balance_btc=1.0,
        candles=candles, warmup_candles=warmup, derisk_mode="withdraw",
    )
    sized = backtest.run_backtest(
        "2023-01-01", "2023-01-31", initial_balance_btc=1.0,
        candles=candles, warmup_candles=warmup, derisk_mode="bullet_size",
    )
    assert withdrawn["rounds_liquidated"] == 1
    assert sized["rounds_liquidated"] == 1
    # Smaller bullets did NOT protect the balance sitting behind them.
    assert sized["final_balance_btc"] == pytest.approx(0.0)
    # Withdrawn BTC survived the same liquidation.
    assert withdrawn["final_balance_btc"] > 0.4


def _bullets_opened(result):
    return sum(r["bullets_used"] for r in result["rounds"])


# An unreachable target keeps take-profits from ending rounds mid-test,
# so these assertions isolate the brake's behavior and nothing else.
NO_TP = 10_000.0


def test_derisk_drawdown_stops_opening_while_price_is_off_its_high():
    # Day 0 sets a high of 200; days 1-9 sit at 150, i.e. -25% below it.
    # With a -10% brake, only day 0 may open a bullet.
    candles = [_candle(0, 100.0, 200.0, 100.0, 200.0)]
    candles += [_candle(i, 150.0, 150.0, 150.0, 150.0) for i in range(1, 10)]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_btc=1.0, candles=candles,
        derisk_mode="drawdown", drawdown_lookback_days=30, drawdown_stop_pct=10.0,
        target_gain_pct=NO_TP,
    )
    assert _bullets_opened(result) == 1


def test_derisk_drawdown_resumes_once_price_recovers():
    # Same dip, then a recovery to 199 (only -0.5% off the high) -> the
    # brake releases and bullets start opening again.
    candles = [_candle(0, 100.0, 200.0, 100.0, 200.0)]
    candles += [_candle(i, 150.0, 150.0, 150.0, 150.0) for i in range(1, 5)]
    candles += [_candle(i, 199.0, 199.0, 199.0, 199.0) for i in range(5, 10)]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_btc=1.0, candles=candles,
        derisk_mode="drawdown", drawdown_lookback_days=30, drawdown_stop_pct=10.0,
        target_gain_pct=NO_TP,
    )
    assert _bullets_opened(result) == 6  # day 0, plus days 5-9


def test_derisk_drawdown_does_not_peek_at_todays_high():
    # Day 0's own high (200) must NOT gate day 0's own open (100): at the
    # daily open that high hasn't happened yet. Guarding against lookahead
    # bias, which would otherwise flatter every drawdown result.
    candles = [_candle(0, 100.0, 200.0, 100.0, 200.0)]
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-01", initial_balance_btc=1.0, candles=candles,
        derisk_mode="drawdown", drawdown_lookback_days=30, drawdown_stop_pct=10.0,
        target_gain_pct=NO_TP,
    )
    assert _bullets_opened(result) == 1


def test_derisk_drawdown_is_a_no_op_in_a_market_that_never_dips():
    candles = _flat_candles(10, 100.0)
    braked = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_btc=1.0, candles=candles,
        derisk_mode="drawdown", drawdown_stop_pct=10.0,
    )
    plain = backtest.run_backtest(
        "2023-01-01", "2023-01-10", initial_balance_btc=1.0, candles=candles,
        derisk_mode="off",
    )
    assert braked["rounds"][0]["bullets_used"] == plain["rounds"][0]["bullets_used"]


def test_invalid_derisk_mode_raises():
    with pytest.raises(ValueError):
        backtest.run_backtest("2023-01-01", "2023-01-10",
                              candles=_flat_candles(3, 100.0), derisk_mode="maybe")


def test_invalid_target_mode_raises():
    with pytest.raises(ValueError):
        backtest.run_backtest("2023-01-01", "2023-01-10",
                              candles=_flat_candles(3, 100.0), target_mode="eur")


def test_empty_candles_raises():
    with pytest.raises(ValueError):
        backtest.run_backtest("2023-01-01", "2023-01-10", candles=[])
