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


# An unreachable-but-sane target keeps take-profits from ending rounds
# mid-test. Must stay below leverage*100 (500 at 5x): _target_price_btc's
# denominator is `leverage - target_gain_pct/100`, which goes negative
# (and the "target" price with it) above that -- confirmed the hard way.
NO_TP = 400.0


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


def test_invalid_entry_timing_raises():
    with pytest.raises(ValueError):
        backtest.run_backtest("2023-01-01", "2023-01-10",
                              candles=_flat_candles(3, 100.0), entry_timing="whenever")


# --- Intraday entry timing -----------------------------------------------

FIFTEEN_MIN_MS = 15 * 60 * 1000


def _intraday(day_index, slot, open_price, high, low, close):
    ts = START_MS + day_index * DAY_MS + slot * FIFTEEN_MIN_MS
    return [ts, open_price, high, low, close, 0.0]


def test_rsi_series_all_gains_is_100():
    closes = [100 + i for i in range(20)]
    rsi = backtest._rsi_series(closes, period=5)
    assert rsi[5] == 100.0
    assert rsi[-1] == 100.0


def test_rsi_series_all_losses_is_0():
    closes = [100 - i for i in range(20)]
    rsi = backtest._rsi_series(closes, period=5)
    assert rsi[5] == 0.0
    assert rsi[-1] == 0.0


def test_rsi_series_none_before_enough_history():
    closes = [100 + i for i in range(10)]
    rsi = backtest._rsi_series(closes, period=14)
    assert all(v is None for v in rsi)


def test_entry_prices_day_low_picks_the_days_minimum():
    candles = [
        _intraday(0, 0, 100, 105, 98, 102),
        _intraday(0, 1, 102, 103, 90, 95),   # new low: 90
        _intraday(0, 2, 95, 96, 92, 94),
    ]
    prices = backtest._entry_prices_by_day(candles, "day_low", rsi_period=14, rsi_threshold=30)
    assert prices["2023-01-01"] == 90


def test_entry_prices_rsi_oversold_falls_back_to_last_close_when_never_triggered():
    # Strictly rising all day -> RSI stays pinned high, signal never fires.
    candles = [_intraday(0, i, 100 + i, 100 + i + 1, 100 + i - 1, 100 + i) for i in range(20)]
    prices = backtest._entry_prices_by_day(candles, "rsi_oversold", rsi_period=14, rsi_threshold=30)
    assert prices["2023-01-01"] == candles[-1][4]


def test_entry_prices_rsi_oversold_triggers_mid_decline():
    # A short flat warmup (to seed RSI) followed by a sharp decline should
    # trigger the signal WHILE the decline is happening -- not at the
    # day's open, and not by falling back to the day's last close.
    closes = [100] * 6 + [99, 97, 94, 90, 86, 82, 78]
    candles = [_intraday(0, i, c, c + 0.5, c - 0.5, c) for i, c in enumerate(closes)]
    prices = backtest._entry_prices_by_day(candles, "rsi_oversold", rsi_period=5, rsi_threshold=30)
    entry = prices["2023-01-01"]
    assert entry < closes[0]
    assert entry != closes[-1]


def test_rolling_std_of_constant_series_is_zero():
    std = backtest._rolling_std([100.0] * 10, period=5)
    assert std[4] == 0.0
    assert std[9] == 0.0


def test_rolling_std_matches_hand_computation():
    # [1,2,3,4,5]: mean=3, population variance=2, std=sqrt(2)
    std = backtest._rolling_std([1.0, 2.0, 3.0, 4.0, 5.0], period=5)
    assert std[4] == pytest.approx(2 ** 0.5)


def test_bollinger_lower_band_below_sma_when_volatile():
    closes = [100.0, 110.0, 90.0, 105.0, 95.0, 100.0]
    band = backtest._bollinger_lower_series(closes, period=5, num_std=2.0)
    sma = backtest._rolling_sma(closes, period=5)
    assert band[4] < sma[4]


def test_bollinger_lower_band_equals_sma_when_flat():
    band = backtest._bollinger_lower_series([100.0] * 10, period=5, num_std=2.0)
    assert band[4] == 100.0  # zero volatility -> band collapses onto the mean


def test_entry_prices_bollinger_triggers_on_a_volatility_spike():
    # Flat warmup (band hugs the mean at ~100), then a sharp one-candle
    # drop clearly below the lower band -- should trigger THAT candle's
    # close, not fall back to the day's last close.
    closes = [100.0] * 10 + [80.0, 90.0, 95.0]
    candles = [_intraday(0, i, c, c + 0.5, c - 0.5, c) for i, c in enumerate(closes)]
    prices = backtest._entry_prices_by_day(
        candles, "bollinger_lower", bollinger_period=10, bollinger_std=2.0,
    )
    entry = prices["2023-01-01"]
    assert entry == 80.0


def test_entry_prices_bollinger_falls_back_when_never_triggered():
    closes = [100.0 + i * 0.01 for i in range(15)]  # near-flat, tiny drift
    candles = [_intraday(0, i, c, c + 0.05, c - 0.05, c) for i, c in enumerate(closes)]
    prices = backtest._entry_prices_by_day(
        candles, "bollinger_lower", bollinger_period=10, bollinger_std=2.0,
    )
    assert prices["2023-01-01"] == candles[-1][4]


def test_run_backtest_accepts_bollinger_lower_entry_timing():
    daily = [_candle(0, 100.0, 100.0, 100.0, 100.0)]
    intraday = [_intraday(0, i, 100.0, 100.5, 99.5, 100.0) for i in range(10)]
    intraday.append(_intraday(0, 10, 100.0, 100.5, 70.0, 75.0))  # sharp drop
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-01", initial_balance_btc=1.0,
        candles=daily, entry_timing="bollinger_lower", intraday_candles=intraday,
        bollinger_period=10, target_gain_pct=NO_TP, derisk_mode="off",
    )
    assert result["rounds"][0]["bullets_used"] == 1
    assert result["entry_timing"] == "bollinger_lower"


def test_entry_prices_split_correctly_across_multiple_days():
    candles = [
        _intraday(0, 0, 100, 100, 90, 95),
        _intraday(1, 0, 200, 200, 150, 180),
    ]
    prices = backtest._entry_prices_by_day(candles, "day_low", rsi_period=14, rsi_threshold=30)
    assert prices["2023-01-01"] == 90
    assert prices["2023-01-02"] == 150


def test_run_backtest_day_low_entry_beats_fixed_open_entry():
    # Outer daily view is flat (isolates the entry-price mechanism from
    # liquidation/target checks, which use the daily candle separately).
    # Real intraday movement -- a dip to 80 -- only exists in the 15m data.
    daily = [_candle(0, 100.0, 100.0, 100.0, 100.0)]
    intraday = [
        _intraday(0, 0, 100, 100, 100, 100),
        _intraday(0, 1, 100, 100, 80, 85),
        _intraday(0, 2, 85, 90, 85, 90),
    ]
    fixed = backtest.run_backtest(
        "2023-01-01", "2023-01-01", initial_balance_btc=1.0,
        candles=daily, entry_timing="fixed", target_gain_pct=NO_TP, derisk_mode="off",
    )
    day_low = backtest.run_backtest(
        "2023-01-01", "2023-01-01", initial_balance_btc=1.0,
        candles=daily, entry_timing="day_low", intraday_candles=intraday,
        target_gain_pct=NO_TP, derisk_mode="off",
    )
    # Same reference close (100) for both -- day_low bought lower (80 vs
    # 100), so it must show strictly better P&L.
    assert day_low["rounds"][0]["end_balance_btc"] > fixed["rounds"][0]["end_balance_btc"]


def test_run_backtest_day_without_intraday_coverage_falls_back_to_daily_open():
    # Two simulated days; intraday data only covers the first. The second
    # day's bullet must still open (at the daily open) instead of the
    # missing-coverage gap silently dropping that day's bullet.
    daily = [
        _candle(0, 100.0, 100.0, 100.0, 100.0),
        _candle(1, 200.0, 200.0, 200.0, 200.0),
    ]
    intraday = [_intraday(0, 0, 100, 100, 90, 95)]  # day 2 has no coverage
    result = backtest.run_backtest(
        "2023-01-01", "2023-01-02", initial_balance_btc=1.0,
        candles=daily, entry_timing="day_low", intraday_candles=intraday,
        target_gain_pct=NO_TP, derisk_mode="off",
    )
    assert sum(r["bullets_used"] for r in result["rounds"]) == 2


def test_run_backtest_raises_if_no_intraday_data_at_all():
    daily = [_candle(0, 100.0, 100.0, 100.0, 100.0)]
    with pytest.raises(ValueError):
        backtest.run_backtest(
            "2023-01-01", "2023-01-01", initial_balance_btc=1.0,
            candles=daily, entry_timing="day_low", intraday_candles=[],
        )
