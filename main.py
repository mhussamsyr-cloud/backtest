"""
BACKTEST v10.0 — LONGS ONLY EXPERIMENT
==========================================
Changes from v9:
  1. LONG_ONLY = True — shorts disabled entirely
       Longs: 84.6% WR, +1.393% avg  (v9)
       Shorts: 78.0% WR, +0.538% avg (v9) — not worth the drag
  2. Output changed to CSV (no xlsxwriter/openpyxl dependency)
  3. MIN_COOLDOWN_CANDLES raised 3 → 4 to further reduce churn
       on long-only mode (less signal pressure without shorts)

Hypothesis: removing shorts should raise overall WR above 84%,
reduce MDD, and improve avg PnL/trade significantly.

Run:    python backtest_v10.py
Output: backtest_v10_results.csv + backtest_v10_summary.txt
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import os
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 720
TOP_N_PAIRS     = 600
MIN_VOLUME_USDT = 3_000_000

ATR_SL_MULT       = 1.2
ATR_TP1_ONLY      = 0.8
MIN_SCORE_PCT     = 0.43   # long threshold
MIN_SCORE_SHORT   = 0.48   # unused in LONG_ONLY mode, kept for reference
MIN_COOLDOWN_CANDLES = 4   # raised from 3 — fewer signals, more spacing
QUALITY_PREMIUM   = 0.60
REGIME_MODE       = 'HARD'
LONG_ONLY         = True   # NEW: disable shorts entirely
LONG_FILTER       = True
MAX_TRADE_HOURS   = 24

OUTPUT_CSV     = '/mnt/user-data/outputs/backtest_v10_trades.csv'
OUTPUT_SUMMARY = '/mnt/user-data/outputs/backtest_v10_summary.txt'

# ── Anchor indicators — at least one must be present per trade ──
LONG_ANCHORS  = {
    'macd_cross_bull',
    'vol_spike_bull',
    'cmf_buying',
    'obv_accum',
    'bull_div',
    'adx_strong_up',
    'aroon_up',
}
SHORT_ANCHORS = {
    'macd_cross_bear',
    'cmf_selling',
    'adx_strong_down',
    'aroon_down',
    'roc_bear',
}

# ─────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────

def calculate_supertrend(df, period=10, multiplier=3):
    try:
        hl2   = (df['high'] + df['low']) / 2
        atr   = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=period
        ).average_true_range()
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        st    = [0.0] * len(df)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper.iloc[i-1]:
                st[i] = lower.iloc[i]
            elif df['close'].iloc[i] < lower.iloc[i-1]:
                st[i] = upper.iloc[i]
            else:
                st[i] = st[i-1]
        return pd.Series(st, index=df.index)
    except:
        return pd.Series([0.0] * len(df), index=df.index)


def add_indicators(df):
    if len(df) < 55:
        return df
    try:
        df = df.copy()
        df['ema_9']       = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21']      = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50']      = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['supertrend']  = calculate_supertrend(df)
        df['rsi']         = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        srsi = ta.momentum.StochRSIIndicator(df['close'])
        df['stoch_rsi_k'] = srsi.stochrsi_k()
        df['stoch_rsi_d'] = srsi.stochrsi_d()
        macd = ta.trend.MACD(df['close'])
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['roc']         = ta.momentum.ROCIndicator(df['close'], window=12).roc()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_pband']    = bb.bollinger_pband()
        df['atr']         = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()
        df['vol_sma']     = df['volume'].rolling(20).mean()
        df['vol_ratio']   = df['volume'] / df['vol_sma'].replace(0, np.nan)
        df['obv']         = ta.volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()
        df['obv_ema']     = df['obv'].ewm(span=20).mean()
        df['mfi']         = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).money_flow_index()
        df['cmf']         = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).chaikin_money_flow()
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']         = adx.adx()
        df['di_plus']     = adx.adx_pos()
        df['di_minus']    = adx.adx_neg()
        df['cci']         = ta.trend.CCIIndicator(
            df['high'], df['low'], df['close']
        ).cci()
        aroon = ta.trend.AroonIndicator(df['high'], df['low'])
        df['aroon_ind']   = aroon.aroon_up() - aroon.aroon_down()
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap']        = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap'].fillna(df['close'], inplace=True)
        df['bull_engulf'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        ).astype(int)
        # bear_engulf removed from scoring (65.6% WR — below baseline)
        df['bull_div'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['rsi'] > df['rsi'].shift(1))
        ).astype(int)
        df['bear_div'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['rsi'] < df['rsi'].shift(1))
        ).astype(int)
    except Exception as e:
        pass
    return df


def score_candle(r1h, p1h, r4h, r15m, vol_ratio):
    ls = ss = 0
    lr = {}; sr = {}

    mcb = r1h['macd'] > r1h['macd_signal'] and p1h['macd'] <= p1h['macd_signal']
    mcs = r1h['macd'] < r1h['macd_signal'] and p1h['macd'] >= p1h['macd_signal']
    spk = vol_ratio > 2.5

    # TREND 6pts
    if r4h['ema_9'] > r4h['ema_21'] and r4h['ema_21'] > r4h['ema_50']:
        ls += 3; lr['4H_uptrend'] = 3
    elif r4h['ema_9'] < r4h['ema_21'] and r4h['ema_21'] < r4h['ema_50']:
        ss += 3; sr['4H_downtrend'] = 3

    if r1h['ema_9'] > r1h['ema_21']:
        ls += 2; lr['1H_bullish'] = 2
    elif r1h['ema_9'] < r1h['ema_21']:
        ss += 2; sr['1H_bearish'] = 2

    if r1h['close'] > r1h['supertrend']:
        ls += 1; lr['supertrend_bull'] = 1
    elif r1h['close'] < r1h['supertrend']:
        ss += 1; sr['supertrend_bear'] = 1

    # MOMENTUM
    # NOTE: rsi_deep_oversold (70.0% WR) and rsi_overbought (62.5% WR) removed
    # rsi_neutral_bear removed (50.0% WR — random noise)
    rsi = r1h['rsi']
    if rsi < 40:    ls += 2;   lr['rsi_oversold'] = 2
    elif rsi <= 50: ls += 1;   lr['rsi_buy_zone'] = 1
    if rsi > 60:    ss += 2;   sr['rsi_sell_zone'] = 2

    # stoch_rsi_bear removed (69.0% WR) — only keeping bull side
    sk = r1h['stoch_rsi_k']; sd = r1h['stoch_rsi_d']
    if sk < 0.2 and sk > sd:
        ls += 2; lr['stoch_rsi_bull'] = 2

    # MACD — boosted weight (+1) as top performer 94.6%/91.3% WR
    if mcb:   ls += 4; lr['macd_cross_bull'] = 4
    elif mcs: ss += 4; sr['macd_cross_bear'] = 4

    # VOLUME — vol_spike_bull boosted (92.0% WR)
    if spk:
        if r1h['close'] > p1h['close']: ls += 4.5; lr['vol_spike_bull'] = 4.5
        else:                           ss += 3;   sr['vol_spike_bear'] = 3

    # mfi_overbought removed (67.9% WR) — only keeping oversold
    if r1h['mfi'] < 20:
        ls += 1.5; lr['mfi_oversold'] = 1.5

    # CMF — boosted (cmf_selling 95.2%, cmf_buying 91.0%)
    if r1h['cmf'] > 0.15:    ls += 2; lr['cmf_buying'] = 2
    elif r1h['cmf'] < -0.15: ss += 2; sr['cmf_selling'] = 2

    # OBV — boosted (91.9% WR)
    if r1h['obv'] > r1h['obv_ema']: ls += 1; lr['obv_accum'] = 1
    else:                            ss += 1; sr['obv_dist'] = 1

    # VOLATILITY
    # upper_bb removed (52.9% WR — coin flip)
    bbp = r1h['bb_pband']
    if bbp < 0.1: ls += 2.5; lr['lower_bb'] = 2.5

    if r1h['cci'] < -150:  ls += 1.5; lr['cci_oversold'] = 1.5
    elif r1h['cci'] > 150: ss += 1.5; sr['cci_overbought'] = 1.5

    if r1h['close'] > r1h['vwap'] * 1.02:
        ss += 1; sr['above_vwap'] = 1

    # TREND STRENGTH — adx boosted (adx_strong_down 91.2%, adx_strong_up 90.7%)
    adx = r1h['adx']
    if adx > 30:
        if r1h['di_plus'] > r1h['di_minus']: ls += 3; lr['adx_strong_up'] = 3
        else:                                 ss += 3; sr['adx_strong_down'] = 3
    elif adx > 25:
        if r1h['di_plus'] > r1h['di_minus']: ls += 1
        else:                                 ss += 1

    # Aroon — boosted (aroon_down 90.0%, aroon_up 91.6%)
    ai = r1h['aroon_ind']
    if ai > 50:    ls += 2; lr['aroon_up'] = 2
    elif ai < -50: ss += 2; sr['aroon_down'] = 2

    roc = r1h['roc']
    if roc > 3:    ls += 1; lr['roc_bull'] = 1
    elif roc < -3: ss += 1; sr['roc_bear'] = 1

    # PATTERNS
    if r1h['bull_div']:   ls += 2.5; lr['bull_div'] = 2.5
    elif r1h['bear_div']: ss += 2;   sr['bear_div'] = 2

    # bull_engulf kept, bear_engulf removed (65.6% WR)
    if r15m['bull_engulf']:
        ls += 1.5; lr['bull_engulf'] = 1.5

    # HTF
    if r4h['close'] > r4h['vwap']: ls += 1; lr['4h_above_vwap'] = 1
    else:                           ss += 1; sr['4h_below_vwap'] = 1

    # 4h_rsi_bear removed (71.8% WR) — only keeping bull side
    if r4h['rsi'] < 50:
        ls += 1; lr['4h_rsi_bull'] = 1

    return ls, ss, lr, sr, mcb, spk


def simulate_trade(idx, df_1h, direction, entry, sl, tp):
    future = df_1h.iloc[idx+1 : idx+1+MAX_TRADE_HOURS]
    if len(future) == 0:
        return 'TIMEOUT', 0.0, 0

    for i, (_, row) in enumerate(future.iterrows()):
        if direction == 'LONG':
            if row['low'] <= sl:
                pnl = round((sl - entry) / entry * 100, 3)
                # ── ASSERTION: SL on a long must always be negative ──
                assert pnl < 0, (
                    f"BUG: SL PnL should be negative! "
                    f"entry={entry:.6f}, sl={sl:.6f}, pnl={pnl}"
                )
                return 'SL', pnl, i+1
            if row['high'] >= tp:
                pnl = round((tp - entry) / entry * 100, 3)
                return 'TP1', pnl, i+1
        else:  # SHORT
            if row['high'] >= sl:
                pnl = round((entry - sl) / entry * 100, 3)  # entry < sl → negative ✓
                # ── ASSERTION: SL on a short must always be negative ──
                assert pnl < 0, (
                    f"BUG: SL PnL should be negative! "
                    f"entry={entry:.6f}, sl={sl:.6f}, pnl={pnl}"
                )
                return 'SL', pnl, i+1
            if row['low'] <= tp:
                pnl = round((entry - tp) / entry * 100, 3)
                return 'TP1', pnl, i+1

    last = future.iloc[-1]['close']
    pnl  = (last-entry)/entry*100 if direction=='LONG' else (entry-last)/entry*100
    return 'TIMEOUT', round(pnl, 3), MAX_TRADE_HOURS


# ─────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────

class BacktesterV10:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.trades           = []
        self.indicator_stats  = defaultdict(lambda: {'triggered':0,'wins':0,'losses':0})
        self.regime_blocked   = 0
        self.filtered_long    = 0
        self.anchor_rejected  = 0  # NEW — track how many trades anchor filter drops

    async def get_pairs(self):
        await self.exchange.load_markets()
        tickers = await self.exchange.fetch_tickers()
        pairs   = [
            s for s in self.exchange.symbols
            if s.endswith('/USDT:USDT') and 'PERP' not in s
            and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_USDT
        ]
        pairs.sort(key=lambda x: tickers.get(x,{}).get('quoteVolume',0), reverse=True)
        pairs = pairs[:TOP_N_PAIRS]
        print(f"✅ {len(pairs)} pairs loaded")
        return pairs

    async def fetch_df(self, symbol, tf, limit=700):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except:
            return None

    async def load_btc_regime(self):
        print("📡 Loading BTC regime...")
        df = await self.fetch_df('BTC/USDT:USDT', '4h', limit=700)
        if df is None:
            return None
        df['ema21']  = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['regime'] = (df['close'] > df['ema21']).map({True:'BULL', False:'BEAR'})
        bull = (df['regime']=='BULL').sum()
        bear = (df['regime']=='BEAR').sum()
        print(f"  BULL: {bull} candles | BEAR: {bear} candles")
        return df['regime']

    async def backtest_pair(self, symbol, btc_regime_series):
        print(f"  📊 {symbol}...")

        df_1h  = await self.fetch_df(symbol, '1h',  limit=LOOKBACK_DAYS*24+150)
        await asyncio.sleep(0.15)
        df_4h  = await self.fetch_df(symbol, '4h',  limit=LOOKBACK_DAYS*6+60)
        await asyncio.sleep(0.15)
        df_15m = await self.fetch_df(symbol, '15m', limit=LOOKBACK_DAYS*96+250)
        await asyncio.sleep(0.2)

        if df_1h is None or df_4h is None or df_15m is None or len(df_1h) < 100:
            return []

        df_1h  = add_indicators(df_1h)
        df_4h  = add_indicators(df_4h)
        df_15m = add_indicators(df_15m)

        required = [
            'ema_9','ema_21','ema_50','rsi','macd','macd_signal',
            'bb_pband','stoch_rsi_k','stoch_rsi_d','atr','vwap',
            'obv','obv_ema','adx','di_plus','di_minus','cmf','mfi',
            'cci','aroon_ind','roc','supertrend','bull_div','bear_div'
        ]

        pair_trades = []
        last_signal_end = {'LONG': -999, 'SHORT': -999}

        for i in range(55, len(df_1h) - MAX_TRADE_HOURS - 1):
            r1h  = df_1h.iloc[i]
            p1h  = df_1h.iloc[i-1]
            ts1h = df_1h.index[i]

            if any(c not in r1h.index or pd.isna(r1h[c]) for c in required):
                continue

            c4h  = df_4h[df_4h.index   <= ts1h]
            c15m = df_15m[df_15m.index <= ts1h]
            if len(c4h) < 2 or len(c15m) < 2:
                continue

            r4h  = c4h.iloc[-1]
            r15m = c15m.iloc[-1]

            vol_avg   = df_1h['volume'].iloc[max(0,i-20):i].mean()
            vol_ratio = r1h['volume'] / vol_avg if vol_avg > 0 else 1.0

            ls, ss, lr, sr, mcb, spk = score_candle(r1h, p1h, r4h, r15m, vol_ratio)

            # Direction-aware max scores (after v9 removals):
            # upper_bb (-2.5) and rsi_neutral_bear (-1) removed from short side
            # Long max:  ~40.0 | Short max: ~29.5
            max_score_long  = 40.0
            max_score_short = 29.5

            thresh_long  = max_score_long  * MIN_SCORE_PCT    # 0.43
            thresh_short = max_score_short * MIN_SCORE_SHORT   # 0.48 — tighter

            signal = None
            if ls > ss and ls >= thresh_long:
                signal = 'LONG';  score = ls; max_score = max_score_long;  reasons = lr
            elif ss > ls and ss >= thresh_short and not LONG_ONLY:
                signal = 'SHORT'; score = ss; max_score = max_score_short; reasons = sr
            if not signal:
                continue

            # ── Cooldown: minimum 3 candles OR trade duration, whichever longer ──
            if i <= last_signal_end[signal]:
                continue

            # ── ANCHOR CHECK — must have at least one high-WR indicator ──
            reason_keys = set(reasons.keys())
            if signal == 'LONG'  and not (reason_keys & LONG_ANCHORS):
                self.anchor_rejected += 1
                continue
            if signal == 'SHORT' and not (reason_keys & SHORT_ANCHORS):
                self.anchor_rejected += 1
                continue

            entry = r15m['close']
            atr   = r1h['atr']
            if pd.isna(atr) or atr == 0 or pd.isna(entry) or entry == 0:
                continue

            # ── BTC regime ──
            btc_reg    = 'N/A'
            is_counter = False
            if btc_regime_series is not None:
                rc = btc_regime_series[btc_regime_series.index <= ts1h]
                if len(rc) > 0:
                    btc_reg    = rc.iloc[-1]
                    is_counter = (signal=='LONG' and btc_reg=='BEAR') or \
                                 (signal=='SHORT' and btc_reg=='BULL')

            if REGIME_MODE == 'HARD' and is_counter:
                self.regime_blocked += 1
                continue

            # ── Long trend filter ──
            if signal == 'LONG' and LONG_FILTER:
                confirms = [
                    r4h['ema_9'] > r4h['ema_21'],
                    r1h['ema_9'] > r1h['ema_21'],
                    mcb,
                    spk and r1h['close'] > p1h['close'],
                    r1h['rsi'] < 35,
                ]
                if not any(confirms):
                    self.filtered_long += 1
                    continue

            pct     = score / max_score
            quality = 'PREMIUM' if pct >= QUALITY_PREMIUM else 'GOOD'

            if signal == 'LONG':
                sl = entry - atr * ATR_SL_MULT
                tp = entry + atr * ATR_TP1_ONLY
            else:
                sl = entry + atr * ATR_SL_MULT
                tp = entry - atr * ATR_TP1_ONLY

            # ── Sanity guard: sl/tp must make directional sense ──
            if signal == 'LONG'  and sl >= entry: continue
            if signal == 'SHORT' and sl <= entry: continue
            if signal == 'LONG'  and tp <= entry: continue
            if signal == 'SHORT' and tp >= entry: continue

            risk_pct = abs((sl - entry) / entry * 100)
            gain_pct = abs((tp - entry) / entry * 100)
            rr       = abs(tp - entry) / abs(sl - entry)

            outcome, pnl, duration = simulate_trade(i, df_1h, signal, entry, sl, tp)
            win = (outcome == 'TP1')

            trade = {
                'symbol':     symbol.replace('/USDT:USDT',''),
                'timestamp':  str(ts1h),
                'direction':  signal,
                'quality':    quality,
                'score':      round(score, 1),
                'score_pct':  round(pct * 100, 1),
                'btc_regime': btc_reg,
                'entry':      entry,
                'tp':         tp,
                'sl':         sl,
                'risk_pct':   round(risk_pct, 2),
                'gain_pct':   round(gain_pct, 2),
                'rr':         round(rr, 2),
                'outcome':    outcome,
                'win':        win,
                'pnl_pct':    pnl,
                'duration_h': duration,
                'vol_ratio':  round(vol_ratio, 2),
                'reasons':    list(reasons.keys()),
            }
            pair_trades.append(trade)

            # Enforce minimum cooldown regardless of how fast trade resolved
            last_signal_end[signal] = i + max(duration + 1, MIN_COOLDOWN_CANDLES)

            for name in reasons:
                self.indicator_stats[name]['triggered'] += 1
                if win: self.indicator_stats[name]['wins'] += 1
                else:   self.indicator_stats[name]['losses'] += 1

        print(f"     → {len(pair_trades)} signals")
        return pair_trades

    # ─────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────

    def print_and_save(self):
        if not self.trades:
            print("❌ No trades found.")
            return

        df = pd.DataFrame(self.trades)

        total  = len(df)
        wins   = df['win'].sum()
        losses = total - wins
        wr     = wins / total * 100
        apnl   = df['pnl_pct'].mean()
        aw     = df[df['win']]['pnl_pct'].mean()     if wins   > 0 else 0

        # ── FIXED: split losses into SL vs TIMEOUT ──
        sl_trades      = df[df['outcome']=='SL']
        timeout_trades = df[df['outcome']=='TIMEOUT']
        al_sl          = sl_trades['pnl_pct'].mean()      if len(sl_trades)      > 0 else 0
        al_timeout     = timeout_trades['pnl_pct'].mean() if len(timeout_trades) > 0 else 0
        al             = df[~df['win']]['pnl_pct'].mean() if losses > 0 else 0

        pf     = abs(aw*wins/(al_sl*len(sl_trades))) if len(sl_trades) > 0 and al_sl != 0 else 99
        tp1r   = df['win'].mean() * 100
        slr    = (df['outcome']=='SL').mean() * 100
        tor    = (df['outcome']=='TIMEOUT').mean() * 100
        cumul  = (1 + df['pnl_pct']/100).cumprod()
        mdd    = ((cumul - cumul.cummax()) / cumul.cummax() * 100).min()
        spd    = total / LOOKBACK_DAYS
        spm    = total / (LOOKBACK_DAYS / 30)
        mr     = apnl * spm

        longs  = df[df['direction']=='LONG']
        shorts = df[df['direction']=='SHORT']
        prem   = df[df['quality']=='PREMIUM']
        good   = df[df['quality']=='GOOD']

        print("\n" + "╔"+"═"*54+"╗")
        print("║" + "  📊 BACKTEST v10 — FINAL RESULTS".center(54) + "║")
        print("╚"+"═"*54+"╝")
        print(f"\n  Settings: score≥{MIN_SCORE_PCT*100:.0f}% | {REGIME_MODE} regime | TP1={ATR_TP1_ONLY}x | SL={ATR_SL_MULT}x")
        print(f"  Pairs: {df['symbol'].nunique()} | Lookback: {LOOKBACK_DAYS}d\n")

        print(f"  {'Signals':20s}: {total}  ({spd:.1f}/day  |  {spm:.0f}/month)")
        print(f"  {'Win Rate':20s}: {wr:.1f}%")
        print(f"  {'Profit Factor':20s}: {pf:.2f}  (uses SL losses only)")
        print(f"  {'Avg PnL/trade':20s}: {apnl:+.3f}%")
        print(f"  {'Avg Win':20s}: {aw:+.3f}%")
        print(f"  {'Avg SL Loss':20s}: {al_sl:+.3f}%  ← should be negative")
        print(f"  {'Avg Timeout PnL':20s}: {al_timeout:+.3f}%")
        print(f"  {'Avg All Losses':20s}: {al:+.3f}%")
        print(f"  {'Monthly est.':20s}: {mr:+.1f}%  (signals × avg PnL)")
        print(f"  {'Max Drawdown':20s}: {mdd:.2f}%")
        print(f"  {'TP1 Rate':20s}: {tp1r:.1f}%")
        print(f"  {'SL Rate':20s}: {slr:.1f}%")
        print(f"  {'Timeout Rate':20s}: {tor:.1f}%")
        print(f"  {'Avg Duration':20s}: {df['duration_h'].mean():.1f}h")
        print(f"  {'Regime blocked':20s}: {self.regime_blocked}")
        print(f"  {'Long filtered':20s}: {self.filtered_long}")
        print(f"  {'Anchor rejected':20s}: {self.anchor_rejected}  ← new filter")

        print(f"\n  ── By Direction ──")
        for label, sub in [('LONG', longs), ('SHORT', shorts)]:
            if len(sub) == 0: continue
            print(f"  {label:6s} | n={len(sub):5d} ({len(sub)/LOOKBACK_DAYS:.1f}/day) | "
                  f"WR={sub['win'].mean()*100:.1f}% | Avg={sub['pnl_pct'].mean():+.3f}%")

        print(f"\n  ── By Quality ──")
        for label, sub in [('PREMIUM', prem), ('GOOD', good)]:
            if len(sub) == 0: continue
            print(f"  {label:8s} | n={len(sub):5d} | "
                  f"WR={sub['win'].mean()*100:.1f}% | Avg={sub['pnl_pct'].mean():+.3f}%")

        print(f"\n  ── By BTC Regime ──")
        for reg in ['BULL','BEAR']:
            sub = df[df['btc_regime']==reg]
            if len(sub) == 0: continue
            print(f"  {reg:5s} | n={len(sub):5d} | "
                  f"WR={sub['win'].mean()*100:.1f}% | Avg={sub['pnl_pct'].mean():+.3f}%")

        print(f"\n  ── Score Band Breakdown ──")
        print(f"  {'Band':10s} {'n':>6} {'WR%':>7} {'Avg%':>8} {'SL%':>7}")
        for lo, hi in [(40,45),(45,50),(50,55),(55,60),(60,65),(65,100)]:
            sub = df[(df['score_pct']>=lo) & (df['score_pct']<hi)]
            if len(sub) < 3: continue
            print(f"  {lo}-{hi}%    {len(sub):>6d} "
                  f"{sub['win'].mean()*100:>6.1f}% "
                  f"{sub['pnl_pct'].mean():>+7.3f}% "
                  f"{(sub['outcome']=='SL').mean()*100:>6.1f}%")

        # ── Indicators ──
        ind_rows = []
        for name, s in self.indicator_stats.items():
            t = s['triggered']
            if t < 10: continue
            ind_rows.append({
                'indicator': name,
                'triggered': t,
                'win_rate':  round(s['wins']/t*100, 1),
                'losses':    s['losses'],
            })
        ind_df = pd.DataFrame(ind_rows).sort_values('win_rate', ascending=False)

        print(f"\n  ── Indicator Win Rates ──")
        print("  TOP 15:")
        for _, r in ind_df.head(15).iterrows():
            bar = '█' * int(r['win_rate']/5)
            print(f"  {r['indicator']:28s} {r['win_rate']:5.1f}%  n={r['triggered']:5d}  {bar}")
        if len(ind_df) > 5:
            print("  BOTTOM (watch these):")
            for _, r in ind_df.tail(5).iterrows():
                print(f"  {r['indicator']:28s} {r['win_rate']:5.1f}%  n={r['triggered']:5d}")

        # ── Top symbols ──
        sym_df = df.groupby('symbol').agg(
            signals  = ('win','count'),
            win_rate = ('win', lambda x: round(x.mean()*100,1)),
            avg_pnl  = ('pnl_pct', lambda x: round(x.mean(),3)),
            avg_dur  = ('duration_h', lambda x: round(x.mean(),1)),
        ).sort_values('win_rate', ascending=False)

        print(f"\n  ── Top 20 Symbols ──")
        print(sym_df[sym_df['signals']>=3].head(20).to_string())

        print(f"{'╔'+'═'*54+'╗'}")
        print("║" + "  ✅ DEPLOY CHECKLIST".center(54) + "║")
        print(f"{'╚'+'═'*54+'╝'}")
        print(f"  TRADE_MODE       = 'TP1_ONLY'")
        print(f"  REGIME_MODE      = '{REGIME_MODE}'")
        print(f"  MIN_SCORE_PCT    = {MIN_SCORE_PCT}  (longs)")
        print(f"  MIN_SCORE_SHORT  = {MIN_SCORE_SHORT}  (shorts)")
        print(f"  ATR_TP1_ONLY     = {ATR_TP1_ONLY}")
        print(f"  ATR_SL_MULT      = {ATR_SL_MULT}")
        print(f"  LONG_FILTER      = {LONG_FILTER}")
        print(f"  ANCHOR_FILTER    = True")
        print(f"  MIN_COOLDOWN     = {MIN_COOLDOWN_CANDLES} candles")
        print(f"\n  Expected live: {spd:.1f}/day | {wr:.1f}% WR | {apnl:+.2f}%/trade")
        print(f"  Monitor via /stats after 2 weeks")
        print(f"  If live WR < {wr-8:.0f}% → raise MIN_SCORE_PCT to 0.45 / MIN_SCORE_SHORT to 0.50")

        self._save_csv(df, ind_df, sym_df, spd, spm, wr, pf, apnl, aw,
                       al, al_sl, al_timeout, mr, mdd, slr, tor)

    def _save_csv(self, df, ind_df, sym_df, spd, spm, wr, pf, apnl, aw,
                  al, al_sl, al_timeout, mr, mdd, slr, tor):
        import os, csv
        os.makedirs('/mnt/user-data/outputs', exist_ok=True)

        # ── All trades CSV ──
        df_export = df.drop(columns=['reasons'], errors='ignore')
        df_export.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Trades saved → {OUTPUT_CSV}")

        # ── Summary text ──
        longs  = df[df['direction']=='LONG']
        prem   = df[df['quality']=='PREMIUM']
        good   = df[df['quality']=='GOOD']

        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║         BACKTEST v10 — SUMMARY                      ║",
            "╚══════════════════════════════════════════════════════╝",
            f"  LONG_ONLY        = {LONG_ONLY}",
            f"  MIN_SCORE_PCT    = {MIN_SCORE_PCT}",
            f"  ATR_TP1_ONLY     = {ATR_TP1_ONLY}",
            f"  ATR_SL_MULT      = {ATR_SL_MULT}",
            f"  MIN_COOLDOWN     = {MIN_COOLDOWN_CANDLES}",
            f"  REGIME_MODE      = {REGIME_MODE}",
            "",
            f"  Pairs tested     : {df['symbol'].nunique()}",
            f"  Signals          : {len(df)}  ({spd:.1f}/day | {spm:.0f}/month)",
            f"  Win Rate         : {wr:.1f}%",
            f"  Profit Factor    : {pf:.2f}",
            f"  Avg PnL/trade    : {apnl:+.3f}%",
            f"  Avg Win          : {aw:+.3f}%",
            f"  Avg SL Loss      : {al_sl:+.3f}%",
            f"  Avg Timeout PnL  : {al_timeout:+.3f}%",
            f"  Monthly est.     : {mr:+.1f}%",
            f"  Max Drawdown     : {mdd:.2f}%",
            f"  TP1 Rate         : {wr:.1f}%",
            f"  SL Rate          : {slr:.1f}%",
            f"  Avg Duration     : {df['duration_h'].mean():.1f}h",
            f"  Regime blocked   : {self.regime_blocked}",
            f"  Anchor rejected  : {self.anchor_rejected}",
            "",
            "  ── By Quality ──",
            f"  PREMIUM  n={len(prem):4d} | WR={prem['win'].mean()*100:.1f}% | Avg={prem['pnl_pct'].mean():+.3f}%" if len(prem) > 0 else "  PREMIUM  n=0",
            f"  GOOD     n={len(good):4d} | WR={good['win'].mean()*100:.1f}% | Avg={good['pnl_pct'].mean():+.3f}%" if len(good) > 0 else "  GOOD     n=0",
            "",
            "  ── Score Band Breakdown ──",
            f"  {'Band':10s} {'n':>6} {'WR%':>7} {'Avg%':>8} {'SL%':>7}",
        ]
        for lo, hi in [(40,45),(45,50),(50,55),(55,60),(60,65),(65,100)]:
            sub = df[(df['score_pct']>=lo) & (df['score_pct']<hi)]
            if len(sub) < 3: continue
            lines.append(
                f"  {lo}-{hi}%    {len(sub):>6d} "
                f"{sub['win'].mean()*100:>6.1f}% "
                f"{sub['pnl_pct'].mean():>+7.3f}% "
                f"{(sub['outcome']=='SL').mean()*100:>6.1f}%"
            )
        lines += ["", "  ── Top Indicators ──"]
        for _, r in ind_df.head(15).iterrows():
            bar = '█' * int(r['win_rate']/5)
            lines.append(f"  {r['indicator']:28s} {r['win_rate']:5.1f}%  n={r['triggered']:5d}  {bar}")
        lines += ["", "  ── Bottom Indicators ──"]
        for _, r in ind_df.tail(5).iterrows():
            lines.append(f"  {r['indicator']:28s} {r['win_rate']:5.1f}%  n={r['triggered']:5d}")
        lines += ["", "  ── Top 20 Symbols ──"]
        lines.append(sym_df[sym_df['signals']>=3].head(20).to_string())

        summary_text = "\n".join(lines)
        with open(OUTPUT_SUMMARY, 'w') as f:
            f.write(summary_text)
        print(f"✅ Summary saved → {OUTPUT_SUMMARY}")
        print(summary_text)

    async def run(self):
        print(f"""
╔══════════════════════════════════════════════════════╗
║         BACKTEST v10.0 — LONGS ONLY EXPERIMENT     ║
║  {LOOKBACK_DAYS}d | {TOP_N_PAIRS} pairs | L≥{MIN_SCORE_PCT*100:.0f}% ONLY | {REGIME_MODE} | TP={ATR_TP1_ONLY}x SL={ATR_SL_MULT}x | cd≥{MIN_COOLDOWN_CANDLES}
╚══════════════════════════════════════════════════════╝

  Changes vs v9:
  • LONG_ONLY = True — shorts fully disabled
  • MIN_COOLDOWN_CANDLES raised 3 → 4
  • Output: CSV (no Excel dependency)
""")
        pairs      = await self.get_pairs()
        btc_regime = await self.load_btc_regime()
        await asyncio.sleep(0.5)

        for i, pair in enumerate(pairs):
            try:
                trades = await self.backtest_pair(pair, btc_regime)
                self.trades.extend(trades)
            except Exception as e:
                print(f"  [ERROR] {pair}: {e}")
            if (i+1) % 50 == 0:
                print(f"  ── {i+1}/{len(pairs)} pairs done | {len(self.trades)} signals so far ──")
            await asyncio.sleep(0.6)

        await self.exchange.close()
        print(f"\n✅ Done — {len(self.trades)} signals from {len(pairs)} pairs\n")
        self.print_and_save()


async def main():
    bt = BacktesterV10()
    await bt.run()

if __name__ == '__main__':
    asyncio.run(main())
