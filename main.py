"""
BACKTEST v7.0 — FINAL SETTINGS VALIDATION
==========================================
Runs the exact settings from advanced_bot_v6.py:

  MIN_SCORE_PCT = 0.43
  REGIME_MODE   = HARD
  TRADE_MODE    = TP1-ONLY (0.6x ATR)
  SL            = 1.5x ATR
  LONG_FILTER   = True
  PAIRS         = Top 600 (min $500k volume)

No sweeping — just validates the final locked config.

Run:    python backtest_v7.py
Output: backtest_v7_results.xlsx
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
# SETTINGS — exact match to advanced_bot_v6.py
# ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS   = 90
TOP_N_PAIRS     = 600
MIN_VOLUME_USDT = 3_000_000

ATR_SL_MULT       = 1.5
ATR_TP1_ONLY      = 0.6
MIN_SCORE_PCT     = 0.43
QUALITY_PREMIUM   = 0.60
REGIME_MODE       = 'HARD'    # HARD / SOFT / OFF
LONG_FILTER       = True
MAX_TRADE_HOURS   = 24

OUTPUT_FILE = '/mnt/user-data/outputs/backtest_v7_results.xlsx'

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
        df['bear_engulf'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        ).astype(int)
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
    rsi = r1h['rsi']
    if rsi < 30:    ls += 3.5; lr['rsi_deep_oversold'] = 3.5
    elif rsi < 40:  ls += 2;   lr['rsi_oversold'] = 2
    elif rsi <= 50: ls += 1;   lr['rsi_buy_zone'] = 1
    if rsi > 70:    ss += 3.5; sr['rsi_deep_overbought'] = 3.5
    elif rsi > 60:  ss += 2;   sr['rsi_overbought'] = 2
    elif rsi >= 50: ss += 1;   sr['rsi_sell_zone'] = 1

    sk = r1h['stoch_rsi_k']; sd = r1h['stoch_rsi_d']
    if sk < 0.2 and sk > sd:   ls += 2; lr['stoch_rsi_bull'] = 2
    elif sk > 0.8 and sk < sd: ss += 2; sr['stoch_rsi_bear'] = 2

    if mcb:   ls += 3; lr['macd_cross_bull'] = 3
    elif mcs: ss += 3; sr['macd_cross_bear'] = 3

    # VOLUME
    if spk:
        if r1h['close'] > p1h['close']: ls += 3.5; lr['vol_spike_bull'] = 3.5
        else:                           ss += 3;   sr['vol_spike_bear'] = 3

    if r1h['mfi'] < 20:   ls += 1.5; lr['mfi_oversold'] = 1.5
    elif r1h['mfi'] > 80: ss += 1.5; sr['mfi_overbought'] = 1.5

    if r1h['cmf'] > 0.15:    ls += 1; lr['cmf_buying'] = 1
    elif r1h['cmf'] < -0.15: ss += 1; sr['cmf_selling'] = 1

    if r1h['obv'] > r1h['obv_ema']: ls += 0.5; lr['obv_accum'] = 0.5
    else:                            ss += 0.5; sr['obv_dist'] = 0.5

    # VOLATILITY
    bbp = r1h['bb_pband']
    if bbp < 0.1:   ls += 2.5; lr['lower_bb'] = 2.5
    elif bbp > 0.9: ss += 2.5; sr['upper_bb'] = 2.5

    if r1h['cci'] < -150:  ls += 1.5; lr['cci_oversold'] = 1.5
    elif r1h['cci'] > 150: ss += 1.5; sr['cci_overbought'] = 1.5

    if r1h['close'] > r1h['vwap'] * 1.02:
        ss += 1; sr['above_vwap'] = 1

    # TREND STRENGTH
    adx = r1h['adx']
    if adx > 30:
        if r1h['di_plus'] > r1h['di_minus']: ls += 2; lr['adx_strong_up'] = 2
        else:                                 ss += 2; sr['adx_strong_down'] = 2
    elif adx > 25:
        if r1h['di_plus'] > r1h['di_minus']: ls += 1
        else:                                 ss += 1

    ai = r1h['aroon_ind']
    if ai > 50:    ls += 1; lr['aroon_up'] = 1
    elif ai < -50: ss += 1; sr['aroon_down'] = 1

    roc = r1h['roc']
    if roc > 3:    ls += 1; lr['roc_bull'] = 1
    elif roc < -3: ss += 1; sr['roc_bear'] = 1

    # PATTERNS
    if r1h['bull_div']:   ls += 2.5; lr['bull_div'] = 2.5
    elif r1h['bear_div']: ss += 2;   sr['bear_div'] = 2

    if r15m['bull_engulf']:   ls += 1.5; lr['bull_engulf'] = 1.5
    elif r15m['bear_engulf']: ss += 1.5; sr['bear_engulf'] = 1.5

    # HTF
    if r4h['close'] > r4h['vwap']: ls += 1; lr['4h_above_vwap'] = 1
    else:                           ss += 1; sr['4h_below_vwap'] = 1

    if r4h['rsi'] < 50:   ls += 1; lr['4h_rsi_bull'] = 1
    elif r4h['rsi'] > 50: ss += 1; sr['4h_rsi_bear'] = 1

    return ls, ss, lr, sr, mcb, spk


def simulate_trade(idx, df_1h, direction, entry, sl, tp):
    future = df_1h.iloc[idx+1 : idx+1+MAX_TRADE_HOURS]
    if len(future) == 0:
        return 'TIMEOUT', 0.0, 0

    for i, (_, row) in enumerate(future.iterrows()):
        if direction == 'LONG':
            if row['low'] <= sl:
                return 'SL',  round((sl - entry) / entry * 100, 3), i+1
            if row['high'] >= tp:
                return 'TP1', round((tp - entry) / entry * 100, 3), i+1
        else:
            if row['high'] >= sl:
                return 'SL',  round((entry - sl) / entry * 100 * -1, 3), i+1
            if row['low'] <= tp:
                return 'TP1', round((entry - tp) / entry * 100, 3), i+1

    last = future.iloc[-1]['close']
    pnl  = (last-entry)/entry*100 if direction=='LONG' else (entry-last)/entry*100
    return 'TIMEOUT', round(pnl, 3), MAX_TRADE_HOURS


# ─────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────

class BacktesterV7:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.trades          = []
        self.indicator_stats = defaultdict(lambda: {'triggered':0,'wins':0,'losses':0})
        self.regime_blocked  = 0
        self.filtered_long   = 0

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
        # Cooldown: track when each direction last fired on this pair
        # Don't allow same direction again until previous trade resolves
        last_signal_end = {'LONG': -999, 'SHORT': -999}  # stores candle index

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
            max_score = 35
            thresh    = max_score * MIN_SCORE_PCT

            signal = None
            if ls > ss and ls >= thresh:
                signal = 'LONG';  score = ls; reasons = lr
            elif ss > ls and ss >= thresh:
                signal = 'SHORT'; score = ss; reasons = sr
            if not signal:
                continue

            # ── Cooldown: skip if previous same-direction trade not resolved ──
            if i <= last_signal_end[signal]:
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

            # ── HARD regime block ──
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

            # Update cooldown — block this direction for trade duration
            last_signal_end[signal] = i + duration + 1

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
        aw     = df[df['win']]['pnl_pct'].mean()    if wins   > 0 else 0
        al     = df[~df['win']]['pnl_pct'].mean()   if losses > 0 else 0
        pf     = abs(aw*wins/(al*losses))            if losses > 0 and al != 0 else 99
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

        # ── Console ──────────────────────────────────────────
        print("\n" + "╔"+"═"*54+"╗")
        print("║" + "  📊 BACKTEST v7 — FINAL RESULTS".center(54) + "║")
        print("╚"+"═"*54+"╝")
        print(f"\n  Settings: score≥{MIN_SCORE_PCT*100:.0f}% | {REGIME_MODE} regime | TP1={ATR_TP1_ONLY}x | SL={ATR_SL_MULT}x")
        print(f"  Pairs: {df['symbol'].nunique()} | Lookback: {LOOKBACK_DAYS}d\n")

        print(f"  {'Signals':20s}: {total}  ({spd:.1f}/day  |  {spm:.0f}/month)")
        print(f"  {'Win Rate':20s}: {wr:.1f}%")
        print(f"  {'Profit Factor':20s}: {pf:.2f}")
        print(f"  {'Avg PnL/trade':20s}: {apnl:+.3f}%")
        print(f"  {'Avg Win':20s}: {aw:+.3f}%")
        print(f"  {'Avg Loss':20s}: {al:+.3f}%")
        print(f"  {'Monthly est.':20s}: {mr:+.1f}%  (signals × avg PnL)")
        print(f"  {'Max Drawdown':20s}: {mdd:.2f}%")
        print(f"  {'TP1 Rate':20s}: {tp1r:.1f}%")
        print(f"  {'SL Rate':20s}: {slr:.1f}%")
        print(f"  {'Timeout Rate':20s}: {tor:.1f}%")
        print(f"  {'Avg Duration':20s}: {df['duration_h'].mean():.1f}h")
        print(f"  {'Regime blocked':20s}: {self.regime_blocked}")
        print(f"  {'Long filtered':20s}: {self.filtered_long}")

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
            print("  BOTTOM 8 (consider removing):")
            for _, r in ind_df.tail(8).iterrows():
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

        # ── Live checklist ──
        print(f"\n{'╔'+'═'*54+'╗'}")
        print("║" + "  ✅ DEPLOY CHECKLIST".center(54) + "║")
        print(f"{'╚'+'═'*54+'╝'}")
        print(f"  TRADE_MODE    = 'TP1_ONLY'")
        print(f"  REGIME_MODE   = '{REGIME_MODE}'")
        print(f"  MIN_SCORE_PCT = {MIN_SCORE_PCT}")
        print(f"  ATR_TP1_ONLY  = {ATR_TP1_ONLY}")
        print(f"  ATR_SL_MULT   = {ATR_SL_MULT}")
        print(f"  LONG_FILTER   = {LONG_FILTER}")
        print(f"\n  Expected live: {spd:.1f}/day | {wr:.1f}% WR | {apnl:+.2f}%/trade")
        print(f"  Monitor via /stats after 2 weeks")
        print(f"  If live WR < {wr-8:.0f}% → raise MIN_SCORE_PCT to 0.45")

        # ── Save Excel ──
        self._save_excel(df, ind_df, sym_df, spd, spm, wr, pf, apnl, aw, al, mr, mdd, slr, tor)

    def _save_excel(self, df, ind_df, sym_df, spd, spm, wr, pf, apnl, aw, al, mr, mdd, slr, tor):
        print(f"\n💾 Saving {OUTPUT_FILE}...")
        os.makedirs('/mnt/user-data/outputs', exist_ok=True)

        longs  = df[df['direction']=='LONG']
        shorts = df[df['direction']=='SHORT']
        prem   = df[df['quality']=='PREMIUM']
        good   = df[df['quality']=='GOOD']

        # Summary
        summary = pd.DataFrame({'Metric': [
            '─── PERFORMANCE ───',
            'Pairs Tested', 'Total Signals', 'Signals/Day', 'Signals/Month',
            'Win Rate %', 'Profit Factor', 'Avg PnL %', 'Avg Win %', 'Avg Loss %',
            'Monthly Return Est %', 'Max Drawdown %',
            'TP1 Rate %', 'SL Rate %', 'Timeout Rate %', 'Avg Duration (h)',
            '─── BY DIRECTION ───',
            'Long Signals', 'Long WR %', 'Long Avg PnL %',
            'Short Signals', 'Short WR %', 'Short Avg PnL %',
            '─── BY QUALITY ───',
            'PREMIUM Signals', 'PREMIUM WR %',
            'GOOD Signals', 'GOOD WR %',
            '─── FILTERS ───',
            'Regime Blocked', 'Long Filtered',
            '─── SETTINGS ───',
            'MIN_SCORE_PCT', 'REGIME_MODE', 'ATR_TP1', 'ATR_SL',
            'LONG_FILTER', 'Pairs Universe', 'Lookback Days',
        ], 'Value': [
            '',
            df['symbol'].nunique(), len(df), round(spd,1), round(spm,0),
            round(wr,1), round(pf,2), round(apnl,3), round(aw,3), round(al,3),
            round(mr,1), round(mdd,2),
            round(wr,1), round(slr,1), round(tor,1), round(df['duration_h'].mean(),1),
            '',
            len(longs), round(longs['win'].mean()*100,1) if len(longs)>0 else 0,
            round(longs['pnl_pct'].mean(),3) if len(longs)>0 else 0,
            len(shorts), round(shorts['win'].mean()*100,1) if len(shorts)>0 else 0,
            round(shorts['pnl_pct'].mean(),3) if len(shorts)>0 else 0,
            '',
            len(prem), round(prem['win'].mean()*100,1) if len(prem)>0 else 0,
            len(good), round(good['win'].mean()*100,1) if len(good)>0 else 0,
            '',
            self.regime_blocked, self.filtered_long,
            '',
            MIN_SCORE_PCT, REGIME_MODE, ATR_TP1_ONLY, ATR_SL_MULT,
            LONG_FILTER, TOP_N_PAIRS, LOOKBACK_DAYS,
        ]})

        # Score bands
        band_rows = []
        for lo, hi in [(40,45),(45,50),(50,55),(55,60),(60,65),(65,100)]:
            sub = df[(df['score_pct']>=lo) & (df['score_pct']<hi)]
            if len(sub) < 3: continue
            band_rows.append({
                'Band': f'{lo}-{hi}%',
                'Signals': len(sub),
                'Win Rate %': round(sub['win'].mean()*100,1),
                'Avg PnL %': round(sub['pnl_pct'].mean(),3),
                'SL Rate %': round((sub['outcome']=='SL').mean()*100,1),
                'Avg Duration': round(sub['duration_h'].mean(),1),
            })

        # Equity curve
        df_eq = df.sort_values('timestamp').copy().reset_index(drop=True)
        df_eq['trade_num']       = range(1, len(df_eq)+1)
        df_eq['cumulative_pnl']  = (1 + df_eq['pnl_pct']/100).cumprod()
        df_eq['running_wr']      = df_eq['win'].expanding().mean() * 100
        df_eq['running_avg_pnl'] = df_eq['pnl_pct'].expanding().mean()

        # Monthly breakdown
        df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M').astype(str)
        monthly = df.groupby('month').agg(
            signals  = ('win','count'),
            win_rate = ('win', lambda x: round(x.mean()*100,1)),
            avg_pnl  = ('pnl_pct', lambda x: round(x.mean(),3)),
            total_pnl= ('pnl_pct', lambda x: round(x.sum(),2)),
            sl_count = ('outcome', lambda x: (x=='SL').sum()),
        ).reset_index()

        # All trades export
        df_export = df.drop(columns=['reasons'], errors='ignore')

        with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
            summary.to_excel(writer, sheet_name='📊 Summary', index=False)
            pd.DataFrame(band_rows).to_excel(writer, sheet_name='📈 Score Bands', index=False)
            monthly.to_excel(writer, sheet_name='📅 Monthly', index=False)
            sym_df.reset_index().to_excel(writer, sheet_name='🏆 By Symbol', index=False)
            ind_df.to_excel(writer, sheet_name='🔬 Indicators', index=False)
            df_eq[['trade_num','symbol','direction','quality','score_pct',
                   'btc_regime','outcome','pnl_pct','duration_h',
                   'cumulative_pnl','running_wr','running_avg_pnl']
            ].to_excel(writer, sheet_name='📉 Equity Curve', index=False)
            df_export.to_excel(writer, sheet_name='📋 All Trades', index=False)

            for sheet in writer.sheets.values():
                sheet.set_column('A:Z', 20)
                sheet.freeze_panes(1, 0)

        print(f"✅ Saved | 7 sheets:")
        print(f"   📊 Summary | 📈 Score Bands | 📅 Monthly")
        print(f"   🏆 By Symbol | 🔬 Indicators | 📉 Equity Curve | 📋 All Trades")

    async def run(self):
        print(f"""
╔══════════════════════════════════════════════════════╗
║           BACKTEST v7.0 — FINAL VALIDATION          ║
║  {LOOKBACK_DAYS}d | {TOP_N_PAIRS} pairs | score≥{MIN_SCORE_PCT*100:.0f}% | {REGIME_MODE} | TP1={ATR_TP1_ONLY}x | SL={ATR_SL_MULT}x
╚══════════════════════════════════════════════════════╝
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
    bt = BacktesterV7()
    await bt.run()

if __name__ == '__main__':
    asyncio.run(main())
