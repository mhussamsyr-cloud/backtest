"""
╔══════════════════════════════════════════════════════════════╗
║     SMC PRO v4 — BACKTEST v3.0                              ║
║                                                              ║
║  Changes from v2 (based on backtest data):                 ║
║    - SHORT ONLY: LONGs killed (35% WR → net loser)         ║
║    - MIN_SCORE: 75 → 83  (75-82 band = 43% WR, useless)   ║
║    - BEAR STRUCTURE REQUIRED: BOS_BULL/MSS_BULL blocked    ║
║    - SL tightened: ATR×0.4 instead of ATR×0.6 min         ║
║      → smaller risk = fatter RR on same TP distances       ║
║    - TP1 RR: 1.5 → 2.0  (earn more on first partial)      ║
║    - TP2 RR: 2.5 → 3.5                                     ║
║    - TP3 RR: 4.0 → 5.5  (ride the real moves)             ║
║    - TIMEOUT: 48H → 72H (give trades room to develop)      ║
║                                                              ║
║  Target: WR>55%, PF>2.0, MaxDD<25%                        ║
║  Output: backtest_smc_v3_results.xlsx                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ta
import ccxt.async_support as ccxt

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('SMCBacktest')

# ── SETTINGS (match SMC Pro v4 exactly) ─────────────────────────────────────

LOOKBACK_DAYS        = 180      # 6 months of data
TOP_N_PAIRS          = 600      # top 100 by volume (SMC is slower per pair, 3 TFs)
MIN_VOLUME_24H       = 1_000_000

# SMC constants
MIN_SCORE            = 85      # RAISED: 83-85 = 46% WR in v2, still bad
OB_TOLERANCE_PCT     = 0.008
OB_IMPULSE_ATR_MULT  = 1.0
STRUCTURE_LOOKBACK   = 20
HH_LL_LOOKBACK       = 10
HH_LL_BONUS          = 8

# v2 filters (kept)
SHORT_ONLY           = True    # LONGs = 35% WR in v1, net negative
REQUIRE_BEAR_STRUCTURE = True  # BOS_BEAR or MSS_BEAR only

# v3 NEW filters
REQUIRE_BOS_ONLY     = True    # BOS_BEAR only — MSS_BEAR = 0% WR in v2
BLOCK_SWEEP          = True    # Sweep=YES = 40% WR vs 62.9% without — kill it
REQUIRE_TRENDING     = True    # HH/LL required — ranging = 33% WR in v2

# Trade management
TP_RR                = [2.0, 3.5, 5.5]     # unchanged from v2
TP_PCT               = [0.50, 0.30, 0.20]  # unchanged
TIMEOUT_HOURS        = 48      # back to 48H — 72H didn't help, TP3 hit 40% fine

# Equity sim
RISK_PER_TRADE       = 0.02
MAX_CONCURRENT       = 5

OUTPUT_FILE = '/mnt/user-data/outputs/backtest_smc_v3_results.xlsx'


# ── INDICATORS (identical to live bot) ──────────────────────────────────────

def add_indicators(df):
    if len(df) < 55:
        return df
    try:
        df['ema_21']  = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_50']  = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], min(200, len(df)-1)).ema_indicator()
        df['rsi']     = ta.momentum.RSIIndicator(df['close'], 14).rsi()

        macd = ta.trend.MACD(df['close'])
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist']   = macd.macd_diff()

        stoch = ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k'] = stoch.stochrsi_k()
        df['srsi_d'] = stoch.stochrsi_d()

        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()

        adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']    = adx_i.adx()
        df['di_pos'] = adx_i.adx_pos()
        df['di_neg'] = adx_i.adx_neg()

        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).money_flow_index()

        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open','close']].max(axis=1)
        lw   = df[['open','close']].min(axis=1) - df['low']

        df['bull_engulf'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)

        df['bear_engulf'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)

        df['bull_pin'] = (
            (lw > body * 2.5) & (lw > uw * 2) & (df['close'] > df['open'])
        ).astype(int)

        df['bear_pin'] = (
            (uw > body * 2.5) & (uw > lw * 2) & (df['close'] < df['open'])
        ).astype(int)

        df['hammer'] = (
            (lw > body * 2.0) & (lw > uw * 1.5)
        ).astype(int)

        df['shooting_star'] = (
            (uw > body * 2.0) & (uw > lw * 1.5)
        ).astype(int)

    except Exception as e:
        pass
    return df


# ── SMC ENGINE (identical logic to live bot) ─────────────────────────────────

def swing_highs_lows(df, left=4, right=4):
    highs, lows = [], []
    n = len(df)
    for i in range(left, n - right):
        hi = df['high'].iloc[i]
        lo = df['low'].iloc[i]
        if all(hi >= df['high'].iloc[i-left:i]) and all(hi >= df['high'].iloc[i+1:i+right+1]):
            highs.append({'i': i, 'price': hi})
        if all(lo <= df['low'].iloc[i-left:i]) and all(lo <= df['low'].iloc[i+1:i+right+1]):
            lows.append({'i': i, 'price': lo})
    return highs, lows


def check_4h_hh_ll(df_4h, direction, lookback=HH_LL_LOOKBACK):
    n = len(df_4h)
    if n < lookback * 2:
        return False
    recent = df_4h.iloc[-lookback:]
    prior  = df_4h.iloc[-(lookback * 2):-lookback]
    if direction == 'LONG':
        return recent['high'].max() > prior['high'].max()
    else:
        return recent['low'].min() < prior['low'].min()


def detect_structure_break(df, highs, lows, lookback=STRUCTURE_LOOKBACK):
    events = []
    close = df['close']
    n = len(df)
    start = max(0, n - lookback - 15)

    for k in range(1, len(highs)):
        ph = highs[k-1]; ch = highs[k]
        if ch['i'] < start: continue
        level = ph['price']
        for j in range(ch['i'], min(ch['i'] + 10, n)):
            if close.iloc[j] > level:
                kind = 'BOS_BULL' if ch['price'] > ph['price'] else 'MSS_BULL'
                events.append({'kind': kind, 'level': level, 'bar': j})
                break

    for k in range(1, len(lows)):
        pl = lows[k-1]; cl = lows[k]
        if cl['i'] < start: continue
        level = pl['price']
        for j in range(cl['i'], min(cl['i'] + 10, n)):
            if close.iloc[j] < level:
                kind = 'BOS_BEAR' if cl['price'] < pl['price'] else 'MSS_BEAR'
                events.append({'kind': kind, 'level': level, 'bar': j})
                break

    if not events:
        return None
    latest = sorted(events, key=lambda x: x['bar'])[-1]
    if latest['bar'] < n - lookback:
        return None
    return latest


def find_order_blocks(df, direction, lookback=60):
    obs = []
    n = len(df)
    start = max(2, n - lookback)

    for i in range(start, n - 3):
        c = df.iloc[i]
        atr_local = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high'] - c['low'])
        min_impulse = atr_local * OB_IMPULSE_ATR_MULT

        if direction == 'LONG':
            if c['close'] >= c['open']: continue
            fwd_high = df['high'].iloc[i+1:min(i+5, n)].max()
            if fwd_high - c['low'] < min_impulse: continue
            ob = {
                'top':    max(c['open'], c['close']),
                'bottom': c['low'],
                'mid':   (max(c['open'], c['close']) + c['low']) / 2,
                'bar':    i
            }
            ob_50 = (ob['top'] + ob['bottom']) / 2
            if (df['close'].iloc[i+1:n] < ob_50).any(): continue
            obs.append(ob)
        else:
            if c['close'] <= c['open']: continue
            fwd_low = df['low'].iloc[i+1:min(i+5, n)].min()
            if c['high'] - fwd_low < min_impulse: continue
            ob = {
                'top':    c['high'],
                'bottom': min(c['open'], c['close']),
                'mid':   (c['high'] + min(c['open'], c['close'])) / 2,
                'bar':    i
            }
            ob_50 = (ob['top'] + ob['bottom']) / 2
            if (df['close'].iloc[i+1:n] > ob_50).any(): continue
            obs.append(ob)

    obs.sort(key=lambda x: x['bar'], reverse=True)
    return obs


def price_in_ob(price, ob, tolerance_pct=OB_TOLERANCE_PCT):
    tol = ob['top'] * tolerance_pct
    return (ob['bottom'] - tol) <= price <= (ob['top'] + tol)


def find_fvg(df, direction, lookback=25):
    fvgs = []
    n = len(df)
    for i in range(max(1, n - lookback), n - 1):
        prev = df.iloc[i-1]; nxt = df.iloc[i+1]
        if direction == 'LONG' and prev['high'] < nxt['low']:
            fvgs.append({'top': nxt['low'], 'bottom': prev['high'],
                         'mid': (nxt['low'] + prev['high']) / 2, 'bar': i})
        elif direction == 'SHORT' and prev['low'] > nxt['high']:
            fvgs.append({'top': prev['low'], 'bottom': nxt['high'],
                         'mid': (prev['low'] + nxt['high']) / 2, 'bar': i})
    return fvgs


def recent_liquidity_sweep(df, direction, highs, lows, lookback=25):
    n = len(df)
    start = n - lookback
    if direction == 'LONG':
        for sl in reversed(lows):
            if sl['i'] < start: continue
            level = sl['price']
            for j in range(sl['i'] + 1, min(sl['i'] + 8, n)):
                c = df.iloc[j]
                if c['low'] < level and c['close'] > level:
                    return {'level': level, 'bar': j, 'type': 'SWEEP_LOW'}
    else:
        for sh in reversed(highs):
            if sh['i'] < start: continue
            level = sh['price']
            for j in range(sh['i'] + 1, min(sh['i'] + 8, n)):
                c = df.iloc[j]
                if c['high'] > level and c['close'] < level:
                    return {'level': level, 'bar': j, 'type': 'SWEEP_HIGH'}
    return None


def pd_zone(df_4h, price):
    hi = df_4h['high'].iloc[-50:].max()
    lo = df_4h['low'].iloc[-50:].min()
    rang = hi - lo
    if rang == 0: return 'NEUTRAL', 0.5
    pos = (price - lo) / rang
    if pos < 0.40:   return 'DISCOUNT', pos
    elif pos > 0.60: return 'PREMIUM',  pos
    return 'NEUTRAL', pos


# ── SCORER (identical to live bot) ───────────────────────────────────────────

def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score = 0
    reasons = []

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1] if len(df_15m) > 0 else pd.Series(dtype=float)
    l4  = df_4h.iloc[-1]

    # 1. Structure (20 pts)
    if structure:
        if 'MSS' in structure['kind']:
            score += 20; reasons.append(f"MSS ({structure['kind']})")
        else:
            score += 14; reasons.append(f"BOS ({structure['kind']})")

    # 2. OB quality (20 pts)
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        if ob_size_pct < 0.8:
            score += 20; reasons.append(f"Tight OB {ob_size_pct:.2f}%")
        elif ob_size_pct < 2.0:
            score += 13; reasons.append(f"OB {ob_size_pct:.2f}%")
        else:
            score += 7;  reasons.append(f"Wide OB {ob_size_pct:.2f}%")

    # 3. 4H Trend Alignment (15 pts)
    e21 = l4.get('ema_21', 0) if hasattr(l4, 'get') else l4['ema_21'] if 'ema_21' in l4.index else 0
    e50 = l4.get('ema_50', 0) if hasattr(l4, 'get') else l4['ema_50'] if 'ema_50' in l4.index else 0
    e200= l4.get('ema_200',0) if hasattr(l4, 'get') else l4['ema_200'] if 'ema_200' in l4.index else 0

    def safe_get(row, key, default=0):
        try:
            v = row[key]
            return default if pd.isna(v) else v
        except: return default

    e21 = safe_get(l4, 'ema_21'); e50 = safe_get(l4, 'ema_50'); e200 = safe_get(l4, 'ema_200')

    if direction == 'LONG':
        if e21 > e50 > e200:
            score += 15; reasons.append("4H Triple EMA Bull")
        elif e21 > e50:
            score += 10; reasons.append("4H EMA 21>50 Bull")
        elif pd_label == 'DISCOUNT':
            score += 6;  reasons.append("4H Discount zone")
    else:
        if e21 < e50 < e200:
            score += 15; reasons.append("4H Triple EMA Bear")
        elif e21 < e50:
            score += 10; reasons.append("4H EMA 21<50 Bear")
        elif pd_label == 'PREMIUM':
            score += 6;  reasons.append("4H Premium zone")

    # 4. HH/LL bonus (8 pts)
    if hh_ll_confirmed:
        score += HH_LL_BONUS; reasons.append("4H HH/LL confirmed")

    # 5. 1H Entry Trigger (25 pts)
    trigger = False
    if direction == 'LONG':
        if safe_get(l1, 'bull_engulf') == 1:
            score += 25; trigger = True; reasons.append("1H Bull Engulf")
        elif safe_get(l1, 'bull_pin') == 1:
            score += 22; trigger = True; reasons.append("1H Bull Pin")
        elif safe_get(l1, 'hammer') == 1:
            score += 18; trigger = True; reasons.append("1H Hammer")
        elif safe_get(p1, 'bull_engulf') == 1:
            score += 14; trigger = True; reasons.append("1H Bull Engulf (prev)")
        elif safe_get(p1, 'bull_pin') == 1:
            score += 11; trigger = True; reasons.append("1H Bull Pin (prev)")
        elif safe_get(p1, 'hammer') == 1:
            score += 9;  trigger = True; reasons.append("1H Hammer (prev)")
    else:
        if safe_get(l1, 'bear_engulf') == 1:
            score += 25; trigger = True; reasons.append("1H Bear Engulf")
        elif safe_get(l1, 'bear_pin') == 1:
            score += 22; trigger = True; reasons.append("1H Bear Pin")
        elif safe_get(l1, 'shooting_star') == 1:
            score += 18; trigger = True; reasons.append("1H Shooting Star")
        elif safe_get(p1, 'bear_engulf') == 1:
            score += 14; trigger = True; reasons.append("1H Bear Engulf (prev)")
        elif safe_get(p1, 'bear_pin') == 1:
            score += 11; trigger = True; reasons.append("1H Bear Pin (prev)")
        elif safe_get(p1, 'shooting_star') == 1:
            score += 9;  trigger = True; reasons.append("1H Shooting Star (prev)")

    if not trigger:
        score -= 12

    # 6. Momentum (12 pts)
    rsi1  = safe_get(l1, 'rsi', 50)
    macd1 = safe_get(l1, 'macd'); ms1  = safe_get(l1, 'macd_signal')
    pm1   = safe_get(p1, 'macd'); pms1 = safe_get(p1, 'macd_signal')
    sk1   = safe_get(l1, 'srsi_k', 0.5); sd1 = safe_get(l1, 'srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55:
            score += 4; reasons.append(f"RSI reset {rsi1:.0f}")
        elif rsi1 < 28:
            score += 3; reasons.append(f"RSI oversold {rsi1:.0f}")
        if macd1 > ms1 and pm1 <= pms1:
            score += 5; reasons.append("MACD bull cross")
        elif macd1 > ms1:
            score += 2; reasons.append("MACD bullish")
        if sk1 < 0.3 and sk1 > sd1:
            score += 3; reasons.append("Stoch bull cross")
    else:
        if 45 <= rsi1 <= 72:
            score += 4; reasons.append(f"RSI overbought zone {rsi1:.0f}")
        elif rsi1 > 72:
            score += 3; reasons.append(f"RSI overbought {rsi1:.0f}")
        if macd1 < ms1 and pm1 >= pms1:
            score += 5; reasons.append("MACD bear cross")
        elif macd1 < ms1:
            score += 2; reasons.append("MACD bearish")
        if sk1 > 0.7 and sk1 < sd1:
            score += 3; reasons.append("Stoch bear cross")

    # 7. Extras (10 pts max)
    extras = 0
    if sweep:
        extras += 4; reasons.append("Liq sweep")
    if fvg_near:
        extras += 3; reasons.append("FVG overlaps OB")

    # 15M vol bonus
    if len(df_15m) > 0:
        vr15 = safe_get(df_15m.iloc[-1], 'vol_ratio', 1.0)
        if   vr15 >= 2.5: extras += 3; reasons.append(f"15M vol spike {vr15:.1f}x")
        elif vr15 >= 1.5: extras += 1; reasons.append(f"15M vol elevated {vr15:.1f}x")

    # VWAP
    close1 = safe_get(l1, 'close'); vwap1 = safe_get(l1, 'vwap')
    if direction == 'LONG' and close1 < vwap1:
        extras = min(extras+1, 10); reasons.append("Below 1H VWAP")
    elif direction == 'SHORT' and close1 > vwap1:
        extras = min(extras+1, 10); reasons.append("Above 1H VWAP")

    score += min(extras, 10)

    return max(0, min(int(score), 100)), reasons


# ── SIGNAL DETECTION ─────────────────────────────────────────────────────────

def analyse_candle(df_4h, df_1h, df_15m, signal_bar_idx):
    """
    Run the full SMC gate + scoring pipeline at a specific 1H bar.
    signal_bar_idx: index into df_1h where we are evaluating.
    Returns signal dict or None.
    """
    # Slice data as-of signal bar (no lookahead)
    df1_slice  = df_1h.iloc[:signal_bar_idx+1].copy()
    df15_slice = df_15m.copy()  # approximate — 15M only used for vol bonus

    if len(df1_slice) < 80:
        return None

    # ── GATE 1: 4H EMA Bias ──
    l4 = df_4h.iloc[-1]
    e21 = l4['ema_21'] if 'ema_21' in l4.index and not pd.isna(l4['ema_21']) else 0
    e50 = l4['ema_50'] if 'ema_50' in l4.index and not pd.isna(l4['ema_50']) else 0
    if e21 > e50:       bias = 'LONG'
    elif e21 < e50:     bias = 'SHORT'
    else:               return None

    # ── v2 GATE: SHORT ONLY ──
    # v1 data: LONGs = 35% WR, net negative. Kill entirely.
    if SHORT_ONLY and bias == 'LONG':
        return None

    price = df1_slice['close'].iloc[-1]

    # ── GATE 2: PD Zone ──
    pd_label, pd_pos = pd_zone(df_4h, price)
    if bias == 'LONG' and pd_label == 'PREMIUM':    return None
    if bias == 'SHORT' and pd_label == 'DISCOUNT':  return None

    # ── HH/LL check (used by v3 TRENDING gate) ──
    hh_ll_ok = check_4h_hh_ll(df_4h, bias, HH_LL_LOOKBACK)

    # ── GATE 3: 1H Structure ──
    highs1, lows1 = swing_highs_lows(df1_slice, left=4, right=4)
    structure = detect_structure_break(df1_slice, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
    if structure:
        s_bull = 'BULL' in structure['kind']
        s_bear = 'BEAR' in structure['kind']
        if bias == 'LONG' and s_bear: return None
        if bias == 'SHORT' and s_bull: return None
    
    # ── v2 GATE: REQUIRE BEAR STRUCTURE ──
    # No-structure signals had 25% WR in v1. Bull structure = 35% WR.
    # Only allow BOS_BEAR or MSS_BEAR.
    if REQUIRE_BEAR_STRUCTURE:
        if structure is None: return None
        if 'BULL' in structure['kind']: return None

    # ── v3 GATE: BOS_BEAR ONLY (block MSS_BEAR) ──
    # MSS_BEAR = 0% WR in v2 (n=3). BOS_BEAR = 59.6% WR.
    if REQUIRE_BOS_ONLY:
        if structure is None or structure['kind'] != 'BOS_BEAR': return None

    # ── v3 GATE: REQUIRE TRENDING (HH/LL confirmed) ──
    # Ranging = 33% WR in v2 (n=3). Trending = 57.4% WR.
    if REQUIRE_TRENDING and not hh_ll_ok:
        return None

    # ── GATE 4: Order Block (hard gate) ──
    obs = find_order_blocks(df1_slice, bias, lookback=60)
    if not obs: return None

    active_ob = None
    for ob in obs:
        if price_in_ob(price, ob, OB_TOLERANCE_PCT):
            active_ob = ob; break

    if not active_ob: return None

    # ── EXTRAS: FVG, Sweep ──
    fvgs = find_fvg(df1_slice, bias, lookback=25)
    fvg_near = None
    for fvg in fvgs:
        if fvg['bottom'] < active_ob['top'] and fvg['top'] > active_ob['bottom']:
            fvg_near = fvg; break

    sweep = recent_liquidity_sweep(df1_slice, bias, highs1, lows1, lookback=20)

    # ── v3 GATE: BLOCK SWEEP setups ──
    # Sweep=YES = 40% WR in v2 vs 62.9% without sweep. Kill them.
    if BLOCK_SWEEP and sweep is not None:
        return None

    # ── GATE 5: Score ──
    score, reasons = score_setup(
        bias, active_ob, structure, sweep, fvg_near,
        df1_slice, df15_slice, df_4h, pd_label, hh_ll_ok
    )

    if score < MIN_SCORE: return None

    # ── Build signal ──
    atr1  = df1_slice['atr'].iloc[-1]
    if pd.isna(atr1) or atr1 <= 0: return None

    entry = price
    if bias == 'LONG':
        sl = active_ob['bottom'] - atr1 * 0.2
        sl = min(sl, entry - atr1 * 0.4)  # v2: tighter SL
    else:
        sl = active_ob['top'] + atr1 * 0.2
        sl = max(sl, entry + atr1 * 0.4)  # v2: tighter SL

    risk = abs(entry - sl)
    if risk < entry * 0.001: return None

    if bias == 'LONG':
        tps = [entry + risk * rr for rr in TP_RR]
    else:
        tps = [entry - risk * rr for rr in TP_RR]

    quality = 'ELITE' if score >= 92 else 'PREMIUM' if score >= 85 else 'HIGH'

    return {
        'direction':  bias,
        'entry':      entry,
        'sl':         sl,
        'tps':        tps,
        'risk':       risk,
        'risk_pct':   risk / entry * 100,
        'score':      score,
        'quality':    quality,
        'hh_ll':      hh_ll_ok,
        'pd_zone':    pd_label,
        'structure':  structure['kind'] if structure else 'none',
        'ob_size':    (active_ob['top'] - active_ob['bottom']) / active_ob['bottom'] * 100,
        'has_fvg':    fvg_near is not None,
        'has_sweep':  sweep is not None,
        'reasons':    ', '.join(reasons[:8]),
    }


# ── TRADE SIMULATION ─────────────────────────────────────────────────────────

def simulate_trade(signal, df_1h_future):
    """
    Simulate a trade on future 1H candles.
    Returns dict with outcome, blended pnl, timestamps.
    """
    entry     = signal['entry']
    sl        = signal['sl']
    tps       = signal['tps']
    direction = signal['direction']
    risk_pct  = signal['risk_pct']  # SL distance %

    tp_hit     = [False, False, False]
    sl_hit     = False
    outcome    = 'TIMEOUT'
    blended_pnl = 0.0
    remaining_pos = 1.0  # fraction of position still open

    timeout_bars = TIMEOUT_HOURS  # 1H candles = hours

    for i, (_, c) in enumerate(df_1h_future.iterrows()):
        if i >= timeout_bars:
            # Timeout: close remaining at last close
            timeout_pct = (c['close'] - entry) / entry * 100 if direction == 'LONG' else (entry - c['close']) / entry * 100
            blended_pnl += remaining_pos * timeout_pct
            outcome = 'TIMEOUT'
            break

        hi = c['high']; lo = c['low']

        if direction == 'LONG':
            # Check SL first (conservative — assume SL can hit before TP in same candle)
            if lo <= sl and not any(tp_hit):
                sl_pct = (sl - entry) / entry * 100  # negative
                blended_pnl += remaining_pos * sl_pct
                outcome = 'SL'; sl_hit = True; break

            # Check TPs in order
            for t_idx, tp in enumerate(tps):
                if not tp_hit[t_idx] and hi >= tp:
                    tp_pct = (tp - entry) / entry * 100
                    close_frac = TP_PCT[t_idx]
                    blended_pnl += close_frac * tp_pct
                    remaining_pos -= close_frac
                    tp_hit[t_idx] = True
                    if t_idx == 2:
                        outcome = 'TP3'; break

            if tp_hit[2]: break
            if remaining_pos <= 0: break

        else:  # SHORT
            # SL check
            if hi >= sl and not any(tp_hit):
                sl_pct = (entry - sl) / entry * 100  # negative
                blended_pnl += remaining_pos * (-abs(sl_pct))
                outcome = 'SL'; sl_hit = True; break

            # Check TPs
            for t_idx, tp in enumerate(tps):
                if not tp_hit[t_idx] and lo <= tp:
                    tp_pct = (entry - tp) / entry * 100
                    close_frac = TP_PCT[t_idx]
                    blended_pnl += close_frac * tp_pct
                    remaining_pos -= close_frac
                    tp_hit[t_idx] = True
                    if t_idx == 2:
                        outcome = 'TP3'; break

            if tp_hit[2]: break
            if remaining_pos <= 0: break
    else:
        # Loop completed without break — timeout on last bar
        if len(df_1h_future) > 0:
            last_close = df_1h_future.iloc[-1]['close']
            timeout_pct = (last_close - entry) / entry * 100 if direction == 'LONG' else (entry - last_close) / entry * 100
            blended_pnl += remaining_pos * timeout_pct

    # Determine outcome label
    if sl_hit:
        outcome = 'SL'
    elif tp_hit[2]:
        outcome = 'TP3'
    elif tp_hit[1]:
        outcome = 'TP2'
    elif tp_hit[0]:
        outcome = 'TP1'
    elif any(tp_hit):
        outcome = f"TP{sum(tp_hit)}"
    else:
        outcome = 'TIMEOUT'

    return {
        'outcome':     outcome,
        'pnl':         round(blended_pnl, 3),
        'tp_hit':      tp_hit,
        'sl_hit':      sl_hit,
        'tp1_hit':     tp_hit[0],
        'tp2_hit':     tp_hit[1],
        'tp3_hit':     tp_hit[2],
    }


# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────

async def run_backtest():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    print(f"""
╔══════════════════════════════════════════════════════╗
║           SMC PRO v4 — BACKTEST v2.0                ║
║  {LOOKBACK_DAYS}d | Top {TOP_N_PAIRS} pairs | Score≥{MIN_SCORE} | 1H trigger   ║
║  TP1={TP_RR[0]}R | TP2={TP_RR[1]}R | TP3={TP_RR[2]}R | Timeout={TIMEOUT_HOURS}H  ║
╚══════════════════════════════════════════════════════╝
""")

    # ── Load pairs ──
    logger.info("Loading pairs...")
    markets = await exchange.load_markets()
    tickers = await exchange.fetch_tickers()
    pairs = []
    for sym, mkt in markets.items():
        if not (mkt.get('swap') and mkt.get('quote') == 'USDT' and mkt.get('active')):
            continue
        vol = (tickers.get(sym, {}).get('quoteVolume') or 0)
        if vol >= MIN_VOLUME_24H:
            pairs.append((sym, vol))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = [p[0] for p in pairs[:TOP_N_PAIRS]]
    logger.info(f"✅ {len(pairs)} pairs loaded")

    limit_1h  = LOOKBACK_DAYS * 24 + 250
    limit_4h  = LOOKBACK_DAYS * 6  + 100
    limit_15m = 100  # only need recent for vol bonus (approximate)

    all_signals = []
    cooldown = {}  # symbol -> last signal ts (prevent re-entry within 48H)

    for i, symbol in enumerate(pairs):
        try:
            raw_4h  = await exchange.fetch_ohlcv(symbol, '4h',  limit=limit_4h)
            raw_1h  = await exchange.fetch_ohlcv(symbol, '1h',  limit=limit_1h)
            raw_15m = await exchange.fetch_ohlcv(symbol, '15m', limit=limit_15m)

            if not raw_4h or not raw_1h or len(raw_4h) < 60 or len(raw_1h) < 100:
                continue

            df_4h  = pd.DataFrame(raw_4h,  columns=['ts','open','high','low','close','volume'])
            df_1h  = pd.DataFrame(raw_1h,  columns=['ts','open','high','low','close','volume'])
            df_15m = pd.DataFrame(raw_15m, columns=['ts','open','high','low','close','volume'])

            for df in [df_4h, df_1h, df_15m]:
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')

            df_4h  = add_indicators(df_4h)
            df_1h  = add_indicators(df_1h)
            df_15m = add_indicators(df_15m)

            # Walk-forward window on 1H candles
            cutoff_ts = df_1h['ts'].max() - pd.Timedelta(days=LOOKBACK_DAYS)
            window_indices = df_1h.index[df_1h['ts'] >= cutoff_ts].tolist()

            sym_signals = 0

            for idx_w in window_indices:
                bar_ts = df_1h.loc[idx_w, 'ts']

                # Cooldown: skip if signal within last 48H on this symbol
                if symbol in cooldown:
                    elapsed = (bar_ts - cooldown[symbol]).total_seconds() / 3600
                    if elapsed < TIMEOUT_HOURS:
                        continue

                # Align 4H data: use 4H bars up to this 1H bar
                df_4h_slice = df_4h[df_4h['ts'] <= bar_ts].copy()
                if len(df_4h_slice) < 60:
                    continue

                # df_15m is approximate (recent only) — only for vol bonus
                df_15m_recent = df_15m[df_15m['ts'] <= bar_ts].tail(5).copy()

                # Run analysis at this bar
                sig = analyse_candle(df_4h_slice, df_1h, df_15m_recent, idx_w)
                if sig is None:
                    continue

                # Simulate trade on future 1H candles
                future_1h = df_1h.iloc[idx_w+1:idx_w+1+TIMEOUT_HOURS+5].copy()
                if len(future_1h) == 0:
                    continue

                result = simulate_trade(sig, future_1h)

                all_signals.append({
                    'symbol':    symbol.replace('/USDT:USDT', ''),
                    'timestamp': bar_ts,
                    'direction': sig['direction'],
                    'score':     sig['score'],
                    'quality':   sig['quality'],
                    'hh_ll':     sig['hh_ll'],
                    'pd_zone':   sig['pd_zone'],
                    'structure': sig['structure'],
                    'ob_size':   round(sig['ob_size'], 3),
                    'has_fvg':   sig['has_fvg'],
                    'has_sweep': sig['has_sweep'],
                    'risk_pct':  round(sig['risk_pct'], 2),
                    'entry':     round(sig['entry'], 6),
                    'sl':        round(sig['sl'], 6),
                    'tp1':       round(sig['tps'][0], 6),
                    'reasons':   sig['reasons'],
                    'outcome':   result['outcome'],
                    'pnl':       result['pnl'],
                    'tp1_hit':   result['tp1_hit'],
                    'tp2_hit':   result['tp2_hit'],
                    'tp3_hit':   result['tp3_hit'],
                })

                cooldown[symbol] = bar_ts
                sym_signals += 1

            if sym_signals > 0:
                logger.info(f"  📊 {symbol.replace('/USDT:USDT','')}... → {sym_signals} signals")

        except Exception as e:
            logger.debug(f"[ERR] {symbol}: {e}")
            continue

        if (i+1) % 20 == 0:
            logger.info(f"── {i+1}/{len(pairs)} pairs | {len(all_signals)} signals ──")

    await exchange.close()

    # ── RESULTS ──────────────────────────────────────────────────────────────

    if not all_signals:
        print("❌ No signals found")
        return

    df = pd.DataFrame(all_signals)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df = df.sort_values('timestamp').reset_index(drop=True)

    total    = len(df)
    days     = LOOKBACK_DAYS
    per_day  = round(total / days, 2)

    wins     = df[df['pnl'] > 0]
    losses   = df[df['pnl'] <= 0]
    wr       = len(wins) / total * 100 if total > 0 else 0

    avg_win  = wins['pnl'].mean()  if len(wins)   > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    pf       = abs(avg_win * len(wins)) / abs(avg_loss * len(losses)) if len(losses) > 0 and avg_loss != 0 else 999

    n_sl      = (df['outcome'] == 'SL').sum()
    n_tp1     = df['tp1_hit'].sum()
    n_tp2     = df['tp2_hit'].sum()
    n_tp3     = df['tp3_hit'].sum()
    n_timeout = (df['outcome'] == 'TIMEOUT').sum()

    # Equity simulation (sequential, 2% risk)
    equity        = 1000.0
    peak_equity   = equity
    max_dd_equity = 0.0
    equity_curve  = [equity]

    for _, row in df.sort_values('timestamp').iterrows():
        sl_frac    = row['risk_pct'] / 100 if row['risk_pct'] > 0 else 0.05
        risk_amt   = equity * RISK_PER_TRADE
        pos_value  = risk_amt / sl_frac
        dollar_pnl = pos_value * (row['pnl'] / 100)
        equity    += dollar_pnl
        equity     = max(equity, 0.01)
        equity_curve.append(equity)
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity * 100
        if dd < max_dd_equity:
            max_dd_equity = dd

    equity_return = (equity_curve[-1] / equity_curve[0] - 1) * 100

    # ── Print Results ────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════╗
║      📊 SMC PRO v4 BACKTEST v3 — RESULTS            ║
╚══════════════════════════════════════════════════════╝

  Settings: Score≥{MIN_SCORE} | 1H trigger | {LOOKBACK_DAYS}d | Top {TOP_N_PAIRS} pairs
  Risk: {RISK_PER_TRADE*100:.0f}%/trade

  ── Raw Signal Stats ──
  Signals         : {total}  ({per_day}/day  |  {round(per_day*30,1)}/month)
  Win Rate        : {wr:.1f}%
  Profit Factor   : {pf:.2f}
  Avg PnL/trade   : +{df['pnl'].mean():.3f}%
  Avg Win         : +{avg_win:.3f}%
  Avg Loss        : {avg_loss:.3f}%

  ── Outcome Breakdown ──
  🎯 TP1 reached  : {n_tp1}  ({n_tp1/total*100:.1f}%)
  🎯 TP2 reached  : {n_tp2}  ({n_tp2/total*100:.1f}%)
  🎯 TP3 reached  : {n_tp3}  ({n_tp3/total*100:.1f}%)
  ⛔ SL hit       : {n_sl}  ({n_sl/total*100:.1f}%)
  ⏰ Timeout      : {n_timeout}  ({n_timeout/total*100:.1f}%)

  ── Equity Simulation ──
  Starting equity : $1,000
  Final equity    : ${equity_curve[-1]:,.2f}
  Total return    : +{equity_return:.1f}%
  Max Drawdown    : {max_dd_equity:.2f}%
""")

    # ── By Quality ──
    print("  ── By Quality ──")
    for q in ['ELITE', 'PREMIUM', 'HIGH']:
        sub = df[df['quality'] == q]
        if len(sub) > 0:
            q_wr = (sub['pnl'] > 0).mean() * 100
            q_avg = sub['pnl'].mean()
            print(f"  {q:8s} | n={len(sub):3d} | WR={q_wr:.1f}% | Avg={q_avg:+.3f}%")

    # ── By Score Band ──
    print("\n  ── By Score Band ──")
    print(f"  {'Band':<12} {'n':>5} {'WR%':>8} {'Avg%':>8} {'SL%':>8}")
    for lo_b, hi_b in [(75,80),(80,85),(85,90),(90,95),(95,101)]:
        sub = df[(df['score'] >= lo_b) & (df['score'] < hi_b)]
        if len(sub) > 0:
            b_wr  = (sub['pnl'] > 0).mean() * 100
            b_avg = sub['pnl'].mean()
            b_sl  = (sub['outcome'] == 'SL').mean() * 100
            print(f"  {lo_b}-{hi_b}%       {len(sub):>5}  {b_wr:>7.1f}%  {b_avg:>+7.3f}%  {b_sl:>7.1f}%")

    # ── By Direction ──
    print("\n  ── By Direction ──")
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) > 0:
            d_wr = (sub['pnl'] > 0).mean() * 100
            d_avg = sub['pnl'].mean()
            print(f"  {d:5s} | n={len(sub):3d} | WR={d_wr:.1f}% | Avg={d_avg:+.3f}%")

    # ── By Structure ──
    print("\n  ── By Structure Type ──")
    for st in df['structure'].unique():
        sub = df[df['structure'] == st]
        if len(sub) >= 3:
            st_wr = (sub['pnl'] > 0).mean() * 100
            st_avg = sub['pnl'].mean()
            print(f"  {st:12s} | n={len(sub):3d} | WR={st_wr:.1f}% | Avg={st_avg:+.3f}%")

    # ── HH/LL Impact ──
    print("\n  ── HH/LL Trend Impact ──")
    for hh in [True, False]:
        sub = df[df['hh_ll'] == hh]
        if len(sub) > 0:
            hh_wr = (sub['pnl'] > 0).mean() * 100
            hh_avg = sub['pnl'].mean()
            tag = 'TRENDING' if hh else 'RANGING'
            print(f"  {tag:10s} | n={len(sub):3d} | WR={hh_wr:.1f}% | Avg={hh_avg:+.3f}%")

    # ── FVG/Sweep Impact ──
    print("\n  ── Confluence Impact ──")
    for col, label in [('has_fvg','FVG'), ('has_sweep','Sweep')]:
        for val in [True, False]:
            sub = df[df[col] == val]
            if len(sub) > 3:
                c_wr = (sub['pnl'] > 0).mean() * 100
                tag = f"{label}={'YES' if val else 'NO '}"
                print(f"  {tag} | n={len(sub):3d} | WR={c_wr:.1f}%")

    # ── Top Symbols ──
    print("\n  ── Top 15 Symbols ──")
    sym_stats = df.groupby('symbol').agg(
        n=('pnl','count'),
        wr=('pnl', lambda x: (x > 0).mean() * 100),
        avg=('pnl','mean')
    ).query('n >= 2').sort_values('wr', ascending=False).head(15)
    print(f"  {'Symbol':<12} {'n':>4} {'WR%':>7} {'Avg%':>8}")
    for sym, row in sym_stats.iterrows():
        print(f"  {sym:<12} {int(row['n']):>4}  {row['wr']:>6.1f}%  {row['avg']:>+7.3f}%")

    # ── Save Excel ──
    print(f"\n  💾 Saving to {OUTPUT_FILE}...")
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All Signals', index=False)

            # Summary sheet
            summary = {
                'Metric': [
                    'Total Signals', 'Per Day', 'Per Month',
                    'Win Rate %', 'Profit Factor',
                    'Avg PnL %', 'Avg Win %', 'Avg Loss %',
                    'TP1 Hit', 'TP2 Hit', 'TP3 Hit', 'SL Hit', 'Timeout',
                    'Starting Equity', 'Final Equity', 'Total Return %', 'Max Drawdown %',
                ],
                'Value': [
                    total, round(per_day,2), round(per_day*30,1),
                    round(wr,1), round(pf,2),
                    round(df['pnl'].mean(),3), round(avg_win,3), round(avg_loss,3),
                    int(n_tp1), int(n_tp2), int(n_tp3), int(n_sl), int(n_timeout),
                    1000, round(equity_curve[-1],2), round(equity_return,1), round(max_dd_equity,2),
                ]
            }
            pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
        print(f"  ✅ Saved!")
    except Exception as e:
        print(f"  ❌ Save failed: {e}")
        # Fallback: save as CSV
        csv_path = OUTPUT_FILE.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"  💾 Saved as CSV: {csv_path}")

if __name__ == '__main__':
    asyncio.run(run_backtest())
