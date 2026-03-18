"""
ADVANCED DAY TRADING SCANNER v9.1
===================================
Fixes from live session analysis (3/17/2026):

  FIX 1 — Score raised
    MIN_SCORE_PCT : 0.47 → 0.55
    Cuts ~60% of weak 50-54% signals that were flooding the feed

  FIX 2 — ATR levels fixed (RR was 0.4:1 — too low)
    TP1 : 0.6x → 1.0x ATR   (RR now 0.67:1 instead of 0.4:1)
    TP2 : 1.2x → 1.8x ATR
    TP3 : 2.0x → 3.0x ATR
    SL  : 1.5x   unchanged

  FIX 3 — Mandatory conviction filter added
    Every signal must have MACD cross OR vol spike (>2.5x)
    Kills trend-only signals with no catalyst
    MACD cross = 94.7% WR in backtest
    Vol spike   = 89-90% WR in backtest
    This alone eliminates most BE-close patterns

Install:
  pip install ccxt ta pandas numpy python-telegram-bot
"""

import asyncio
import ccxt.async_support as ccxt
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
# ★  SETTINGS v9.1
# ═══════════════════════════════════════════════════

REGIME_MODE           = 'HARD'
USE_LONG_TREND_FILTER = True

ATR_SL_MULT  = 1.5
ATR_TP1_MULT = 1.0    # FIX 2: was 0.6 → RR now 0.67:1
ATR_TP2_MULT = 1.8    # FIX 2: was 1.2
ATR_TP3_MULT = 3.0    # FIX 2: was 2.0

# Direction-aware partial close ratios (unchanged)
LONG_TP1_PCT  = 0.33
LONG_TP2_PCT  = 0.33
LONG_TP3_PCT  = 0.34
SHORT_TP1_PCT = 0.70
SHORT_TP2_PCT = 0.30

MIN_SCORE_PCT       = 0.55    # FIX 1: was 0.47 — kills weak 50-54% band
QUALITY_PREMIUM_PCT = 0.70    # recalibrated for new floor

SCAN_INTERVAL_MIN = 15
MIN_VOLUME_USDT   = 500_000
MAX_TRADE_HOURS   = 24

# ═══════════════════════════════════════════════════


class AdvancedDayTradingScanner:
    def __init__(self, telegram_token, telegram_chat_id,
                 binance_api_key=None, binance_secret=None):
        self.telegram_token = telegram_token
        self.telegram_bot   = Bot(token=telegram_token)
        self.chat_id        = telegram_chat_id
        self.exchange = ccxt.binance({
            'apiKey':          binance_api_key,
            'secret':          binance_secret,
            'enableRateLimit': True,
            'options':         {'defaultType': 'future'},
        })
        self.signal_history = deque(maxlen=300)
        self.active_trades  = {}
        self.pair_cooldown  = {}
        self.btc_regime     = None
        self.btc_price      = None
        self.btc_ema        = None
        self.stats = {
            'total_signals':   0,
            'long_signals':    0,
            'short_signals':   0,
            'premium_signals': 0,
            'good_signals':    0,
            'tp1_hits':        0,
            'tp2_hits':        0,
            'tp3_hits':        0,
            'sl_hits':         0,
            'timeouts':        0,
            'be_saves':        0,
            'regime_blocked':  0,
            'filtered_long':   0,
            'conviction_filtered': 0,   # FIX 3 counter
            'last_scan':       None,
            'pairs_scanned':   0,
            'session_start':   datetime.now(),
        }
        self.is_scanning = False

    # ── BTC Regime ────────────────────────────────────────────

    async def update_btc_regime(self):
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=30)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['ema21']     = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            prev            = self.btc_regime
            self.btc_price  = df['close'].iloc[-1]
            self.btc_ema    = df['ema21'].iloc[-1]
            self.btc_regime = 'BULL' if self.btc_price > self.btc_ema else 'BEAR'
            if prev and prev != self.btc_regime:
                logger.info(f"🔄 Regime flip: {prev} → {self.btc_regime}")
            logger.info(f"📡 BTC: {self.btc_regime} (${self.btc_price:,.0f})")
        except Exception as e:
            logger.error(f"Regime error: {e}")

    # ── Pairs ─────────────────────────────────────────────────

    async def get_all_usdt_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = [
                s for s in self.exchange.symbols
                if s.endswith('/USDT:USDT')
                and 'PERP' not in s
                and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_USDT
            ]
            pairs.sort(key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            logger.info(f"✅ {len(pairs)} pairs")
            return pairs
        except Exception as e:
            logger.error(f"Pairs error: {e}")
            return []

    # ── Data ──────────────────────────────────────────────────

    async def fetch_data(self, symbol):
        data = {}
        try:
            for tf, limit in [('1h', 100), ('4h', 100), ('15m', 50)]:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                data[tf] = df
                await asyncio.sleep(0.05)
            return data
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}")
            return None

    # ── Indicators ────────────────────────────────────────────

    def _supertrend(self, df, period=10, mult=3):
        try:
            hl2   = (df['high'] + df['low']) / 2
            atr   = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=period
            ).average_true_range()
            upper = hl2 + mult * atr
            lower = hl2 - mult * atr
            st = [0.0] * len(df)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper.iloc[i-1]:   st[i] = lower.iloc[i]
                elif df['close'].iloc[i] < lower.iloc[i-1]: st[i] = upper.iloc[i]
                else:                                        st[i] = st[i-1]
            return pd.Series(st, index=df.index)
        except:
            return pd.Series([0.0]*len(df), index=df.index)

    def add_indicators(self, df):
        if len(df) < 30:
            return df
        try:
            df['ema_9']       = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21']      = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50']      = ta.trend.EMAIndicator(df['close'], window=min(50,len(df)-1)).ema_indicator()
            df['supertrend']  = self._supertrend(df)
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
            logger.error(f"Indicators: {e}")
        return df

    def _vol_spike(self, df):
        if len(df) < 20: return False, 1.0
        avg = df['volume'].iloc[-20:].mean()
        if avg == 0 or pd.isna(avg): return False, 1.0
        r = df['volume'].iloc[-1] / avg
        return r > 2.5, round(r, 2)

    # ── Scoring v8 (unchanged weights) ───────────────────────

    def _score(self, r1h, p1h, r4h, r15m, vol_ratio):
        ls = ss = 0
        lr = []; sr = []

        mcb = r1h['macd'] > r1h['macd_signal'] and p1h['macd'] <= p1h['macd_signal']
        mcs = r1h['macd'] < r1h['macd_signal'] and p1h['macd'] >= p1h['macd_signal']
        spk = vol_ratio > 2.5

        # TREND 6pts
        if r4h['ema_9'] > r4h['ema_21'] > r4h['ema_50']:
            ls += 3; lr.append('🔥 4H Uptrend')
        elif r4h['ema_9'] < r4h['ema_21'] < r4h['ema_50']:
            ss += 3; sr.append('🔥 4H Downtrend')

        if r1h['ema_9'] > r1h['ema_21']:
            ls += 2; lr.append('1H EMA Bull')
        elif r1h['ema_9'] < r1h['ema_21']:
            ss += 2; sr.append('1H EMA Bear')

        if r1h['close'] > r1h['supertrend']:
            ls += 1; lr.append('SuperTrend ↑')
        elif r1h['close'] < r1h['supertrend']:
            ss += 1; sr.append('SuperTrend ↓')

        # MOMENTUM ~7pts
        rsi = r1h['rsi']
        if rsi < 30:    ls += 2.0; lr.append(f'RSI {rsi:.0f}')
        elif rsi < 40:  ls += 2;   lr.append(f'RSI Low {rsi:.0f}')
        elif rsi <= 50: ls += 1;   lr.append(f'RSI {rsi:.0f}')
        if rsi > 70:    ss += 2.0; sr.append(f'RSI {rsi:.0f}')
        elif rsi > 60:  ss += 2;   sr.append(f'RSI High {rsi:.0f}')

        sk = r1h['stoch_rsi_k']; sd = r1h['stoch_rsi_d']
        if sk < 0.2 and sk > sd:   ls += 2; lr.append('⚡ StochRSI ↑')
        elif sk > 0.8 and sk < sd: ss += 2; sr.append('⚡ StochRSI ↓')

        if mcb: ls += 3; lr.append('🎯 MACD ↑')
        elif mcs: ss += 3; sr.append('🎯 MACD ↓')

        # VOLUME 5.5pts
        if spk:
            if r1h['close'] > p1h['close']: ls += 3.5; lr.append(f'🚀 Vol {vol_ratio:.1f}x')
            else:                           ss += 3;   sr.append(f'💥 Dump {vol_ratio:.1f}x')

        if r1h['mfi'] < 20:   ls += 0.5; lr.append(f'MFI {r1h["mfi"]:.0f}')
        elif r1h['mfi'] > 80: ss += 1.5; sr.append(f'MFI {r1h["mfi"]:.0f}')

        if r1h['cmf'] > 0.15:    ls += 1; lr.append('CMF Buy')
        elif r1h['cmf'] < -0.15: ss += 1; sr.append('CMF Sell')

        if r1h['obv'] > r1h['obv_ema']: ls += 0.5; lr.append('OBV ↑')
        else:                            ss += 0.5; sr.append('OBV ↓')

        # VOLATILITY
        bbp = r1h['bb_pband']
        if bbp < 0.1:   ls += 2.5; lr.append('💎 Lower BB')
        elif bbp > 0.9: ss += 0.5; sr.append('Upper BB')

        if r1h['cci'] < -150:  ls += 1.5; lr.append('CCI ↓↓')
        elif r1h['cci'] > 150: ss += 0.5; sr.append('CCI ↑↑')

        if r1h['close'] > r1h['vwap'] * 1.02:
            ss += 1; sr.append('Above VWAP')

        # TREND STRENGTH 4pts
        adx = r1h['adx']
        if adx > 30:
            if r1h['di_plus'] > r1h['di_minus']: ls += 2; lr.append(f'ADX {adx:.0f}↑')
            else:                                 ss += 2; sr.append(f'ADX {adx:.0f}↓')
        elif adx > 25:
            if r1h['di_plus'] > r1h['di_minus']: ls += 1
            else:                                 ss += 1

        ai = r1h['aroon_ind']
        if ai > 50:    ls += 1; lr.append('Aroon ↑')
        elif ai < -50: ss += 1; sr.append('Aroon ↓')

        roc = r1h['roc']
        if roc > 3:    ls += 1; lr.append('ROC+')
        elif roc < -3: ss += 1; sr.append('ROC-')

        # PATTERNS 3.5pts
        if r1h['bull_div']:   ls += 2.5; lr.append('🎯 Bull Div')
        elif r1h['bear_div']: ss += 2;   sr.append('🎯 Bear Div')

        if r15m['bull_engulf']:   ls += 1.5; lr.append('📊 Bull Engulf')
        elif r15m['bear_engulf']: ss += 1.5; sr.append('📊 Bear Engulf')

        # HTF 1pt
        if r4h['close'] > r4h['vwap']: ls += 1; lr.append('4H VWAP ↑')
        else:                           ss += 1; sr.append('4H VWAP ↓')

        return ls, ss, lr, sr, mcb, spk

    # ── Signal Detection ──────────────────────────────────────

    def detect_signal(self, data, symbol):
        try:
            if not data or '1h' not in data:
                return None

            for tf in data:
                data[tf] = self.add_indicators(data[tf])

            df1  = data['1h']; df4 = data['4h']; df15 = data['15m']
            if len(df1) < 50:
                return None

            r1h  = df1.iloc[-1];  p1h  = df1.iloc[-2]
            r4h  = df4.iloc[-1];  r15m = df15.iloc[-1]

            for c in ['ema_9','ema_21','rsi','macd','vwap','bb_pband','atr']:
                if c not in r1h.index or pd.isna(r1h[c]):
                    return None

            spike, vol_ratio = self._vol_spike(df1)
            ls, ss, lr, sr, mcb, spk = self._score(r1h, p1h, r4h, r15m, vol_ratio)

            max_score = 35
            thresh    = max_score * MIN_SCORE_PCT
            signal    = None

            if ls > ss and ls >= thresh:
                signal = 'LONG';  score = ls; reasons = lr
            elif ss > ls and ss >= thresh:
                signal = 'SHORT'; score = ss; reasons = sr
            if not signal:
                return None

            # ── FIX 3: Mandatory conviction filter ──────────────
            # Signal must have MACD cross OR volume spike
            # Trend-only setups (EMA+ST+ADX) are filtered out —
            # they produced most of the BE-close patterns in live
            mcs = r1h['macd'] < r1h['macd_signal'] and p1h['macd'] >= p1h['macd_signal']
            high_conviction = mcb or mcs or spk
            if not high_conviction:
                self.stats['conviction_filtered'] += 1
                logger.info(f"⚡ No conviction: {symbol} {signal} (no MACD cross or vol spike)")
                return None

            # ── HARD regime block ──
            if REGIME_MODE == 'HARD' and self.btc_regime:
                if signal == 'LONG' and self.btc_regime == 'BEAR':
                    self.stats['regime_blocked'] += 1; return None
                if signal == 'SHORT' and self.btc_regime == 'BULL':
                    self.stats['regime_blocked'] += 1; return None

            # ── Long trend filter ──
            if signal == 'LONG' and USE_LONG_TREND_FILTER:
                confirms = [
                    r4h['ema_9'] > r4h['ema_21'],
                    r1h['ema_9'] > r1h['ema_21'],
                    mcb,
                    spk and r1h['close'] > p1h['close'],
                    r1h['rsi'] < 35,
                ]
                if not any(confirms):
                    self.stats['filtered_long'] += 1; return None

            pct     = score / max_score
            quality = 'PREMIUM 💎' if pct >= QUALITY_PREMIUM_PCT else 'GOOD ✅'

            entry = r15m['close']
            atr   = r1h['atr']
            if pd.isna(atr) or atr == 0 or pd.isna(entry) or entry == 0:
                return None

            # ── TP/SL levels ──
            if signal == 'LONG':
                sl    = entry - atr * ATR_SL_MULT
                tp1   = entry + atr * ATR_TP1_MULT
                tp2   = entry + atr * ATR_TP2_MULT
                tp3   = entry + atr * ATR_TP3_MULT
            else:
                sl    = entry + atr * ATR_SL_MULT
                tp1   = entry - atr * ATR_TP1_MULT
                tp2   = entry - atr * ATR_TP2_MULT
                tp3   = entry - atr * ATR_TP3_MULT

            rr       = abs(tp1 - entry) / abs(sl - entry)
            risk_pct = abs((sl  - entry) / entry * 100)
            tp1_pct  = abs((tp1 - entry) / entry * 100)
            tp2_pct  = abs((tp2 - entry) / entry * 100)
            tp3_pct  = abs((tp3 - entry) / entry * 100)
            tid      = f"{symbol.replace('/USDT:USDT','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Conviction tag for message
            conviction_tag = ''
            if mcb or mcs: conviction_tag = '🎯 MACD cross'
            if spk:        conviction_tag += (' + ' if conviction_tag else '') + f'🚀 Vol {vol_ratio:.1f}x'

            if signal == 'LONG':
                close_plan = (
                    f"📋 <b>Close plan (LONG):</b>\n"
                    f"  • TP1 → close <b>{int(LONG_TP1_PCT*100)}%</b> → move SL to BE\n"
                    f"  • TP2 → close <b>{int(LONG_TP2_PCT*100)}%</b>\n"
                    f"  • TP3 → close remaining <b>{int(LONG_TP3_PCT*100)}%</b> (runner)"
                )
            else:
                close_plan = (
                    f"📋 <b>Close plan (SHORT):</b>\n"
                    f"  • TP1 → close <b>{int(SHORT_TP1_PCT*100)}%</b> → move SL to BE\n"
                    f"  • TP2 → close remaining <b>{int(SHORT_TP2_PCT*100)}%</b>"
                )

            return {
                'trade_id':        tid,
                'symbol':          symbol.replace('/USDT:USDT', ''),
                'full_symbol':     symbol,
                'signal':          signal,
                'quality':         quality,
                'score':           score,
                'score_pct':       pct * 100,
                'entry':           entry,
                'stop_loss':       sl,
                'tp1':             tp1, 'tp1_pct': tp1_pct,
                'tp2':             tp2, 'tp2_pct': tp2_pct,
                'tp3':             tp3, 'tp3_pct': tp3_pct,
                'rr':              rr,
                'risk_pct':        risk_pct,
                'conviction_tag':  conviction_tag,
                'close_plan':      close_plan,
                'reasons':         reasons[:10],
                'tp1_hit':         False,
                'tp2_hit':         False,
                'tp3_hit':         False,
                'sl_hit':          False,
                'be_active':       False,
                'timestamp':       datetime.now(),
                'btc_regime':      self.btc_regime or 'N/A',
            }

        except Exception as e:
            logger.error(f"Signal {symbol}: {e}")
            return None

    # ── Format signal message ─────────────────────────────────

    def _fmt_signal(self, sig):
        e  = '🚀' if sig['signal'] == 'LONG' else '🔻'
        re = '🐂' if sig['btc_regime'] == 'BULL' else '🐻'
        pf = int(sig['score_pct'] / 10)

        m  = f"{'─'*40}\n"
        m += f"{e} <b>{sig['signal']} — {sig['quality']}</b>\n"
        m += f"{'─'*40}\n\n"
        m += f"<b>Pair:</b>   #{sig['symbol']}  {re} {sig['btc_regime']}\n"
        m += f"<b>Score:</b>  {sig['score']:.1f}/35  ({sig['score_pct']:.0f}%)\n"
        m += f"{'▰'*pf}{'▱'*(10-pf)}\n"
        m += f"<b>Catalyst:</b> {sig['conviction_tag']}\n\n"
        m += f"<b>Entry:</b>   <code>${sig['entry']:.6f}</code>\n"
        m += f"<b>TP1:</b>     <code>${sig['tp1']:.6f}</code>  +{sig['tp1_pct']:.2f}%\n"
        m += f"<b>TP2:</b>     <code>${sig['tp2']:.6f}</code>  +{sig['tp2_pct']:.2f}%\n"
        m += f"<b>TP3:</b>     <code>${sig['tp3']:.6f}</code>  +{sig['tp3_pct']:.2f}%\n"
        m += f"<b>SL:</b>      <code>${sig['stop_loss']:.6f}</code>  -{sig['risk_pct']:.2f}%\n"
        m += f"<b>RR (TP1):</b> {sig['rr']:.1f}:1\n\n"
        m += f"{sig['close_plan']}\n\n"
        m += f"<b>Why:</b>\n"
        for r in sig['reasons']:
            m += f"  • {r}\n"
        m += f"\n<i>🆔 {sig['trade_id']}</i>\n"
        m += f"<i>⏰ {sig['timestamp'].strftime('%H:%M UTC')} | v9.1</i>"
        return m

    # ── Telegram ──────────────────────────────────────────────

    async def send_msg(self, text):
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Telegram: {e}")

    # ── TP / SL Alerts ────────────────────────────────────────

    async def _tp1_alert(self, trade, price):
        gain = abs((price - trade['entry']) / trade['entry'] * 100)
        if trade['signal'] == 'LONG':
            close_pct   = int(LONG_TP1_PCT * 100)
            next_target = f"${trade['tp2']:.6f} (+{trade['tp2_pct']:.2f}%)"
        else:
            close_pct   = int(SHORT_TP1_PCT * 100)
            next_target = f"${trade['tp2']:.6f} (+{trade['tp2_pct']:.2f}%)"

        m  = f"✅ <b>TP1 HIT</b> ✅\n\n"
        m += f"<b>{trade['symbol']}</b> {trade['signal']}\n"
        m += f"Entry:  ${trade['entry']:.6f}\n"
        m += f"TP1:    ${price:.6f}  <b>+{gain:.2f}%</b>\n\n"
        m += f"✂️ Close <b>{close_pct}%</b> of position\n"
        m += f"🔒 Move SL → breakeven (${trade['entry']:.6f})\n"
        m += f"🎯 Next target: {next_target}\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp1_hit']   = True
        trade['be_active'] = True
        self.stats['tp1_hits'] += 1

    async def _tp2_alert(self, trade, price):
        gain = abs((price - trade['entry']) / trade['entry'] * 100)
        if trade['signal'] == 'LONG':
            close_pct  = int(LONG_TP2_PCT * 100)
            has_runner = True
        else:
            close_pct  = int(SHORT_TP2_PCT * 100)
            has_runner = False

        m  = f"💰 <b>TP2 HIT</b> 💰\n\n"
        m += f"<b>{trade['symbol']}</b> {trade['signal']}\n"
        m += f"Entry:  ${trade['entry']:.6f}\n"
        m += f"TP2:    ${price:.6f}  <b>+{gain:.2f}%</b>\n\n"
        m += f"✂️ Close <b>{close_pct}%</b> of position\n"
        if has_runner:
            m += f"🏃 <b>{int(LONG_TP3_PCT*100)}% runner</b> still open → TP3 at ${trade['tp3']:.6f} (+{trade['tp3_pct']:.2f}%)\n"
            m += f"🔒 SL still at breakeven\n"
        else:
            m += f"✅ <b>Trade complete — full position closed</b>\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp2_hit'] = True
        self.stats['tp2_hits'] += 1

    async def _tp3_alert(self, trade, price):
        gain = abs((price - trade['entry']) / trade['entry'] * 100)
        m  = f"🔥 <b>TP3 HIT — FULL RUNNER!</b> 🔥\n\n"
        m += f"<b>{trade['symbol']}</b> {trade['signal']}\n"
        m += f"Entry:  ${trade['entry']:.6f}\n"
        m += f"TP3:    ${price:.6f}  <b>+{gain:.2f}%</b>\n\n"
        m += f"✅ <b>Close remaining {int(LONG_TP3_PCT*100)}% — trade complete</b>\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp3_hit'] = True
        self.stats['tp3_hits'] += 1

    async def _sl_alert(self, trade, price, be_save=False):
        if be_save:
            m  = f"🔒 <b>BREAKEVEN CLOSE</b>\n\n"
            m += f"<b>{trade['symbol']}</b> {trade['signal']}\n"
            m += f"TP1 was hit ✅ — SL moved to entry\n"
            m += f"Closed remainder at breakeven — <b>no loss</b>\n"
            m += f"\n<i>{trade['trade_id']}</i>"
            self.stats['be_saves'] += 1
        else:
            loss = abs((price - trade['entry']) / trade['entry'] * 100)
            m  = f"⛔ <b>STOP LOSS</b>\n\n"
            m += f"<b>{trade['symbol']}</b> {trade['signal']}\n"
            m += f"Entry: ${trade['entry']:.6f}\n"
            m += f"SL:    ${price:.6f}  <b>-{loss:.2f}%</b>\n\n"
            m += f"<i>Next signal incoming 🎯</i>"
            self.stats['sl_hits'] += 1
        await self.send_msg(m)

    # ── Trade Tracker ─────────────────────────────────────────

    async def track_trades(self):
        logger.info("📡 Tracker started")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30); continue

                done = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        # Timeout
                        if datetime.now() - t['timestamp'] > timedelta(hours=MAX_TRADE_HOURS):
                            logger.info(f"⏰ Timeout: {t['symbol']}")
                            self.stats['timeouts'] += 1
                            done.append(tid); continue

                        ticker    = await self.exchange.fetch_ticker(t['full_symbol'])
                        price     = ticker['last']
                        direction = t['signal']

                        # Active SL: entry price if BE is on, else original SL
                        active_sl = t['entry'] if t['be_active'] else t['stop_loss']

                        if direction == 'LONG':
                            # TP3 — runner (only after TP2)
                            if not t['tp3_hit'] and t['tp2_hit'] and price >= t['tp3']:
                                await self._tp3_alert(t, price)
                                done.append(tid); continue

                            # TP2 — (only after TP1)
                            if not t['tp2_hit'] and t['tp1_hit'] and price >= t['tp2']:
                                await self._tp2_alert(t, price)
                                # Don't close — runner still waiting for TP3

                            # TP1
                            if not t['tp1_hit'] and price >= t['tp1']:
                                await self._tp1_alert(t, price)

                            # SL (uses BE price if active)
                            if price <= active_sl:
                                be_save = t['be_active'] and active_sl == t['entry']
                                await self._sl_alert(t, price, be_save=be_save)
                                done.append(tid)

                        else:  # SHORT
                            # TP2 — close fully (no TP3 runner for shorts)
                            if not t['tp2_hit'] and t['tp1_hit'] and price <= t['tp2']:
                                await self._tp2_alert(t, price)
                                done.append(tid); continue

                            # TP1
                            if not t['tp1_hit'] and price <= t['tp1']:
                                await self._tp1_alert(t, price)

                            # SL
                            if price >= active_sl:
                                be_save = t['be_active'] and active_sl == t['entry']
                                await self._sl_alert(t, price, be_save=be_save)
                                done.append(tid)

                    except Exception as e:
                        logger.error(f"Track {tid}: {e}")

                for tid in done:
                    self.active_trades.pop(tid, None)

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Tracker: {e}")
                await asyncio.sleep(60)

    # ── Scanner ───────────────────────────────────────────────

    async def scan_all(self):
        if self.is_scanning: return []
        self.is_scanning = True
        signals = []

        await self.update_btc_regime()
        pairs = await self.get_all_usdt_pairs()

        for pair in pairs:
            try:
                data = await self.fetch_data(pair)
                if data:
                    sig = self.detect_signal(data, pair)
                    if sig:
                        ck   = f"{sig['symbol']}_{sig['signal']}"
                        last = self.pair_cooldown.get(ck)
                        if last and (datetime.now() - last).total_seconds() < 4 * 3600:
                            logger.info(f"⏳ Cooldown: {sig['symbol']} {sig['signal']}")
                            await asyncio.sleep(0.4); continue

                        self.pair_cooldown[ck] = datetime.now()
                        signals.append(sig)
                        self.active_trades[sig['trade_id']] = sig
                        self.signal_history.append(sig)
                        self.stats['total_signals'] += 1
                        if sig['signal'] == 'LONG': self.stats['long_signals'] += 1
                        else:                       self.stats['short_signals'] += 1
                        if 'PREMIUM' in sig['quality']: self.stats['premium_signals'] += 1
                        else:                           self.stats['good_signals'] += 1
                        await self.send_msg(self._fmt_signal(sig))
                        await asyncio.sleep(1)
                await asyncio.sleep(0.4)
            except Exception as e:
                logger.error(f"❌ {pair}: {e}")

        self.stats['last_scan']     = datetime.now()
        self.stats['pairs_scanned'] = len(pairs)
        logger.info(
            f"✅ Scan done — {len(signals)} signals | BTC:{self.btc_regime} | "
            f"blocked:{self.stats['regime_blocked']} "
            f"conviction_filtered:{self.stats['conviction_filtered']} "
            f"tracking:{len(self.active_trades)}"
        )
        self.is_scanning = False
        return signals

    # ── Daily report ──────────────────────────────────────────

    async def send_daily_report(self):
        while True:
            await asyncio.sleep(24 * 3600)
            try:
                s   = self.stats
                tp1 = s['tp1_hits']; tp2 = s['tp2_hits']; tp3 = s['tp3_hits']
                sl  = s['sl_hits'];  be  = s['be_saves']
                tot = tp1 + sl
                wr  = round(tp1 / tot * 100, 1) if tot > 0 else 0
                hrs = round((datetime.now() - s['session_start']).total_seconds() / 3600, 1)

                cutoff    = datetime.now() - timedelta(hours=24)
                day_sigs  = [t for t in self.signal_history if t['timestamp'] >= cutoff]
                day_long  = sum(1 for t in day_sigs if t['signal'] == 'LONG')
                day_short = sum(1 for t in day_sigs if t['signal'] == 'SHORT')

                re = '🐂' if self.btc_regime == 'BULL' else '🐻'

                m  = f"{'─'*38}\n📅 <b>24H DAILY REPORT — v9.1</b>\n{'─'*38}\n\n"
                m += f"{re} BTC: <b>{self.btc_regime}</b>  |  Session: {hrs}h\n\n"
                m += f"<b>── Today's Signals ──</b>\n"
                m += f"  Total: <b>{len(day_sigs)}</b>  ({day_long}L / {day_short}S)\n\n"
                m += f"<b>── TP Performance ──</b>\n"
                m += f"  ✅ TP1 hits: <b>{tp1}</b>\n"
                m += f"  💰 TP2 hits: <b>{tp2}</b>  ({round(tp2/max(tp1,1)*100)}% of TP1s)\n"
                m += f"  🔥 TP3 hits: <b>{tp3}</b>  ({round(tp3/max(tp1,1)*100)}% of TP1s)\n"
                m += f"  🔒 BE saves: <b>{be}</b>\n"
                m += f"  ❌ SL hits:  <b>{sl}</b>\n\n"

                bar_filled = int(wr / 10)
                bar = '▰' * bar_filled + '▱' * (10 - bar_filled)
                m += f"<b>TP1 Win Rate: {wr}%</b>\n{bar}\n\n"

                if wr >= 92:   status = "🔥 Excellent"
                elif wr >= 85: status = "✅ Good — within range"
                elif wr >= 78: status = "⚠️ Watch closely"
                else:          status = "🚨 Below target — raise MIN_SCORE to 0.60"

                m += f"{status}\n\n"
                m += f"  Conviction filtered: {s['conviction_filtered']}\n"
                m += f"  Regime blocked: {s['regime_blocked']}\n"
                m += f"  Tracking now: {len(self.active_trades)} trades\n"
                m += f"<i>⏰ {datetime.now().strftime('%d %b %Y %H:%M UTC')}</i>"

                await self.send_msg(m)
                logger.info(f"📅 Daily report | WR:{wr}% | TP1:{tp1} TP2:{tp2} TP3:{tp3} SL:{sl} BE:{be}")

            except Exception as e:
                logger.error(f"Daily report: {e}")

    # ── Run ───────────────────────────────────────────────────

    async def run(self):
        logger.info(
            f"🚀 v9.1 started | "
            f"MIN_SCORE={MIN_SCORE_PCT} | "
            f"TP1={ATR_TP1_MULT}x TP2={ATR_TP2_MULT}x TP3={ATR_TP3_MULT}x SL={ATR_SL_MULT}x | "
            f"Conviction filter ON"
        )
        asyncio.create_task(self.track_trades())
        asyncio.create_task(self.send_daily_report())
        while True:
            try:
                await self.scan_all()
                await asyncio.sleep(SCAN_INTERVAL_MIN * 60)
            except Exception as e:
                logger.error(f"Run: {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ─────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────

class BotCommands:
    def __init__(self, scanner: AdvancedDayTradingScanner):
        self.s = scanner

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🚀 <b>Day Trading Scanner v9.1</b>\n"
            "MACD/Vol conviction filter | TP1/TP2/TP3 ladder\n\n"
            "/scan /stats /trades /regime /help",
            parse_mode=ParseMode.HTML
        )

    async def cmd_scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await update.message.reply_text("⚠️ Scan already running!")
            return
        await update.message.reply_text("🔍 Scanning...")
        asyncio.create_task(self.s.scan_all())

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s   = self.s.stats
        tp1 = s['tp1_hits']; tp2 = s['tp2_hits']; tp3 = s['tp3_hits']
        sl  = s['sl_hits'];  be  = s['be_saves']
        tot = tp1 + sl
        wr  = round(tp1/tot*100, 1) if tot > 0 else 0
        hrs = round((datetime.now()-s['session_start']).total_seconds()/3600, 1)
        spd = round(s['total_signals']/max(hrs,0.1), 1)

        m  = f"📊 <b>STATS — v9.1</b>\n\nSession: {hrs}h\n"
        re = '🐂' if self.s.btc_regime == 'BULL' else '🐻'
        m += f"BTC: {re} {self.s.btc_regime}\n\n"
        m += f"<b>Signals:</b> {s['total_signals']} ({spd}/h)\n"
        m += f"  🟢 Long:    {s['long_signals']}\n"
        m += f"  🔴 Short:   {s['short_signals']}\n"
        m += f"  💎 Premium: {s['premium_signals']}\n\n"
        m += f"<b>Filters:</b>\n"
        m += f"  ⚡ Conviction filtered: {s['conviction_filtered']}\n"
        m += f"  🚫 Regime blocked:      {s['regime_blocked']}\n"
        m += f"  📉 Long filtered:       {s['filtered_long']}\n\n"
        m += f"<b>TP Performance:</b>\n"
        m += f"  ✅ TP1: {tp1}  ({wr}% rate)\n"
        m += f"  💰 TP2: {tp2}  ({round(tp2/max(tp1,1)*100)}% of TP1s extended)\n"
        m += f"  🔥 TP3: {tp3}  ({round(tp3/max(tp1,1)*100)}% of TP1s ran full)\n"
        m += f"  🔒 BE saves: {be}\n"
        m += f"  ❌ SL: {sl}\n\n"
        m += f"Tracking: {len(self.s.active_trades)} trades"
        if s['last_scan']:
            m += f"\nLast scan: {s['last_scan'].strftime('%H:%M')}"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        trades = self.s.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades.")
            return
        m = f"📡 <b>ACTIVE ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:10]:
            age   = int((datetime.now()-t['timestamp']).total_seconds()/3600)
            tp1_s = '✅' if t['tp1_hit'] else '⏳'
            tp2_s = '✅' if t['tp2_hit'] else '⏳'
            tp3_s = '✅' if t.get('tp3_hit') else '⏳'
            be_s  = ' 🔒BE' if t['be_active'] else ''
            m += f"<b>{t['symbol']}</b> {t['signal']} {t['quality']}{be_s}\n"
            m += f"  Entry: ${t['entry']:.6f} | {age}h old\n"
            m += f"  TP1:{tp1_s} TP2:{tp2_s} TP3:{tp3_s}\n"
            m += f"  Score: {t['score_pct']:.0f}% | {t['conviction_tag']}\n\n"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_regime(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        r  = self.s.btc_regime or 'Unknown'
        e  = '🐂' if r == 'BULL' else '🐻'
        m  = f"{e} <b>BTC Regime: {r}</b>\n\n"
        if self.s.btc_price and self.s.btc_ema:
            m += f"Price: ${self.s.btc_price:,.2f}\n"
            m += f"EMA21: ${self.s.btc_ema:,.2f}\n\n"
        m += f"{'✅ LONGs active\n🚫 SHORTs BLOCKED' if r == 'BULL' else '✅ SHORTs active\n🚫 LONGs BLOCKED'}\n"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        m  = "📚 <b>SCANNER v9.1</b>\n\n"
        m += "<b>Filters (v9.1 new):</b>\n"
        m += "  ⚡ Conviction required — MACD cross OR vol spike\n"
        m += "  📊 Score ≥ 55% (was 47%)\n\n"
        m += "<b>TP Levels:</b>\n"
        m += f"  TP1 = {ATR_TP1_MULT}x ATR\n"
        m += f"  TP2 = {ATR_TP2_MULT}x ATR\n"
        m += f"  TP3 = {ATR_TP3_MULT}x ATR\n"
        m += f"  SL  = {ATR_SL_MULT}x ATR  |  RR ≈ 0.67:1\n\n"
        m += "<b>Close Strategy:</b>\n"
        m += "  LONG:  33% TP1 → 33% TP2 → 34% TP3\n"
        m += "  SHORT: 70% TP1 → 30% TP2 (no TP3 runner)\n"
        m += "  SL moves to BE after TP1 fills\n\n"
        m += "<b>Position sizing:</b>\n"
        m += "  💎 PREMIUM: 3-5% of portfolio\n"
        m += "  ✅ GOOD:    1-2% of portfolio\n\n"
        m += "/scan /stats /trades /regime /help"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

async def main():
    # ════════════════════════════════════
    TELEGRAM_TOKEN   = "YOUR_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None
    # ════════════════════════════════════

    scanner = AdvancedDayTradingScanner(
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        binance_api_key=BINANCE_API_KEY,
        binance_secret=BINANCE_SECRET,
    )

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = BotCommands(scanner)

    for cmd, fn in [
        ('start',  cmds.cmd_start),
        ('scan',   cmds.cmd_scan),
        ('stats',  cmds.cmd_stats),
        ('trades', cmds.cmd_trades),
        ('regime', cmds.cmd_regime),
        ('help',   cmds.cmd_help),
    ]:
        app.add_handler(CommandHandler(cmd, fn))

    await app.initialize()
    await app.start()
    logger.info("🤖 v9.1 online")

    try:
        await scanner.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
