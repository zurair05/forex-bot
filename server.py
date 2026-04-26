"""
SMC Signal Dashboard
====================
Run:   python server.py
Open:  http://localhost:5000

Install once:
    pip install flask flask-cors yfinance pandas numpy
"""

import os, time, logging, webbrowser, threading, random
from datetime import datetime, timezone
from typing import Optional

try:
    from flask import Flask, jsonify, request, Response
    from flask_cors import CORS
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"\n[ERROR] Missing package: {e}")
    print("Run:  pip install flask flask-cors yfinance pandas numpy\n")
    exit(1)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("SMC")

app  = Flask(__name__)
CORS(app)

# ── Pairs ─────────────────────────────────────────────────────────────
PAIRS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X", "GBPJPY": "GBPJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
    "XAUUSD": "GC=F",
}
PIP = {
    "EURUSD":0.0001,"GBPUSD":0.0001,"AUDUSD":0.0001,"USDCAD":0.0001,
    "USDJPY":0.01,  "GBPJPY":0.01,  "XAUUSD":0.1,
}
DIGITS = {
    "EURUSD":5,"GBPUSD":5,"AUDUSD":5,"USDCAD":5,
    "USDJPY":3,"GBPJPY":3,"XAUUSD":2,
}

# ── Cache ─────────────────────────────────────────────────────────────
_bar_cache:    dict = {}
_signal_cache: dict = {}
_price_cache:  dict = {}
_price_ts:     float = 0.0
BAR_TTL    = 300
SIGNAL_TTL = 900
PRICE_TTL  = 60

# ── Trade lifecycle & performance tracking ────────────────────────────
_trade_state:    dict  = {}
_signal_history: list  = []   # all completed signals (kept forever in memory)
_session_stats:  dict  = {    # rolling session stats
    "total": 0, "wins": 0, "losses": 0,
    "net_pips": 0.0, "consecutive_losses": 0,
    "daily_pnl_pips": 0.0, "daily_reset_date": "",
    "best_trade_pips": 0.0, "worst_trade_pips": 0.0,
    "total_pips_won": 0.0, "total_pips_lost": 0.0,
}
_daily_drawdown_pips: float = 0.0   # accumulated loss pips today
_session_start:       str   = datetime.now(timezone.utc).isoformat()

# ── Risk limits (pause signal generation if breached) ─────────────────
MAX_CONSECUTIVE_LOSSES = 3      # pause after 3 losses in a row
MAX_DAILY_LOSS_PIPS    = 90     # pause after 90 pips loss in one day (3× SL)
MAX_SIGNALS_PER_PAIR_PER_DAY = 2  # don't over-trade same pair

# ── Market hours ──────────────────────────────────────────────────────
def market_status():
    now = datetime.now(timezone.utc)
    wd, h = now.weekday(), now.hour
    closed, reason = False, ""
    if wd == 4 and h >= 21: closed=True; reason="Weekend — Friday close 21:00 UTC"
    elif wd == 5:            closed=True; reason="Weekend — Saturday"
    elif wd == 6 and h < 21: closed=True; reason="Weekend — Sunday (opens 21:00 UTC)"
    if (now.month, now.day) in [(12,25),(1,1)]: closed=True; reason="Public holiday"
    sessions = []
    for name,(o,c) in [("Sydney",(21,6)),("Tokyo",(0,9)),("London",(7,16)),("New York",(12,21))]:
        if o < c:
            if o <= h < c: sessions.append(name)
        else:
            if h >= o or h < c: sessions.append(name)
    hrs = ""
    if closed:
        if wd==6: hrs=f"Opens in {21-h}h"
        elif wd==5: hrs="Opens Sunday 21:00 UTC"
        elif wd==4: hrs="Opens Sunday 21:00 UTC"
    return {"is_open": not closed, "is_closed": closed, "reason": reason,
            "sessions": sessions, "next_open": hrs,
            "time_utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "weekday": now.strftime("%A")}

# ── Data fetching ─────────────────────────────────────────────────────
def fetch_bars(pair, tf="H1"):
    key = f"{pair}_{tf}"
    if key in _bar_cache and time.time()-_bar_cache[key]["ts"] < BAR_TTL:
        return _bar_cache[key]["data"]
    ticker   = PAIRS.get(pair)
    interval = {"M15":"15m","H1":"1h","H4":"4h","D1":"1d"}.get(tf,"1h")
    period   = {"M15":"5d", "H1":"30d","H4":"60d","D1":"1y"}.get(tf,"30d")
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        bars = [{"time":int(pd.Timestamp(ts).timestamp()),
                 "open":float(r["Open"]),"high":float(r["High"]),
                 "low":float(r["Low"]),"close":float(r["Close"]),
                 "volume":int(r.get("Volume",0))}
                for ts,r in df.iterrows()]
        _bar_cache[key] = {"data":bars,"ts":time.time()}
        return bars
    except Exception as e:
        log.warning(f"Fetch {pair} {tf}: {e}")
        return None

def fetch_prices():
    global _price_cache, _price_ts
    if _price_cache and time.time()-_price_ts < PRICE_TTL:
        return _price_cache
    out = {}
    for pair, ticker in PAIRS.items():
        try:
            df = yf.download(ticker, period="2d", interval="1h",
                             progress=False, auto_adjust=True)
            if df is None or df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if df.empty: continue
            last = df.iloc[-1]; prev = df.iloc[-2] if len(df)>1 else last
            close_price = round(float(last["Close"]),5)
            prev_close  = round(float(prev["Close"]),5)
            day_high    = round(float(last["High"]),5)
            day_low     = round(float(last["Low"]),5)
            out[pair] = {
                "price":      close_price,
                "day_high":   day_high,
                "day_low":    day_low,
                "prev_close": prev_close,
                "change_pct": round((close_price-prev_close)/prev_close*100,3) if prev_close!=0 else 0,
                "change_abs": round(close_price-prev_close, 5),
            }
        except: pass
    if out: _price_cache=out; _price_ts=time.time()
    return out

def demo_bars(pair, n=120):
    prices = fetch_prices()
    base   = prices.get(pair,{}).get("price",0)
    fb     = {"EURUSD":1.0855,"GBPUSD":1.2685,"USDJPY":149.82,
              "GBPJPY":192.30,"AUDUSD":0.6512,"USDCAD":1.3810,"XAUUSD":2321.0}
    if base==0: base = fb.get(pair,1.0)
    vol    = {"EURUSD":0.0008,"GBPUSD":0.0012,"USDJPY":0.12,
              "GBPJPY":0.18,"AUDUSD":0.0007,"USDCAD":0.0009,"XAUUSD":2.8}.get(pair,0.001)
    today  = datetime.now(timezone.utc).strftime("%Y%m%d")
    seed   = int(today) + sum(ord(c)*(i+1) for i,c in enumerate(pair))
    rng    = random.Random(seed)
    bars   = []; price = base*(1+rng.uniform(-0.002,0.002))
    for i in range(n):
        o=price; c=o+(base-o)*0.04+rng.gauss(0,vol)
        bars.append({"time":int(time.time())-(n-i)*3600,
                     "open":round(o,5),"high":round(max(o,c)+abs(rng.gauss(0,vol*.4)),5),
                     "low":round(min(o,c)-abs(rng.gauss(0,vol*.4)),5),
                     "close":round(c,5),"volume":rng.randint(500,5000)})
        price=c
    if bars: bars[-1]["close"]=base; bars[-1]["high"]=max(bars[-1]["high"],base); bars[-1]["low"]=min(bars[-1]["low"],base)
    return bars

# ── SMC Engine ────────────────────────────────────────────────────────
def detect_swings(bars, n=5):
    swings=[]; psh=psL=None
    for i in range(n,len(bars)-n):
        h=bars[i]["high"]; l=bars[i]["low"]
        if all(bars[j]["high"]<h for j in range(i-n,i+n+1) if j!=i):
            swings.append({"idx":i,"price":h,"kind":"HH" if(psh is None or h>psh) else "LH","hi":True}); psh=h
        if all(bars[j]["low"]>l  for j in range(i-n,i+n+1) if j!=i):
            swings.append({"idx":i,"price":l,"kind":"HL" if(psL is None or l>psL) else "LL","hi":False}); psL=l
    return sorted(swings,key=lambda s:s["idx"])

def get_trend(swings):
    if len(swings)<4: return "ranging"
    r=swings[-8:]; bull=sum(1 for s in r if s["kind"] in("HH","HL")); bear=len(r)-bull
    return "bullish" if bull>bear+1 else "bearish" if bear>bull+1 else "ranging"

def get_structure(swings,trend):
    choch=bos=False; cd=bd=""
    for s in swings[-6:]:
        k=s["kind"]
        if k=="HH" and trend=="bearish": choch=True;cd="bullish"
        if k=="LL" and trend=="bullish": choch=True;cd="bearish"
        if k=="HH" and trend=="bullish": bos=True;  bd="bullish"
        if k=="LL" and trend=="bearish": bos=True;  bd="bearish"
    return choch,cd,bos,bd

def detect_obs(bars, min_body=0.55, lb=50):
    obs=[]; lim=min(lb,len(bars)-4)
    for i in range(2,lim):
        b=bars[i]; body=abs(b["close"]-b["open"]); rng=b["high"]-b["low"]
        if rng==0 or body/rng<min_body: continue
        after=bars[max(i-3,0):i]
        if b["close"]<b["open"]:
            mx=max((x["close"] for x in after),default=0)
            if rng>0 and (mx-b["close"])/rng>=1.5:
                obs.append({"hi":b["high"],"lo":b["low"],"mid":(b["high"]+b["low"])/2,"bull":True,"tapped":False})
        elif b["close"]>b["open"]:
            mn=min((x["close"] for x in after),default=float("inf"))
            if rng>0 and (b["close"]-mn)/rng>=1.5:
                obs.append({"hi":b["high"],"lo":b["low"],"mid":(b["high"]+b["low"])/2,"bull":False,"tapped":False})
    return obs[-8:]

def detect_fvgs(bars, pip=0.0001):
    fvgs=[]; lim=min(60,len(bars)-2)
    for i in range(2,lim):
        if bars[i-2]["low"]>bars[i]["high"] and (bars[i-2]["low"]-bars[i]["high"])/pip>=2:
            fvgs.append({"hi":bars[i-2]["low"],"lo":bars[i]["high"],"mid":(bars[i-2]["low"]+bars[i]["high"])/2,"bull":True})
        if bars[i-2]["high"]<bars[i]["low"] and (bars[i]["low"]-bars[i-2]["high"])/pip>=2:
            fvgs.append({"hi":bars[i]["low"],"lo":bars[i-2]["high"],"mid":(bars[i]["low"]+bars[i-2]["high"])/2,"bull":False})
    return fvgs[-8:]



# ── ADR (Average Daily Range) ──────────────────────────────────────────
def calc_adr(bars, days=14):
    from collections import defaultdict
    if not bars or len(bars) < 2: return 0, 0, 0
    daily = defaultdict(lambda: {"hi": 0, "lo": float("inf")})
    for b in bars:
        d = datetime.fromtimestamp(b["time"], tz=timezone.utc).strftime("%Y-%m-%d")
        daily[d]["hi"] = max(daily[d]["hi"], b["high"])
        daily[d]["lo"] = min(daily[d]["lo"], b["low"])
    day_ranges = [v["hi"]-v["lo"] for v in daily.values() if v["hi"] > 0]
    if len(day_ranges) < 2: return 0, 0, 0
    adr = sum(day_ranges[-days:]) / min(len(day_ranges), days)
    today_range = day_ranges[-1] if day_ranges else 0
    pct = round(today_range / adr * 100) if adr > 0 else 0
    return round(adr, 5), round(today_range, 5), pct

def grade_signal(score_pct, conf_count, in_killzone, adr_pct):
    if score_pct > 85 and conf_count >= 4 and in_killzone and adr_pct < 70: return "A"
    if score_pct > 85 and conf_count >= 4: return "A"
    if score_pct > 75 and conf_count >= 3: return "B"
    return "C"

_daily_trade_count = {}

def can_trade_today(pair):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _daily_trade_count.get(f"{pair}_{today}", 0) < MAX_SIGNALS_PER_PAIR_PER_DAY

def record_daily_trade(pair):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{pair}_{today}"
    _daily_trade_count[key] = _daily_trade_count.get(key, 0) + 1

def is_trading_paused():
    if _session_stats["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        return True, f"{MAX_CONSECUTIVE_LOSSES} consecutive losses — paused. Reset manually."
    if _daily_drawdown_pips <= -MAX_DAILY_LOSS_PIPS:
        return True, f"Daily loss limit hit ({_daily_drawdown_pips:.1f}p) — paused."
    return False, ""

# ── ICT Killzone Filter ────────────────────────────────────────────────
def get_killzone():
    """
    Returns current ICT killzone name or None.
    Killzones (UTC):
      London     07:00–10:00  — highest probability, Judas Swing, daily direction
      New York   12:00–15:00  — continuation or reversal of London move
      Asian      22:00–00:00  — range building, watch for NY/London sweep
    """
    h = datetime.now(timezone.utc).hour
    m = datetime.now(timezone.utc).minute
    t = h * 60 + m
    if 420 <= t < 600:   return "London"      # 07:00–10:00
    if 720 <= t < 900:   return "New York"    # 12:00–15:00
    if 1320 <= t < 1440: return "Asian"       # 22:00–24:00
    if 0 <= t < 60:      return "Asian"       # 00:00–01:00
    return None


def is_killzone():
    return get_killzone() is not None


# ── Strategy 4: Asian Session Range ────────────────────────────────────
def get_asian_range(bars):
    """
    Marks the Asian session high/low from the last completed Asian session.
    Asian session = 22:00–07:00 UTC (9 hours).
    Returns {"hi": float, "lo": float, "range_pips": float} or None.
    """
    if not bars or len(bars) < 10:
        return None

    now_ts   = datetime.now(timezone.utc)
    # Find bars from last Asian session (prev day 22:00 to today 07:00)
    asian_bars = []
    for b in bars:
        bt = datetime.fromtimestamp(b["time"], tz=timezone.utc)
        h  = bt.hour
        if h >= 22 or h < 7:  # 22:00–07:00 UTC
            asian_bars.append(b)

    # Only use bars from last 24 hours
    cutoff = now_ts.timestamp() - 86400
    asian_bars = [b for b in asian_bars if b["time"] > cutoff]

    if len(asian_bars) < 3:
        return None

    hi = max(b["high"]  for b in asian_bars)
    lo = min(b["low"]   for b in asian_bars)
    return {"hi": hi, "lo": lo, "mid": (hi + lo) / 2}


# ── Strategy 5: Breaker Blocks ─────────────────────────────────────────
def detect_breakers(bars, pip=0.0001):
    """
    Breaker Block = Failed Order Block.
    A bullish OB that price broke through (bearish close through the OB bottom)
    then came back to retest from the other side = bearish breaker.
    And vice versa.
    Returns list of breaker blocks with direction.
    """
    breakers = []
    lim      = min(80, len(bars) - 6)
    obs      = detect_obs(bars, lb=lim)

    for ob in obs:
        # Check if price broke through this OB after it formed
        lo, hi = ob["lo"], ob["hi"]
        price  = bars[0]["close"]  # most recent (bars are reversed)

        if ob["bull"]:
            # Bull OB broken to downside = bearish breaker
            if price < lo - pip * 3:
                breakers.append({
                    "type": "Bear Breaker",
                    "hi": hi, "lo": lo,
                    "mid": (hi + lo) / 2,
                    "bull": False,
                    "color": "red",
                })
        else:
            # Bear OB broken to upside = bullish breaker
            if price > hi + pip * 3:
                breakers.append({
                    "type": "Bull Breaker",
                    "hi": hi, "lo": lo,
                    "mid": (hi + lo) / 2,
                    "bull": True,
                    "color": "green",
                })

    return breakers[-4:]


# ── Strategy 6: Judas Swing Detection ─────────────────────────────────
def detect_judas_swing(bars, asian_range, pip=0.0001):
    """
    Judas Swing = London open manipulative sweep of Asian range
    that then reverses and establishes the day's true direction.
    Detects if the most recent bars swept Asian H/L then reversed.
    Returns: "bull_reversal", "bear_reversal", or None
    """
    if not asian_range or len(bars) < 5:
        return None

    kz = get_killzone()
    if kz not in ("London", "New York"):
        return None

    recent = bars[:6]  # most recent 6 bars
    lows   = [b["low"]   for b in recent]
    highs  = [b["high"]  for b in recent]
    closes = [b["close"] for b in recent]

    swept_low  = min(lows)  < asian_range["lo"] - pip
    swept_high = max(highs) > asian_range["hi"] + pip

    # After sweeping low, closed back above — bull reversal
    if swept_low and closes[0] > asian_range["lo"]:
        return "bull_reversal"

    # After sweeping high, closed back below — bear reversal
    if swept_high and closes[0] < asian_range["hi"]:
        return "bear_reversal"

    return None


# ── MACD ──────────────────────────────────────────────────────────────
def analyse(bars, pair):
    if not bars or len(bars)<50: return {}
    p=PIP.get(pair,0.0001); rev=list(reversed(bars)); price=rev[0]["close"]
    lb=min(100,len(rev)); rhi=max(b["high"] for b in rev[:lb]); rlo=min(b["low"] for b in rev[:lb]); mid=(rhi+rlo)/2

    # Classic SMC
    sw=detect_swings(rev); trend=get_trend(sw); choch,cd,bos,bd=get_structure(sw,trend)
    obs=detect_obs(rev)
    for ob in obs:
        if ob["lo"]<=price<=ob["hi"]: ob["tapped"]=True
    fvgs=detect_fvgs(rev,p)
    near_fvgs=[f for f in fvgs if abs(f["mid"]-price)/p<100]
    liq=[{"price":s["price"],"type":"BSL" if s["hi"] else "SSL"} for s in sw[-12:]]

    # ICT: breakers, Asian range, Judas swing, killzone
    breakers    = detect_breakers(rev, p)
    asian_range = get_asian_range(bars)
    judas       = detect_judas_swing(rev, asian_range, p)
    kz          = get_killzone()

    # Asian range position
    near_asian_hi = asian_range and abs(price - asian_range["hi"]) / p < 20
    near_asian_lo = asian_range and abs(price - asian_range["lo"]) / p < 20
    above_asian   = asian_range and price > asian_range["hi"]
    below_asian   = asian_range and price < asian_range["lo"]

    return {"price":price,"trend":trend,"mid":mid,
            "disc":price<mid,"prem":price>mid,
            "choch":choch,"cd":cd,"bos":bos,"bd":bd,
            "obs":obs,"tapped":[o for o in obs if o["tapped"]],
            "fvgs":fvgs,"near":near_fvgs,"liq":liq,
            # Session / ICT
            "breakers":breakers,
            "asian_range":asian_range,
            "near_asian_hi":near_asian_hi,"near_asian_lo":near_asian_lo,
            "above_asian":above_asian,"below_asian":below_asian,
            "judas":judas,"killzone":kz,
            "ob_count":len(obs),"fvg_count":len(fvgs),
            "swing_count":len(sw)}

def score(buy, htf, mtf, ltf):
    """
    SMC + ICT confluence scorer.
    Pure Smart Money Concepts (HTF trend, OB, FVG, CHoCH, BOS, Liquidity,
    Premium/Discount) layered with the four ICT concepts that go with it
    (Killzones, Asian range, Judas swing, Breaker blocks).
    Signal only fires if total > 70%.
    """
    s=0.0; c=[]

    # ── Classic SMC ───────────────────────────────────────────────────
    if buy  and htf.get("trend")=="bullish": s+=0.20;c.append("HTF↑")
    if not buy and htf.get("trend")=="bearish": s+=0.20;c.append("HTF↓")
    if any(o["bull"]==buy for o in mtf.get("tapped",[])): s+=0.20;c.append("OB_tap")
    if any(f["bull"]==buy for f in mtf.get("near",[])): s+=0.15;c.append("FVG")
    if mtf.get("choch") and mtf["cd"]==("bullish" if buy else "bearish"): s+=0.20;c.append("CHoCH")
    elif ltf.get("choch") and ltf["cd"]==("bullish" if buy else "bearish"): s+=0.14;c.append("LTF_CHoCH")
    if mtf.get("bos") and mtf["bd"]==("bullish" if buy else "bearish"): s+=0.10;c.append("BOS")
    if any(l["type"]==("SSL" if buy else "BSL") for l in mtf.get("liq",[])): s+=0.10;c.append("Liq_sweep")
    if buy  and mtf.get("disc"): s+=0.10;c.append("Discount")
    if not buy and mtf.get("prem"):  s+=0.10;c.append("Premium")
    if ltf.get("choch") and ltf["cd"]==("bullish" if buy else "bearish"): s+=0.10;c.append("LTF↑" if buy else "LTF↓")

    # ── ICT Killzone ──────────────────────────────────────────────────
    kz = mtf.get("killzone")
    if kz == "London":   s+=0.12;c.append("London_KZ")
    elif kz == "New York": s+=0.10;c.append("NY_KZ")
    elif kz is None:     s-=0.05  # outside killzone — silent penalty

    # ── Asian Range ───────────────────────────────────────────────────
    if buy  and mtf.get("near_asian_lo"):  s+=0.10;c.append("Asian_Lo")
    if not buy and mtf.get("near_asian_hi"): s+=0.10;c.append("Asian_Hi")
    if buy  and mtf.get("above_asian"):    s+=0.08;c.append("Asian_Break↑")
    if not buy and mtf.get("below_asian"):  s+=0.08;c.append("Asian_Break↓")

    # ── Judas Swing (London open sweep + reversal) ───────────────────
    judas = mtf.get("judas")
    if buy  and judas == "bull_reversal": s+=0.15;c.append("Judas_Sweep↑")
    if not buy and judas == "bear_reversal": s+=0.15;c.append("Judas_Sweep↓")

    # ── Breaker Blocks ────────────────────────────────────────────────
    breakers = mtf.get("breakers", [])
    if buy  and any(b["bull"] for b in breakers): s+=0.12;c.append("Breaker↑")
    if not buy and any(not b["bull"] for b in breakers): s+=0.12;c.append("Breaker↓")

    return round(min(s,1.0),3),c


# ── Correlation map — pairs that move together ─────────────────────────
CORRELATIONS = {
    # Pairs that are strongly correlated (>0.8) — if one fires, reduce score on others
    frozenset(["EURUSD","GBPUSD"]): 0.85,
    frozenset(["GBPUSD","GBPJPY"]): 0.80,
    frozenset(["EURUSD","AUDUSD"]): 0.75,
    frozenset(["USDJPY","GBPJPY"]): 0.82,
    frozenset(["EURUSD","USDCAD"]): -0.80,  # inverse
}

def get_correlated_active(pair: str, trade_state: dict) -> list:
    """Return list of correlated pairs that already have active trades."""
    active_pairs = [p for p,s in trade_state.items() if s.get("status") in ("active","tp1_hit")]
    correlated = []
    for ap in active_pairs:
        key = frozenset([pair, ap])
        if key in CORRELATIONS:
            correlated.append((ap, CORRELATIONS[key]))
    return correlated

def calc_limit_entry(direction, price, mtf, p, sl_pips):
    """
    Compute a realistic *limit-order* entry — NOT the current market price.
    Real SMC signals wait for price to pull back to a key technical level
    (Order Block, FVG, or Breaker) before entering.

    BUY  → entry below current price (a discount level)
    SELL → entry above current price (a premium level)
    """
    if direction not in ("BUY", "SELL"):
        return price  # WAIT — caller will not use this anyway

    candidates = []

    if direction == "BUY":
        # Order Block top / mid — price often retests bullish OB before continuing up
        for ob in mtf.get("obs", []) or []:
            top = ob.get("hi"); mid = ob.get("mid")
            if ob.get("bull") and top and top < price:
                candidates.append(top)
            if ob.get("bull") and mid and mid < price:
                candidates.append(mid)
        # FVG midpoint — fair value gaps tend to fill
        for fvg in mtf.get("fvgs", []) or []:
            m = fvg.get("mid")
            if fvg.get("bull") and m and m < price:
                candidates.append(m)
        # Breaker block top — flipped support, retest entry
        for bk in mtf.get("breakers", []) or []:
            top = bk.get("hi")
            if bk.get("bull") and top and top < price:
                candidates.append(top)
        # Asian range low — institutional pullback level
        ar = mtf.get("asian_range")
        if ar and ar.get("lo") and ar["lo"] < price:
            candidates.append(ar["lo"])
    else:  # SELL
        for ob in mtf.get("obs", []) or []:
            bot = ob.get("lo"); mid = ob.get("mid")
            if (ob.get("bull") is False) and bot and bot > price:
                candidates.append(bot)
            if (ob.get("bull") is False) and mid and mid > price:
                candidates.append(mid)
        for fvg in mtf.get("fvgs", []) or []:
            m = fvg.get("mid")
            if (fvg.get("bull") is False) and m and m > price:
                candidates.append(m)
        for bk in mtf.get("breakers", []) or []:
            bot = bk.get("lo")
            if (bk.get("bull") is False) and bot and bot > price:
                candidates.append(bot)
        ar = mtf.get("asian_range")
        if ar and ar.get("hi") and ar["hi"] > price:
            candidates.append(ar["hi"])

    # Filter to a sensible distance window: at least ~25% of SL away,
    # at most ~120% of SL (otherwise the entry is unreachable).
    min_d = max(p * 5, p * sl_pips * 0.25)
    max_d = p * sl_pips * 1.2
    valid = []
    for c in candidates:
        d = (price - c) if direction == "BUY" else (c - price)
        if min_d <= d <= max_d:
            valid.append(c)

    if valid:
        # Closest to current price = highest fill probability
        return max(valid) if direction == "BUY" else min(valid)

    # Fallback — half-SL pullback (so the signal is still a limit, not market)
    offset = p * sl_pips * 0.5
    return price - offset if direction == "BUY" else price + offset


def make_signal(pair, htf, mtf, ltf, bars=None):
    p=PIP.get(pair,0.0001); dp=DIGITS.get(pair,5); price=mtf.get("price",0)
    bs,bc=score(True, htf,mtf,ltf); ss,sc=score(False,htf,mtf,ltf)

    # Check risk pauses before allowing signal
    paused, pause_reason = is_trading_paused()
    daily_ok = can_trade_today(pair)

    if bs>=ss and bs>0.70: direction,sc_val,conf="BUY", bs,bc
    elif ss>bs and ss>0.70: direction,sc_val,conf="SELL",ss,sc
    else: direction,sc_val,conf="WAIT",max(bs,ss),[]

    # Override to WAIT if risk limits hit
    if direction != "WAIT" and paused:
        direction = "WAIT"; conf = []; sc_val = max(bs,ss)
    if direction != "WAIT" and not daily_ok:
        direction = "WAIT"; conf = []

    # ADR calculation
    adr, today_range, adr_pct = calc_adr(bars, 14) if bars else (0, 0, 0)
    adr_pips      = round(adr / p) if p > 0 else 0
    today_pips    = round(today_range / p) if p > 0 else 0

    # Downgrade signal if >80% of daily range already used
    if direction != "WAIT" and adr_pct > 80:
        sc_val = round(sc_val * 0.85, 3)  # reduce score by 15%
        conf.append(f"ADR_late({adr_pct}%)")

    # Minimum confluence count filter (need at least 2 real factors)
    real_conf = [c for c in conf if not c.startswith("ADR") and not c.startswith("vs_")]
    if direction != "WAIT" and len(real_conf) < 2:
        direction = "WAIT"

    # Correlation penalty — if a correlated pair already has an active trade
    if direction != "WAIT":
        corr_active = get_correlated_active(pair, _trade_state)
        for corr_pair, corr_strength in corr_active:
            penalty = round(corr_strength * 0.15, 3)
            sc_val  = round(sc_val - penalty, 3)
            if not any("Corr" in x for x in conf):
                conf.append(f"Corr_penalty({corr_pair})")
            log.info(f"  {pair}: corr penalty -{penalty} due to {corr_pair}")
        if sc_val <= 0.70:
            direction = "WAIT"
            log.info(f"  {pair}: dropped to WAIT after correlation penalty")
    # Fixed 30-pip SL (classic SMC default — no ATR auto-sizing)
    sl_pips = 30
    sl_d    = p * sl_pips

    # ── Realistic limit entry (NOT current market price) ─────────────────
    # Previously: entry = round(price, dp), which meant every signal "started"
    # at the live scan-time price. Now we compute a pullback level from SMC
    # confluence (OB / FVG / Breaker / Asian range). The signal waits in
    # "pending" status until price actually trades to that level.
    if direction in ("BUY", "SELL"):
        raw_entry = calc_limit_entry(direction, price, mtf, p, sl_pips)
    else:
        raw_entry = price  # WAIT signal — entry not used

    entry = round(raw_entry, dp)
    sl  = round(entry - sl_d if direction=="BUY" else entry + sl_d, dp)
    tp1 = round(entry + sl_d*1.5 if direction=="BUY" else entry - sl_d*1.5, dp)
    tp2 = round(entry + sl_d*3.0 if direction=="BUY" else entry - sl_d*3.0, dp)
    rr  = round(abs(tp1-entry)/abs(entry-sl), 2) if abs(entry-sl) > 0 else 0

    # Distance from current price → entry (in pips). 0 = market fill, >0 = waiting
    if direction in ("BUY", "SELL") and p > 0:
        entry_distance_pips = round(abs(price - entry) / p, 1)
    else:
        entry_distance_pips = 0
    entry_type = "limit" if entry_distance_pips >= 1 else "market"

    zones=[]
    for ob in mtf.get("obs",[])[-3:]:
        zones.append({"type":"Bull OB" if ob["bull"] else "Bear OB",
                      "hi":round(ob["hi"],dp),"lo":round(ob["lo"],dp),
                      "status":"Tapped" if ob["tapped"] else "Untouched",
                      "color":"green" if ob["bull"] else "red"})
    for fvg in mtf.get("near",[])[:2]:
        zones.append({"type":"Bull FVG" if fvg["bull"] else "Bear FVG",
                      "hi":round(fvg["hi"],dp),"lo":round(fvg["lo"],dp),
                      "status":"Unfilled","color":"amber"})
    # Breaker blocks
    for bk in mtf.get("breakers",[])[:2]:
        zones.append({"type":bk["type"],
                      "hi":round(bk["hi"],dp),"lo":round(bk["lo"],dp),
                      "status":"Retest Zone","color":bk["color"]})
    # Asian range
    ar = mtf.get("asian_range")
    if ar:
        zones.append({"type":"Asian Hi","price":round(ar["hi"],dp),"status":"Session Level","color":"blue"})
        zones.append({"type":"Asian Lo","price":round(ar["lo"],dp),"status":"Session Level","color":"blue"})

    now_ts = datetime.now(timezone.utc).isoformat()
    kz = mtf.get("killzone")
    # Confluence breakdown for UI chart — SMC + ICT only
    bull_breakdown = {
        "HTF Trend":     0.20 if htf.get("trend")=="bullish" else 0,
        "Order Block":   0.20 if any(o["bull"] for o in mtf.get("tapped",[])) else 0,
        "FVG":           0.15 if any(f["bull"] for f in mtf.get("near",[])) else 0,
        "CHoCH":         0.20 if (mtf.get("choch") and mtf.get("cd")=="bullish") else 0,
        "BOS":           0.10 if (mtf.get("bos") and mtf.get("bd")=="bullish") else 0,
        "Liq Sweep":     0.10 if any(l["type"]=="SSL" for l in mtf.get("liq",[])) else 0,
        "Killzone":      0.12 if kz in ("London","New York") else 0,
        "Asian Range":   0.10 if mtf.get("near_asian_lo") else 0,
        "Judas Swing":   0.15 if mtf.get("judas")=="bull_reversal" else 0,
        "Breaker Block": 0.12 if any(b["bull"] for b in mtf.get("breakers",[])) else 0,
        "Discount Zone": 0.10 if mtf.get("disc") else 0,
    }
    bear_breakdown = {
        "HTF Trend":     0.20 if htf.get("trend")=="bearish" else 0,
        "Order Block":   0.20 if any(not o["bull"] for o in mtf.get("tapped",[])) else 0,
        "FVG":           0.15 if any(not f["bull"] for f in mtf.get("near",[])) else 0,
        "CHoCH":         0.20 if (mtf.get("choch") and mtf.get("cd")=="bearish") else 0,
        "BOS":           0.10 if (mtf.get("bos") and mtf.get("bd")=="bearish") else 0,
        "Liq Sweep":     0.10 if any(l["type"]=="BSL" for l in mtf.get("liq",[])) else 0,
        "Killzone":      0.12 if kz in ("London","New York") else 0,
        "Asian Range":   0.10 if mtf.get("near_asian_hi") else 0,
        "Judas Swing":   0.15 if mtf.get("judas")=="bear_reversal" else 0,
        "Breaker Block": 0.12 if any(not b["bull"] for b in mtf.get("breakers",[])) else 0,
        "Premium Zone":  0.10 if mtf.get("prem") else 0,
    }
    active_breakdown = bull_breakdown if direction=="BUY" else bear_breakdown
    return {"pair":pair,"direction":direction,"score":sc_val,
            "score_pct":round(sc_val*100),"conf":conf,
            "entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":rr,
            "current_price": round(price, dp),
            "entry_type": entry_type,
            "entry_distance_pips": entry_distance_pips,
            "sl_pips":sl_pips,"tp1_pips":int(sl_pips*1.5),"tp2_pips":int(sl_pips*3.0),
            "trend":mtf.get("trend","ranging"),
            "htf_trend":htf.get("trend","ranging"),
            "mtf_trend":mtf.get("trend","ranging"),
            "ltf_trend":ltf.get("trend","ranging"),
            "choch":mtf.get("choch",False),"choch_dir":mtf.get("cd",""),
            "bos":mtf.get("bos",False),"bos_dir":mtf.get("bd",""),
            "disc":mtf.get("disc",False),"prem":mtf.get("prem",False),
            "buy_score":round(bs*100),"sell_score":round(ss*100),
            "zones":zones,"ts":now_ts,
            "signal_age_min":0,
            "conf_breakdown":active_breakdown,
            "ob_count":mtf.get("ob_count",0),
            "fvg_count":mtf.get("fvg_count",0),
            # SMC + ICT context fields
            "killzone":mtf.get("killzone"),
            "asian_range":mtf.get("asian_range"),
            "judas":mtf.get("judas"),
            "breaker_count":len(mtf.get("breakers",[])),
            # Quality metrics
            "grade": grade_signal(
                round(sc_val*100), len(real_conf),
                mtf.get("killzone") in ("London","New York"),
                adr_pct
            ),
            "conf_count": len(real_conf),
            "adr_pips": adr_pips,
            "today_pips": today_pips,
            "adr_pct": adr_pct,
            "paused": paused,
            "pause_reason": pause_reason if paused else "",
            "daily_trade_count": _daily_trade_count.get(
                f"{pair}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}", 0
            )}

def check_trade_outcomes(prices: dict):
    """
    Drive the signal lifecycle on each tick.

    pending → active : when price reaches the limit entry
    pending → cancel : if price has moved past SL before ever filling
                       (the setup is invalidated; no PnL recorded)
    active  → tp/sl  : existing TP1 / TP2 / SL outcome logic
    """
    global _trade_state, _signal_cache, _signal_history

    for pair, state in list(_trade_state.items()):
        status = state.get("status")
        if status not in ("pending", "active", "tp1_hit"):
            continue

        cur = prices.get(pair, {}).get("price", 0)
        if cur == 0:
            continue

        direction = state["direction"]
        entry     = state["entry"]
        sl        = state["sl"]
        tp1       = state["tp1"]
        tp2       = state["tp2"]

        # ── PENDING: waiting for price to reach the limit entry ──────────
        if status == "pending":
            # Invalidate if SL is breached before fill (setup is wrong)
            invalidated = (direction == "BUY"  and cur <= sl) or \
                          (direction == "SELL" and cur >= sl)
            if invalidated:
                log.info(f"  {pair}: PENDING signal CANCELLED — price hit SL before filling")
                _trade_state.pop(pair, None)
                _signal_cache.pop(pair, None)
                continue
            # Fill if price reaches entry
            filled = (direction == "BUY"  and cur <= entry) or \
                     (direction == "SELL" and cur >= entry)
            if filled:
                state["status"]    = "active"
                state["filled_at"] = datetime.now(timezone.utc).isoformat()
                log.info(f"  {pair}: Limit FILLED @ {entry} — now ACTIVE "
                         f"(SL={sl} TP1={tp1} TP2={tp2})")
            else:
                # still pending, nothing else to do
                continue

        outcome   = None
        hit_price = cur

        if direction == "BUY":
            if cur <= sl:
                outcome   = "sl_hit"
                hit_price = sl
            elif cur >= tp2:
                outcome   = "tp2_hit"
                hit_price = tp2
            elif cur >= tp1 and state.get("status") == "active":
                outcome   = "tp1_hit"
                hit_price = tp1
        else:  # SELL
            if cur >= sl:
                outcome   = "sl_hit"
                hit_price = sl
            elif cur <= tp2:
                outcome   = "tp2_hit"
                hit_price = tp2
            elif cur <= tp1 and state.get("status") == "active":
                outcome   = "tp1_hit"
                hit_price = tp1

        if outcome:
            dp  = DIGITS.get(pair, 5)
            p   = PIP.get(pair, 0.0001)
            pnl_pips = (hit_price - entry) / p if direction == "BUY" else (entry - hit_price) / p
            pnl_pips = round(pnl_pips, 1)

            log.info(f"  {pair} {outcome.upper()}: entry={entry} hit={hit_price:.{dp}f} "
                     f"pnl={pnl_pips:+.1f}pips")

            # Record in history
            completed = {**state,
                "status":    outcome,
                "hit_price": round(hit_price, dp),
                "hit_at":    datetime.now(timezone.utc).isoformat(),
                "pnl_pips":  pnl_pips,
                "result":    "win" if "tp" in outcome else "loss"}
            _signal_history.insert(0, completed)
            # No cap on history — keep all completed trades

            # Update session stats
            global _daily_drawdown_pips
            _session_stats["total"] += 1
            if "tp" in outcome:
                _session_stats["wins"] += 1
                _session_stats["consecutive_losses"] = 0
                _session_stats["total_pips_won"] += max(pnl_pips, 0)
                _session_stats["best_trade_pips"] = max(_session_stats["best_trade_pips"], pnl_pips)
            else:
                _session_stats["losses"] += 1
                _session_stats["consecutive_losses"] += 1
                _session_stats["total_pips_lost"] += abs(min(pnl_pips, 0))
                _session_stats["worst_trade_pips"] = min(_session_stats["worst_trade_pips"], pnl_pips)
                _daily_drawdown_pips += pnl_pips  # pnl_pips is negative for losses
            _session_stats["net_pips"] = round(
                _session_stats["total_pips_won"] - _session_stats["total_pips_lost"], 1)
            _session_stats["daily_pnl_pips"] = round(_daily_drawdown_pips, 1)
            _session_stats["win_rate"] = round(
                _session_stats["wins"] / max(_session_stats["total"], 1) * 100)
            # Reset daily drawdown at midnight
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if _session_stats.get("daily_reset_date") != today:
                _session_stats["daily_reset_date"] = today
                _daily_drawdown_pips = 0
                log.info("Daily drawdown counter reset")
            # Record trade for daily limit tracking
            record_daily_trade(pair)
            log.info(f"  Stats: {_session_stats['wins']}W/{_session_stats['losses']}L "
                     f"WR={_session_stats.get('win_rate',0)}% "
                     f"Net={_session_stats['net_pips']}p "
                     f"ConsecL={_session_stats['consecutive_losses']}")

            # Update state — TP1 keeps signal alive (watching for TP2),
            # TP2 and SL both clear the signal → new signal allowed
            if outcome == "tp1_hit":
                _trade_state[pair]["status"] = "tp1_hit"
                # Move virtual SL to breakeven after TP1
                _trade_state[pair]["sl"] = entry
                log.info(f"  {pair}: TP1 hit — SL moved to breakeven, watching TP2")
            else:
                # Signal fully completed — remove from cache so next scan gives fresh signal
                _trade_state.pop(pair, None)
                _signal_cache.pop(pair, None)
                log.info(f"  {pair}: Signal CLOSED ({outcome}) — ready for new signal")


def activate_signal(sig: dict):
    """
    Register a new signal in the trade state tracker.

    A signal is born "pending" — meaning the user / market hasn't yet pulled
    back to the limit entry. It only becomes "active" once price actually
    trades to the entry level. SL/TP outcomes are evaluated only while
    active so a signal cannot hit TP/SL before it fills.
    """
    global _trade_state
    pair = sig["pair"]
    if sig["direction"] == "WAIT":
        return
    # Only register if not already tracking this pair
    if pair not in _trade_state or _trade_state[pair].get("status") not in ("pending","active","tp1_hit"):
        # If the entry is essentially the current price, treat as immediate fill
        initial_status = "pending" if sig.get("entry_type") == "limit" else "active"
        _trade_state[pair] = {
            "pair":      pair,
            "direction": sig["direction"],
            "entry":     sig["entry"],
            "sl":        sig["sl"],
            "tp1":       sig["tp1"],
            "tp2":       sig["tp2"],
            "score_pct": sig["score_pct"],
            "conf":      sig["conf"],
            "status":    initial_status,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "filled_at": None if initial_status == "pending" else datetime.now(timezone.utc).isoformat(),
            "hit_price": None,
            "hit_at":    None,
        }
        if initial_status == "pending":
            log.info(f"  {pair}: Signal PENDING — limit @ {sig['entry']} "
                     f"(distance {sig.get('entry_distance_pips',0)}p), waiting to fill")
        else:
            log.info(f"  {pair}: Signal ACTIVATED — watching SL={sig['sl']} TP1={sig['tp1']} TP2={sig['tp2']}")


def run_scan(force=False):
    global _signal_cache, _last_scan_ts
    _last_scan_ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    ms=market_status(); prices=fetch_prices(); out=[]

    # Step 1: Check if any active signals hit TP or SL
    # If hit → clears cache so a fresh signal can be generated below
    check_trade_outcomes(prices)

    now = datetime.now(timezone.utc)
    for pair in PAIRS:
        cur_price = prices.get(pair, {}).get("price", 0)
        state     = _trade_state.get(pair, {})
        cached    = _signal_cache.get(pair)

        # ── Compute cached-signal age (seconds) ─────────────────────────
        cache_age = float("inf")
        if cached and cached.get("generated_at"):
            try:
                gen_at = datetime.fromisoformat(cached["generated_at"])
                cache_age = (now - gen_at).total_seconds()
            except Exception:
                cache_age = float("inf")

        # ── HOLD conditions (do NOT regenerate the signal) ──────────────
        # 1. Trade is pending fill, active, or partially closed (tp1_hit)
        #    → must let it run until check_trade_outcomes closes it.
        # 2. Signal is younger than SIGNAL_TTL (15 min by default)
        #    → prevents auto-scan from churning a signal every 5 min just
        #    because the live market price drifted. The signal completes
        #    (TP/SL/cancel) before a new one can replace it.
        in_trade  = state.get("status") in ("pending", "active", "tp1_hit")
        ttl_alive = cached is not None and cache_age < SIGNAL_TTL

        if (in_trade or ttl_alive) and not force:
            if cached:
                sig = dict(cached)
                sig["cached"]        = True
                sig["trade_status"]  = state.get("status") or sig.get("direction","WAIT").lower()
                p = PIP.get(pair, 0.0001)
                # Always refresh current price + entry distance
                if cur_price:
                    sig["current_price"] = round(cur_price, DIGITS.get(pair, 5))
                    if sig.get("entry") and p > 0:
                        sig["entry_distance_pips"] = round(
                            abs(cur_price - sig["entry"]) / p, 1)
                # Live P&L pips only meaningful after fill
                if state.get("status") in ("active", "tp1_hit") and cur_price and sig.get("entry"):
                    pnl = (cur_price - sig["entry"]) / p if sig["direction"] == "BUY"                           else (sig["entry"] - cur_price) / p
                    sig["live_pnl_pips"] = round(pnl, 1)
                else:
                    sig["live_pnl_pips"] = 0
                # Refresh visible age so UI can show "X min ago"
                sig["signal_age_min"] = round(cache_age / 60, 1)
                out.append(sig)
                why = "in-trade" if in_trade else f"ttl({int(SIGNAL_TTL-cache_age)}s left)"
                log.info(f"  {pair}: HOLDING signal ({why})")
                continue

        # Step 3: Generate fresh signal (TP/SL hit OR TTL expired OR first scan OR forced)
        try:
            if ms["is_open"]:
                bh=fetch_bars(pair,"H4"); bm=fetch_bars(pair,"H1"); bl=fetch_bars(pair,"M15")
                if not bm: raise Exception("no data")
            else:
                bh=bm=bl=demo_bars(pair)
            htf=analyse(bh or bm,pair); mtf=analyse(bm,pair); ltf=analyse(bl or bm,pair)
            sig=make_signal(pair,htf,mtf,ltf,bars=bm)
            sig["cached"]        = False
            sig["demo"]          = not ms["is_open"]
            sig["trade_status"]  = "new"
            sig["live_pnl_pips"] = 0
            sig["signal_age_min"]= 0
            sig["generated_at"]  = datetime.now(timezone.utc).isoformat()
            _signal_cache[pair]  = sig

            # Step 4: Register new valid signal in trade tracker
            if sig["direction"] != "WAIT":
                activate_signal(sig)

            out.append(sig)
            log.info(f"  {pair}: {sig['direction']} {sig['score_pct']}% [NEW]")
        except Exception as e:
            log.warning(f"  {pair}: {e}")
            if pair in _signal_cache: out.append(_signal_cache[pair])
    return out

# ── API routes ────────────────────────────────────────────────────────
_last_scan_ts: str = ""

@app.route("/api/status")
def api_status():
    ms = market_status()
    ms["last_scan"] = _last_scan_ts
    ms["auto_scan_interval"] = AUTO_SCAN_INTERVAL
    return jsonify(ms)

@app.route("/api/prices")
def api_prices():
    return jsonify({"prices":fetch_prices(),"ts":datetime.now(timezone.utc).isoformat()})

@app.route("/api/scan")
def api_scan():
    force=request.args.get("force","0")=="1"
    sigs=run_scan(force=force)
    ms=market_status()
    return jsonify({"signals":sigs,"market":ms,"count":len(sigs),
                    "ts":datetime.now(timezone.utc).isoformat()})

@app.route("/api/stats")
def api_stats():
    """Full performance statistics."""
    paused, pause_reason = is_trading_paused()
    total = _session_stats.get("total", 0)
    wins  = _session_stats.get("wins", 0)
    return jsonify({
        "stats":           _session_stats,
        "paused":          paused,
        "pause_reason":    pause_reason,
        "daily_drawdown":  _daily_drawdown_pips,
        "history_count":   len(_signal_history),
        "session_start":   _session_start,
        "ts":              datetime.now(timezone.utc).isoformat(),
        # Weekly breakdown
        "weekly": _weekly_stats(),
    })


def _weekly_stats():
    """Compute weekly P&L from signal history."""
    from collections import defaultdict
    weeks = defaultdict(lambda: {"wins":0,"losses":0,"pips":0.0})
    for h in _signal_history:
        try:
            dt  = datetime.fromisoformat(h.get("hit_at","")).strftime("%Y-W%W")
            pnl = h.get("pnl_pips", 0)
            if h.get("result") == "win":  weeks[dt]["wins"]   += 1
            else:                          weeks[dt]["losses"] += 1
            weeks[dt]["pips"] += pnl
        except: pass
    return [{"week": k, **v, "pips": round(v["pips"],1)} for k, v in sorted(weeks.items())[-8:]]


@app.route("/api/reset-pause", methods=["POST"])
def api_reset_pause():
    """Manually reset the consecutive loss counter to resume trading."""
    global _daily_drawdown_pips
    _session_stats["consecutive_losses"] = 0
    _daily_drawdown_pips = 0
    log.info("Trading pause manually reset")
    return jsonify({"message": "Trading resumed", "stats": _session_stats})


@app.route("/api/export-history")
def api_export_history():
    """Download signal history as CSV."""
    import io
    lines = ["pair,direction,score,entry,sl,tp1,tp2,status,pnl_pips,result,opened_at,hit_at"]
    for h in _signal_history:
        lines.append(",".join(str(h.get(k,"")) for k in
            ["pair","direction","score_pct","entry","sl","tp1","tp2",
             "status","pnl_pips","result","opened_at","hit_at"]))
    csv = "\n".join(lines)
    from flask import make_response
    resp = make_response(csv)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=smc_signal_history.csv"
    return resp


@app.route("/api/trade-state")
def api_trade_state():
    """Returns active trade states and signal history."""
    return jsonify({
        "active":  _trade_state,
        "history": _signal_history,
        "ts":      datetime.now(timezone.utc).isoformat()
    })


@app.route("/api/bars/<pair>")
def api_bars(pair):
    tf=request.args.get("tf","H1"); limit=int(request.args.get("limit",100))
    ms=market_status()
    if ms["is_open"]:
        bars=fetch_bars(pair.upper(),tf)
        if not bars: bars=demo_bars(pair.upper())
    else:
        bars=demo_bars(pair.upper())
    return jsonify({"pair":pair,"bars":bars[-limit:],"demo":not ms["is_open"]})

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

# ══════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMC Signal Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#05080b;--bg1:#090d12;--bg2:#0d1219;--bg3:#111820;
  --b:rgba(255,255,255,.06);--b2:rgba(255,255,255,.11);
  --g:#00ffb3;--r:#ff3d5a;--a:#ffb800;--bl:#3d9fff;
  --ga:rgba(0,255,179,.08);--ra:rgba(255,61,90,.08);--aa:rgba(255,184,0,.08);--bla:rgba(61,159,255,.08);
  --t:#dde6f0;--t2:#6b8099;--t3:#2d404f;
  --mono:'JetBrains Mono',monospace;--disp:'Bebas Neue',sans-serif;--body:'DM Sans',sans-serif;
}
html,body{height:100%}
body{background:var(--bg);color:var(--t);font-family:var(--body);font-size:14px;overflow:hidden}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.02) 2px,rgba(0,0,0,.02) 4px);pointer-events:none;z-index:0}
::-webkit-scrollbar{width:3px;height:3px}::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}
.app{position:relative;z-index:1;display:grid;grid-template-rows:52px 24px 1fr;height:100vh;overflow:hidden}
/* Nav */
nav{display:flex;align-items:center;justify-content:space-between;padding:0 1.25rem;background:rgba(5,8,11,.95);border-bottom:1px solid var(--b);backdrop-filter:blur(12px)}
.logo{font-family:var(--disp);font-size:20px;letter-spacing:.05em}.logo em{color:var(--g);font-style:normal}
.nav-r{display:flex;align-items:center;gap:.6rem}
.mkt-badge{display:flex;align-items:center;gap:5px;font-family:var(--mono);font-size:9px;padding:3px 10px;border-radius:3px;letter-spacing:.1em;border:1px solid}
.open-b{color:var(--g);border-color:rgba(0,255,179,.25);background:var(--ga)}
.closed-b{color:var(--a);border-color:rgba(255,184,0,.25);background:var(--aa)}
.bd{width:5px;height:5px;border-radius:50%}
.open-b .bd{background:var(--g);animation:pulse 1.2s infinite}
.closed-b .bd{background:var(--a)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.15}}
.clk{font-family:var(--mono);font-size:10px;color:var(--t3)}
.nbtn{font-family:var(--mono);font-size:10px;font-weight:700;padding:5px 14px;border-radius:3px;cursor:pointer;transition:opacity .15s;text-transform:uppercase;border:none}
.scan-btn{background:var(--g);color:#000}.scan-btn:hover{opacity:.8}.scan-btn:disabled{opacity:.35;cursor:default}
.force-btn{background:transparent;color:var(--a);border:1px solid rgba(255,184,0,.3)!important}.force-btn:hover{background:var(--aa)}
.snd-btn{background:transparent;color:var(--t3);border:1px solid var(--b)!important;font-size:14px;padding:4px 8px}
/* Status bar */
.statusbar{display:flex;align-items:center;justify-content:space-between;padding:0 1.25rem;background:var(--bg1);border-bottom:1px solid var(--b);font-family:var(--mono);font-size:9px;color:var(--t3);height:24px}
/* Ticker */
.ticker{position:relative;overflow:hidden;display:flex;align-items:center;padding:0 1rem;gap:1.75rem;font-family:var(--mono);font-size:9px;flex-shrink:0}
.tick{display:flex;gap:6px;align-items:center;white-space:nowrap;cursor:pointer;padding:3px 0}
.tick:hover .tsym{color:var(--g)}.tsym{color:var(--t);font-weight:500}.tpx{color:var(--t2)}.tup{color:var(--g)}.tdn{color:var(--r)}
.thigh{color:var(--t3);font-size:8px}.tlow{color:var(--t3);font-size:8px}
/* Main */
.main{display:grid;grid-template-columns:310px 1fr;overflow:hidden}
/* Left */
.left{border-right:1px solid var(--b);background:var(--bg1);overflow-y:auto;display:flex;flex-direction:column}
.sect{padding:.8rem 1rem;border-bottom:1px solid var(--b)}
.slbl{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--t3);margin-bottom:.5rem}
/* Pairs */
.pgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.pb{font-family:var(--mono);font-size:10px;font-weight:500;padding:6px 3px;border-radius:4px;border:1px solid var(--b);background:var(--bg2);color:var(--t2);cursor:pointer;text-align:center;transition:all .15s;position:relative}
.pb:hover{color:var(--t);border-color:var(--b2)}.pb.on{background:var(--ga);border-color:rgba(0,255,179,.3);color:var(--g)}
.pb-sig{position:absolute;top:2px;right:3px;width:5px;height:5px;border-radius:50%;display:none}
.pb-buy{background:var(--g);display:block}.pb-sell{background:var(--r);display:block}.pb-wait{background:var(--a);display:block}
/* Score */
.big-score{font-family:var(--disp);font-size:52px;line-height:1;letter-spacing:.02em;margin-bottom:.3rem}
.bs-bull{color:var(--g)}.bs-bear{color:var(--r)}.bs-wait{color:var(--a)}
.bar-bg{height:4px;background:var(--bg3);border-radius:2px;overflow:hidden;margin-bottom:.3rem}
.bar-fill{height:100%;border-radius:2px;transition:width .5s,background .3s}
.blbl{display:flex;justify-content:space-between;font-family:var(--mono);font-size:8px;color:var(--t3)}
/* Signal */
.sig-dir{font-family:var(--disp);font-size:36px;letter-spacing:.05em}
.sd-buy{color:var(--g)}.sd-sell{color:var(--r)}.sd-wait{color:var(--a)}
.cache-tag{font-family:var(--mono);font-size:8px;padding:2px 6px;border-radius:3px}
.ct-live{background:var(--ga);color:var(--g);border:1px solid rgba(0,255,179,.2)}
.ct-cached{background:var(--bla);color:var(--bl);border:1px solid rgba(61,159,255,.2)}
.ct-demo{background:var(--aa);color:var(--a);border:1px solid rgba(255,184,0,.2)}
.lvgrid{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin:.5rem 0}
.lv{background:var(--bg3);border-radius:4px;padding:5px 7px}
.lvk{font-family:var(--mono);font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:1px}
.lvv{font-family:var(--mono);font-size:12px;font-weight:500}
.lv-e{color:var(--bl)}.lv-s{color:var(--r)}.lv-t{color:var(--g)}.lv-rr{color:var(--t)}.lv-age{color:var(--t2);font-size:11px}
.ctags{display:flex;flex-wrap:wrap;gap:3px;margin-top:.35rem}
.ctag{font-family:var(--mono);font-size:8px;padding:2px 5px;border-radius:3px;background:var(--ga);color:var(--g);border:1px solid rgba(0,255,179,.15)}
.ctag.bear{background:var(--ra);color:var(--r);border-color:rgba(255,61,90,.15)}
.ctag.neutral{background:var(--bg3);color:var(--t2);border-color:var(--b)}
/* Multi TF bias */
.tf-row{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.tf-card{background:var(--bg2);border:1px solid var(--b);border-radius:4px;padding:5px 7px;text-align:center}
.tf-label{font-family:var(--mono);font-size:8px;color:var(--t3);text-transform:uppercase;margin-bottom:2px}
.tf-trend{font-family:var(--mono);font-size:10px;font-weight:500}
.trend-bull{color:var(--g)}.trend-bear{color:var(--r)}.trend-range{color:var(--a)}
/* Scores breakdown */
.breakdown-list{display:flex;flex-direction:column;gap:3px}
.brow{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:9px}
.brow-name{color:var(--t2);min-width:88px;font-size:8px}
.brow-bar{flex:1;height:3px;background:var(--bg3);border-radius:2px;overflow:hidden}
.brow-fill{height:100%;border-radius:2px;background:var(--g);transition:width .4s}
.brow-fill.zero{background:var(--bg3)}
.brow-val{color:var(--t3);min-width:24px;text-align:right;font-size:8px}
/* Risk calc */
.risk-row{display:flex;gap:6px;align-items:center;font-family:var(--mono);font-size:9px;margin-bottom:4px}
.risk-input{background:var(--bg2);border:1px solid var(--b);color:var(--t);padding:4px 6px;border-radius:3px;font-family:var(--mono);font-size:10px;width:100%;outline:none}
.risk-input:focus{border-color:var(--b2)}
.risk-result{background:var(--bg2);border:1px solid var(--b);border-radius:4px;padding:6px 8px;font-family:var(--mono);font-size:10px;display:grid;grid-template-columns:1fr 1fr;gap:4px}
.rr-key{color:var(--t3);font-size:8px;text-transform:uppercase}
.rr-val{color:var(--g);font-weight:500}
/* Zones */
.zlist{display:flex;flex-direction:column;gap:3px}
.zrow{display:flex;align-items:center;gap:5px;padding:4px 7px;border-radius:3px;border:1px solid var(--b);background:var(--bg2)}
.zdot{width:6px;height:6px;border-radius:1px;flex-shrink:0}
.zg{background:var(--g)}.zr{background:var(--r)}.za{background:var(--a)}.zb{background:var(--bl)}
.zname{font-family:var(--mono);font-size:9px;font-weight:500;min-width:64px;color:var(--t)}
.zprice{font-family:var(--mono);font-size:9px;color:var(--t2);flex:1}
.zst{font-family:var(--mono);font-size:8px}
.zst-g{color:var(--g)}.zst-a{color:var(--a)}.zst-x{color:var(--t3)}
/* Metrics */
.mgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.mc{background:var(--bg2);border:1px solid var(--b);border-radius:3px;padding:.45rem .6rem}
.mck{font-family:var(--mono);font-size:8px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);margin-bottom:2px}
.mcv{font-family:var(--mono);font-size:13px;font-weight:500}
.mcg{color:var(--g)}.mcr{color:var(--r)}.mca{color:var(--a)}
/* Trade status */
.trade-status{font-family:var(--mono);font-size:8px;padding:2px 6px;border-radius:3px;display:inline-block;margin-bottom:.3rem}
.ts-active{background:var(--ga);color:var(--g);border:1px solid rgba(0,255,179,.2)}
.ts-tp1{background:var(--bla);color:var(--bl);border:1px solid rgba(61,159,255,.2)}
/* Right */
.right{display:flex;flex-direction:column;overflow:hidden;background:var(--bg)}
.chart-area{flex:1;padding:.9rem 1.25rem .4rem;display:flex;flex-direction:column;min-height:0;overflow:hidden}
.chart-hdr{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.65rem}
.csym{font-family:var(--disp);font-size:26px;letter-spacing:.04em;color:var(--t)}
.cmeta{font-family:var(--mono);font-size:9px;color:var(--t2);display:flex;gap:.6rem;margin-top:2px;flex-wrap:wrap}
.cpx{font-family:var(--mono);font-size:13px;font-weight:500;text-align:right}
.cup{color:var(--g)}.cdn{color:var(--r)}
.chart-wrap{flex:1;position:relative;min-height:200px;background:var(--bg)}
/* Table */
.tbl-wrap{padding:.4rem 1.25rem .6rem;border-top:1px solid var(--b);overflow-x:auto}
.tlbl{display:flex;align-items:center;justify-content:space-between;font-family:var(--mono);font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);padding:.3rem 0}
.export-btn{font-family:var(--mono);font-size:8px;padding:2px 8px;border-radius:3px;border:1px solid var(--b2);background:transparent;color:var(--t2);cursor:pointer;text-transform:uppercase;letter-spacing:.06em}
.export-btn:hover{background:var(--bg2);color:var(--t)}
table{width:100%;border-collapse:collapse;min-width:600px}
th{font-family:var(--mono);font-size:8px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);padding:4px 7px;text-align:left;border-bottom:1px solid var(--b)}
td{font-family:var(--mono);font-size:10px;padding:5px 7px;border-bottom:1px solid var(--b);color:var(--t2)}
tr:last-child td{border-bottom:none}tr:hover td{background:var(--bg2)}
.db{color:var(--g);font-weight:700}.ds{color:var(--r);font-weight:700}.dw{color:var(--a)}
.sp{font-size:8px;padding:1px 5px;border-radius:2px}
.sph{background:var(--ga);color:var(--g)}.spm{background:var(--aa);color:var(--a)}.spl{background:var(--ra);color:var(--r)}
/* Log */
.log{height:110px;border-top:1px solid var(--b);overflow-y:auto;padding:.5rem 1.25rem;background:var(--bg1);display:flex;flex-direction:column;gap:2px}
.lr{display:flex;gap:8px;font-family:var(--mono);font-size:9px;line-height:1.8;align-items:baseline}
.lt{color:var(--t3);min-width:38px}.lp{min-width:46px;font-weight:500}
.lg{color:var(--g)}.lr2{color:var(--r)}.lb{color:var(--bl)}.la{color:var(--a)}
.lmsg{color:var(--t2);flex:1}
.ltag{font-size:8px;padding:1px 4px;border-radius:2px;margin-left:2px}
.ltb{background:var(--ga);color:var(--g)}.lts{background:var(--ra);color:var(--r)}
.lti{background:var(--bla);color:var(--bl)}.ltw{background:var(--aa);color:var(--a)}
/* Closed screen */
.closed-screen{display:none;position:fixed;inset:0;z-index:50;background:var(--bg);flex-direction:column;align-items:center;justify-content:center;gap:1rem;text-align:center}
.closed-screen.show{display:flex}
.cs-icon{font-size:44px}.cs-title{font-family:var(--disp);font-size:44px;letter-spacing:.05em;color:var(--a)}
.cs-reason{font-family:var(--mono);font-size:11px;color:var(--t2);max-width:400px;line-height:1.8}
.cs-next{font-family:var(--mono);font-size:11px;color:var(--g);padding:6px 16px;border:1px solid rgba(0,255,179,.3);border-radius:3px;background:var(--ga)}
.cs-time{font-family:var(--mono);font-size:9px;color:var(--t3)}
.sess-row{display:flex;gap:8px}
.sess-card{background:var(--bg2);border:1px solid var(--b);border-radius:5px;padding:.5rem .8rem;text-align:left}
.sc-name{font-family:var(--disp);font-size:13px;letter-spacing:.04em;margin-bottom:1px}
.sc-time{font-family:var(--mono);font-size:8px;color:var(--t2)}.sc-st{font-family:var(--mono);font-size:8px;margin-top:2px}
.sc-open{color:var(--g)}.sc-cls{color:var(--t3)}
.demo-btn{font-family:var(--mono);font-size:10px;padding:5px 12px;border-radius:3px;border:1px solid var(--b2);background:transparent;color:var(--t2);cursor:pointer;margin-top:.25rem}
.demo-btn:hover{background:var(--bg2);color:var(--t)}
/* Spinner */
.spin-overlay{display:none;position:fixed;inset:0;z-index:100;background:rgba(5,8,11,.8);backdrop-filter:blur(8px);flex-direction:column;align-items:center;justify-content:center;gap:1rem}
.spin-overlay.show{display:flex}
.spinner{width:40px;height:40px;border:2px solid var(--b2);border-top-color:var(--g);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.spin-txt{font-family:var(--disp);font-size:22px;letter-spacing:.1em;color:var(--g)}
.spin-sub{font-family:var(--mono);font-size:10px;color:var(--t2)}
/* Toast */
.tf-btn{font-family:var(--mono);font-size:9px;padding:2px 7px;border-radius:3px;border:1px solid var(--b);background:var(--bg2);color:var(--t3);cursor:pointer;transition:all .12s}
.tf-btn:hover{color:var(--t);border-color:var(--b2)}
.tf-btn.on{background:var(--bg3);border-color:var(--b2);color:var(--t)}
.toast{position:fixed;bottom:1.25rem;right:1.25rem;z-index:200;font-family:var(--mono);font-size:10px;padding:.6rem .9rem;border-radius:4px;background:var(--bg2);border:1px solid var(--b2);color:var(--t);max-width:280px;transform:translateY(60px);opacity:0;transition:all .25s;pointer-events:none}
.toast.show{transform:translateY(0);opacity:1}
.tb{border-color:rgba(0,255,179,.3);background:var(--ga)}.ts{border-color:rgba(255,61,90,.3);background:var(--ra)}.tw{border-color:rgba(255,184,0,.3);background:var(--aa)}
@media(max-width:800px){.main{grid-template-columns:1fr}.right{display:none}body{overflow:auto}.app{height:auto;overflow:visible}}
</style>
</head>
<body>
<div class="app">
<nav>
  <div class="logo">SMC<em>FX</em> · SIGNALS</div>
  <div class="nav-r">
    <button class="nbtn snd-btn" id="sndBtn" onclick="toggleSound()" title="Sound alerts">🔔</button>
    <div id="mktBadge" class="mkt-badge open-b"><div class="bd"></div><span id="mktLbl">CONNECTING</span></div>
    <div class="clk" id="clk">--:--:-- UTC</div>
    <div style="font-family:var(--mono);font-size:9px;color:var(--t3)" id="localTimeEl"></div>
    <button class="nbtn force-btn" onclick="doScan(true)">↺ FORCE</button>
    <button class="nbtn scan-btn" id="scanBtn" onclick="doScan(false)">⟳ SCAN ALL</button>
  </div>
</nav>

<div class="statusbar">
  <span id="autoScanStatus">⟳ Auto-scan every 5 min</span>
  <span id="countdownEl" style="color:var(--t2)">Next scan: —</span>
  <span id="lastScanEl">Last scan: —</span>
</div>

<div class="main">
  <div class="left">

    <!-- Ticker (vertical in left panel for space) -->
    <div class="sect" style="padding:.5rem 1rem">
      <div class="pgrid">
        <button class="pb on" onclick="selPair(this,'EURUSD')">EUR/USD<span class="pb-sig" id="sig_EURUSD"></span></button>
        <button class="pb"    onclick="selPair(this,'GBPUSD')">GBP/USD<span class="pb-sig" id="sig_GBPUSD"></span></button>
        <button class="pb"    onclick="selPair(this,'USDJPY')">USD/JPY<span class="pb-sig" id="sig_USDJPY"></span></button>
        <button class="pb"    onclick="selPair(this,'GBPJPY')">GBP/JPY<span class="pb-sig" id="sig_GBPJPY"></span></button>
        <button class="pb"    onclick="selPair(this,'AUDUSD')">AUD/USD<span class="pb-sig" id="sig_AUDUSD"></span></button>
        <button class="pb"    onclick="selPair(this,'XAUUSD')">XAU/USD<span class="pb-sig" id="sig_XAUUSD"></span></button>
      </div>
    </div>

    <!-- Score + Signal -->
    <div class="sect">
      <div class="slbl">SMC Score</div>
      <div class="big-score bs-wait" id="bigScore">—</div>
      <div class="bar-bg"><div class="bar-fill" id="barFill" style="width:0%;background:var(--a)"></div></div>
      <div class="blbl"><span>0</span><span>THRESHOLD >70%</span><span>100%</span></div>
    </div>

    <div class="sect">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.4rem">
        <div style="display:flex;align-items:center;gap:7px">
          <div class="sig-dir sd-wait" id="sigDir">WAIT</div>
          <div id="gradeTag" style="font-family:var(--mono);font-size:13px;font-weight:700;padding:2px 8px;border-radius:3px;display:none"></div>
        </div>
        <span class="cache-tag ct-demo" id="cacheTag">—</span>
      </div>
      <div id="tradeStatusBadge"></div>
      <div class="lvgrid">
        <div class="lv"><div class="lvk">Entry</div><div class="lvv lv-e" id="lvE">—</div></div>
        <div class="lv"><div class="lvk">Stop Loss</div><div class="lvv lv-s" id="lvS">—</div></div>
        <div class="lv"><div class="lvk">TP1 · 1.5R</div><div class="lvv lv-t" id="lvT1">—</div></div>
        <div class="lv"><div class="lvk">TP2 · 3.0R</div><div class="lvv lv-t" id="lvT2">—</div></div>
        <div class="lv"><div class="lvk">R:R</div><div class="lvv lv-rr" id="lvRR">—</div></div>
        <div class="lv"><div class="lvk">Age</div><div class="lvv lv-age" id="lvAge">—</div></div>
        <div class="lv" style="grid-column:span 2"><div class="lvk">Live P&amp;L</div><div class="lvv" id="lvPnl" style="color:var(--t2)">— pips</div></div>
        <div class="lv"><div class="lvk">ADR</div><div class="lvv" id="lvADR" style="color:var(--t2)">—</div></div>
        <div class="lv"><div class="lvk">ADR Used</div><div class="lvv" id="lvADRpct" style="color:var(--t2)">—</div></div>
      </div>
      <div class="ctags" id="ctags"><span class="ctag neutral">Run scan first</span></div>
    </div>

    <!-- Multi-TF Bias -->
    <div class="sect">
      <div class="slbl">Multi-Timeframe Bias</div>
      <div class="tf-row">
        <div class="tf-card"><div class="tf-label">H4 · HTF</div><div class="tf-trend trend-range" id="tfH4">—</div></div>
        <div class="tf-card"><div class="tf-label">H1 · MTF</div><div class="tf-trend trend-range" id="tfH1">—</div></div>
        <div class="tf-card"><div class="tf-label">M15 · LTF</div><div class="tf-trend trend-range" id="tfM15">—</div></div>
      </div>
    </div>

    <!-- ICT Context Panel -->
    <div class="sect" style="padding:.65rem 1rem">
      <div class="slbl">ICT Context</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
        <div class="lv" style="grid-column:span 2"><div class="lvk">Killzone</div><div class="lvv" id="infoKZ" style="font-size:10px;color:var(--t3)">None</div></div>
        <div class="lv" style="grid-column:span 2"><div class="lvk">Judas Swing</div><div class="lvv" id="infoJudas" style="font-size:10px;color:var(--t3)">None</div></div>
      </div>
    </div>

    <!-- Confluence Breakdown -->
    <div class="sect">
      <div style="display:flex;justify-content:space-between;margin-bottom:.4rem">
        <div class="slbl" style="margin:0">Confluence Breakdown</div>
        <div style="font-family:var(--mono);font-size:9px;color:var(--t3)">
          <span style="color:var(--g)" id="bsBull">B:—%</span>
          <span style="margin:0 4px;color:var(--t3)">|</span>
          <span style="color:var(--r)" id="bsBear">S:—%</span>
        </div>
      </div>
      <div class="breakdown-list" id="breakdownList">
        <div style="font-family:var(--mono);font-size:9px;color:var(--t3)">Scan to see breakdown</div>
      </div>
    </div>

    <!-- Risk Calculator -->
    <div class="sect">
      <div class="slbl">Risk Calculator</div>
      <div class="risk-row">
        <span style="color:var(--t2);min-width:55px;font-family:var(--mono);font-size:9px">Balance $</span>
        <input class="risk-input" id="rcBalance" type="number" value="10000" min="100" oninput="calcRisk()">
      </div>
      <div class="risk-row">
        <span style="color:var(--t2);min-width:55px;font-family:var(--mono);font-size:9px">Risk %</span>
        <input class="risk-input" id="rcRisk" type="number" value="1" min="0.1" max="10" step="0.1" oninput="calcRisk()">
      </div>
      <div class="risk-row">
        <span style="color:var(--t2);min-width:55px;font-family:var(--mono);font-size:9px">SL Pips</span>
        <input class="risk-input" id="rcSLPips" type="number" value="30" min="5" max="200" step="1" oninput="calcRisk()">
      </div>
      <div class="risk-result" id="rcResult">
        <div><div class="rr-key">Risk $</div><div class="rr-val" id="rcDollar">$100</div></div>
        <div><div class="rr-key">Lot Size</div><div class="rr-val" id="rcLots">0.33</div></div>
        <div><div class="rr-key">SL Cost</div><div class="rr-val" id="rcSL">$30</div></div>
        <div><div class="rr-key">TP1 Profit</div><div class="rr-val" id="rcTP1">$45</div></div>
        <div style="grid-column:span 2"><div class="rr-key">Pip Value (std lot)</div><div class="rr-val" id="rcPipVal" style="color:var(--t2);font-size:10px">—</div></div>
      </div>
    </div>

    <!-- SMC Zones -->
    <div class="sect">
      <div class="slbl">SMC Zones</div>
      <div class="zlist" id="zlist"><div class="zrow"><span class="zname" style="color:var(--t3)">Scan to detect</span></div></div>
    </div>

    <!-- P&L History -->
    <div class="sect">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.4rem">
        <div class="slbl" style="margin:0">P&amp;L History</div>
        <button class="export-btn" onclick="exportCSV()">⬇ CSV</button>
      </div>
      <div id="histList" style="display:flex;flex-direction:column;gap:3px;max-height:150px;overflow-y:auto">
        <div style="font-family:var(--mono);font-size:9px;color:var(--t3);padding:3px 0">No completed signals yet</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-top:.4rem">
        <div class="mc"><div class="mck">Total</div><div class="mcv" id="hTotal">0</div></div>
        <div class="mc"><div class="mck">Win Rate</div><div class="mcv mcg" id="hWR">—</div></div>
        <div class="mc"><div class="mck">Net Pips</div><div class="mcv" id="hNet">0</div></div>
      </div>
    </div>

    <!-- Pause Warning Banner -->
    <div id="pauseBanner" style="display:none;background:rgba(255,61,90,.12);border:1px solid rgba(255,61,90,.3);border-radius:4px;padding:8px 10px;margin:.8rem 1rem 0;font-family:var(--mono);font-size:9px">
      <div style="color:var(--r);font-weight:700;margin-bottom:3px">⛔ TRADING PAUSED</div>
      <div id="pauseReason" style="color:var(--t2);line-height:1.6"></div>
      <button onclick="resetPause()" style="margin-top:6px;font-family:var(--mono);font-size:9px;padding:3px 8px;border-radius:3px;background:transparent;color:var(--r);border:1px solid rgba(255,61,90,.3);cursor:pointer">Reset &amp; Resume</button>
    </div>

    <!-- Session Stats -->
    <div class="sect">
      <div class="slbl">Session Stats</div>
      <div class="mgrid">
        <div class="mc"><div class="mck">Signals</div><div class="mcv" id="mSig">0</div></div>
        <div class="mc"><div class="mck">Buy</div><div class="mcv mcg" id="mBuy">0</div></div>
        <div class="mc"><div class="mck">Sell</div><div class="mcv mcr" id="mSell">0</div></div>
        <div class="mc"><div class="mck">Avg Score</div><div class="mcv mca" id="mAvg">—</div></div>
        <div class="mc"><div class="mck">Last Scan</div><div class="mcv" id="mLast" style="font-size:9px;color:var(--t2)">—</div></div>
        <div class="mc"><div class="mck">Mode</div><div class="mcv mcg" id="mMode" style="font-size:10px">LIVE</div></div>
      </div>
      <div class="mgrid" style="margin-top:4px">
        <div class="mc"><div class="mck">Win Rate</div><div class="mcv mcg" id="stWR">—</div></div>
        <div class="mc"><div class="mck">Net Pips</div><div class="mcv" id="stNet">—</div></div>
        <div class="mc"><div class="mck">Daily P&L</div><div class="mcv" id="stDaily">—</div></div>
        <div class="mc"><div class="mck">ConsecL</div><div class="mcv mcr" id="stCL">0</div></div>
        <div class="mc"><div class="mck">Best</div><div class="mcv mcg" id="stBest">—</div></div>
        <div class="mc"><div class="mck">Worst</div><div class="mcv mcr" id="stWorst">—</div></div>
      </div>
    </div>

    <!-- Weekly Performance -->
    <div class="sect">
      <div class="slbl">Weekly Performance</div>
      <div id="weeklyList" style="display:flex;flex-direction:column;gap:3px;font-family:var(--mono);font-size:9px">
        <div style="color:var(--t3)">No weekly data yet</div>
      </div>
    </div>
  </div><!-- /left -->

  <div class="right">
    <!-- Ticker strip -->
    <div class="ticker" style="background:var(--bg1);border-bottom:1px solid var(--b);height:26px">
      <span class="tick" id="tick_EURUSD" onclick="selPair(null,'EURUSD')"><span class="tsym">EUR/USD</span><span class="tpx">—</span></span>
      <span class="tick" id="tick_GBPUSD" onclick="selPair(null,'GBPUSD')"><span class="tsym">GBP/USD</span><span class="tpx">—</span></span>
      <span class="tick" id="tick_USDJPY" onclick="selPair(null,'USDJPY')"><span class="tsym">USD/JPY</span><span class="tpx">—</span></span>
      <span class="tick" id="tick_GBPJPY" onclick="selPair(null,'GBPJPY')"><span class="tsym">GBP/JPY</span><span class="tpx">—</span></span>
      <span class="tick" id="tick_AUDUSD" onclick="selPair(null,'AUDUSD')"><span class="tsym">AUD/USD</span><span class="tpx">—</span></span>
      <span class="tick" id="tick_USDCAD" onclick="selPair(null,'USDCAD')"><span class="tsym">USD/CAD</span><span class="tpx">—</span></span>
      <span class="tick" id="tick_XAUUSD" onclick="selPair(null,'XAUUSD')"><span class="tsym">XAU/USD</span><span class="tpx">—</span></span>
    </div>

    <div class="chart-area">
      <div class="chart-hdr">
        <div>
          <div style="display:flex;align-items:center;gap:10px">
            <div class="csym" id="csym">—/—</div>
            <div style="display:flex;gap:3px" id="tfBtns">
              <button class="tf-btn on" onclick="switchTF(this,'M15')">M15</button>
              <button class="tf-btn on" onclick="switchTF(this,'H1')">H1</button>
              <button class="tf-btn"    onclick="switchTF(this,'H4')">H4</button>
              <button class="tf-btn"    onclick="switchTF(this,'D1')">D1</button>
            </div>
          </div>
          <div class="cmeta" style="margin-top:4px">
            <span id="cBias">Bias: —</span>
            <span id="cStruct">Structure: —</span>
            <span id="cSess">Session: —</span>
            <span id="cKZ" style="color:var(--g)"></span>
          </div>
        </div>
        <div style="text-align:right">
          <div class="cpx cup" id="cpx">—</div>
          <div style="font-family:var(--mono);font-size:8px;color:var(--t3)" id="cpxChg">—</div>
          <div style="font-family:var(--mono);font-size:8px;color:var(--t3)" id="cpxSrc">—</div>
        </div>
      </div>
      <div class="chart-wrap"><canvas id="chart" role="img" aria-label="SMC price chart"></canvas></div>
      <div style="display:flex;gap:12px;padding:3px 0;font-family:var(--mono);font-size:8px;color:var(--t3)">
        <span><span style="color:rgba(0,255,179,.9)">─</span> Entry</span>
        <span><span style="color:rgba(255,61,90,.9)">- -</span> SL</span>
        <span><span style="color:rgba(0,255,179,.7)">- -</span> TP1</span>
        <span><span style="color:rgba(0,255,179,.95)">─</span> TP2</span>
        <span><span style="color:rgba(255,184,0,.5)">···</span> BB</span>
        <span><span style="color:rgba(61,159,255,.5)">···</span> EMA200</span>
      </div>
    </div>

    <div class="tbl-wrap">
      <div class="tlbl">
        <span>All Pairs · Scan Results</span>
        <button class="export-btn" onclick="exportCSV()">⬇ Export History CSV</button>
      </div>
      <table><thead><tr>
        <th>Pair</th><th>Signal</th><th>Grade</th><th>Score</th><th>HTF</th><th>Entry</th>
        <th>Stop Loss</th><th>TP1 (1.5R)</th><th>TP2 (3R)</th><th>R:R</th><th>Age</th><th>Confluences</th>
      </tr></thead>
      <tbody id="tbody"><tr><td colspan="11" style="color:var(--t3);text-align:center;padding:1rem">Click SCAN ALL or wait for auto-scan</td></tr></tbody>
      </table>
    </div>
    <div class="log" id="logEl"></div>
  </div>
</div><!-- /main -->
</div><!-- /app -->

<!-- Closed screen -->
<div class="closed-screen" id="closedScr">
  <div class="cs-icon">🔒</div>
  <div class="cs-title">MARKET CLOSED</div>
  <div class="cs-reason" id="csReason">—</div>
  <div class="cs-next" id="csNext"></div>
  <div class="cs-time" id="csTime"></div>
  <div class="sess-row">
    <div class="sess-card"><div class="sc-name">SYDNEY</div><div class="sc-time">21:00–06:00</div><div class="sc-st" id="ss0">—</div></div>
    <div class="sess-card"><div class="sc-name">TOKYO</div><div class="sc-time">00:00–09:00</div><div class="sc-st" id="ss1">—</div></div>
    <div class="sess-card"><div class="sc-name">LONDON</div><div class="sc-time">07:00–16:00</div><div class="sc-st" id="ss2">—</div></div>
    <div class="sess-card"><div class="sc-name">NEW YORK</div><div class="sc-time">12:00–21:00</div><div class="sc-st" id="ss3">—</div></div>
  </div>
  <button class="demo-btn" onclick="showDemo()">View demo signals anyway</button>
</div>

<div class="spin-overlay" id="spinner">
  <div class="spinner"></div>
  <div class="spin-txt">SCANNING</div>
  <div class="spin-sub" id="spinSub">Fetching market data...</div>
</div>
<div class="toast" id="toast"></div>

<script>
const DIGITS={EURUSD:5,GBPUSD:5,USDJPY:3,GBPJPY:3,AUDUSD:5,USDCAD:5,XAUUSD:2};
const PIP_V={EURUSD:10,GBPUSD:10,USDJPY:9.3,GBPJPY:7.8,AUDUSD:10,USDCAD:7.3,XAUUSD:10};
let pair='EURUSD', allSigs=[], chart=null;
let cntBuy=0,cntSell=0,scores=[];
let soundOn=true, lastSigIds={};

// ── Clock ─────────────────────────────────────────────────────
function tick(){
  const n=new Date(),p=v=>String(v).padStart(2,'0');
  document.getElementById('clk').textContent=p(n.getUTCHours())+':'+p(n.getUTCMinutes())+':'+p(n.getUTCSeconds())+' UTC';
}
tick(); setInterval(tick,1000);

// ── Sound alert ───────────────────────────────────────────────
let audioCtx=null;
function playBeep(freq=880,dur=0.15){
  if(!soundOn) return;
  try{
    if(!audioCtx) audioCtx=new(window.AudioContext||window.webkitAudioContext)();
    const osc=audioCtx.createOscillator(); const g=audioCtx.createGain();
    osc.connect(g); g.connect(audioCtx.destination);
    osc.frequency.value=freq; osc.type='sine';
    g.gain.setValueAtTime(0.3,audioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001,audioCtx.currentTime+dur);
    osc.start(); osc.stop(audioCtx.currentTime+dur);
  }catch(e){}
}
function toggleSound(){
  soundOn=!soundOn;
  document.getElementById('sndBtn').textContent=soundOn?'🔔':'🔕';
  document.getElementById('sndBtn').style.color=soundOn?'var(--g)':'var(--t3)';
}

// ── Market status ─────────────────────────────────────────────
async function checkStatus(){
  try{
    const d=await(await fetch('/api/status',{signal:AbortSignal.timeout(6000)})).json();
    const badge=document.getElementById('mktBadge');
    const lbl=document.getElementById('mktLbl');
    ['Sydney','Tokyo','London','New York'].forEach((n,i)=>{
      const el=document.getElementById('ss'+i); if(!el) return;
      const on=d.sessions.includes(n); el.textContent=on?'● ACTIVE':'○ CLOSED';
      el.className='sc-st '+(on?'sc-open':'sc-cls');
    });
    if(d.is_open){
      badge.className='mkt-badge open-b'; lbl.textContent=(d.sessions[0]||'OPEN')+' SESSION';
      document.getElementById('cSess').textContent='Session: '+(d.sessions.join('+')||'—');
      if(!isDemoMode) document.getElementById('closedScr').classList.remove('show');
      if(d.last_scan){const ls=document.getElementById('lastScanEl');if(ls)ls.textContent='Last scan: '+d.last_scan;}
    } else {
      badge.className='mkt-badge closed-b'; lbl.textContent='MARKET CLOSED';
      document.getElementById('csReason').textContent=d.reason;
      document.getElementById('csNext').textContent=d.next_open||'';
      document.getElementById('csTime').textContent=d.time_utc;
      if(!isDemoMode) document.getElementById('closedScr').classList.add('show');
    }
    return d;
  }catch(e){return null;}
}

let isDemoMode=false;
function showDemo(){
  isDemoMode=true;
  document.getElementById('closedScr').classList.remove('show');
  document.getElementById('mMode').textContent='DEMO'; document.getElementById('mMode').className='mcv mca';
  addLog('Demo mode — real latest prices, simulated bars','SYS','warn');
  doScan(true);
}

// ── Prices ────────────────────────────────────────────────────
async function loadPrices(){
  try{
    const d=await(await fetch('/api/prices',{signal:AbortSignal.timeout(15000)})).json();
    Object.entries(d.prices||{}).forEach(([p,info])=>{
      const el=document.getElementById('tick_'+p); if(!el) return;
      const dp=DIGITS[p]||5; const chg=info.change_pct||0;
      const hi=info.day_high?info.day_high.toFixed(dp):''; const lo=info.day_low?info.day_low.toFixed(dp):'';
      el.innerHTML=`<span class="tsym">${p.slice(0,3)}/${p.slice(3)}</span>`+
        `<span class="tpx">${info.price.toFixed(dp)}</span>`+
        `<span class="${chg>=0?'tup':'tdn'}">${chg>=0?'+':''}${chg.toFixed(2)}%</span>`+
        (hi?`<span class="thigh">H:${hi}</span>`:'')+
        (lo?`<span class="tlow">L:${lo}</span>`:'');
    });
    // Update spread display for active pair
    if(d.prices[pair]){
      const info=d.prices[pair];
      const chg=info.change_abs||0; const dp=DIGITS[pair]||5;
      const el=document.getElementById('cpxChg');
      if(el){el.textContent=(chg>=0?'+':'')+chg.toFixed(dp)+' ('+(info.change_pct>=0?'+':'')+info.change_pct.toFixed(2)+'%)';
              el.style.color=chg>=0?'var(--g)':'var(--r)';}
    }
  }catch(e){}
}

// ── Risk calculator ───────────────────────────────────────────
function calcRisk(){
  const bal=parseFloat(document.getElementById('rcBalance').value)||10000;
  const riskPct=parseFloat(document.getElementById('rcRisk').value)||1;
  const riskUSD=bal*riskPct/100;
  const pipVal=PIP_V[pair]||10;
  const lots=Math.max(0.01,Math.min((riskUSD/(30*pipVal)).toFixed(2),100));
  document.getElementById('rcDollar').textContent='$'+riskUSD.toFixed(2);
  document.getElementById('rcLots').textContent=lots;
  document.getElementById('rcSL').textContent='$'+(lots*30*pipVal).toFixed(2);
  document.getElementById('rcTP1').textContent='$'+(lots*45*pipVal).toFixed(2);
}

// ── Scan ─────────────────────────────────────────────────────
async function doScan(force){
  const btn=document.getElementById('scanBtn'); btn.disabled=true;
  document.getElementById('spinner').classList.add('show');
  const steps=['Checking market...','Fetching OHLCV...','Detecting Order Blocks...','Scanning FVGs...','Analysing CHoCH & BOS...','Scoring confluences...','Building signals...'];
  let si=0; const iv=setInterval(()=>document.getElementById('spinSub').textContent=steps[Math.min(si++,steps.length-1)],700);
  try{
    const d=await(await fetch('/api/scan'+(force?'?force=1':''),{signal:AbortSignal.timeout(120000)})).json();
    clearInterval(iv);
    allSigs=d.signals||[]; buildTable(allSigs);
    // Update pair signal dots
    allSigs.forEach(s=>{
      const dot=document.getElementById('sig_'+s.pair); if(!dot) return;
      dot.className='pb-sig '+(s.direction==='BUY'?'pb-buy':s.direction==='SELL'?'pb-sell':'pb-wait');
    });
    // Stats
    const tradeSigs=allSigs.filter(s=>s.direction!=='WAIT');
    cntBuy=allSigs.filter(s=>s.direction==='BUY').length;
    cntSell=allSigs.filter(s=>s.direction==='SELL').length;
    scores=tradeSigs.map(s=>s.score_pct);
    document.getElementById('mSig').textContent=tradeSigs.length;
    document.getElementById('mBuy').textContent=cntBuy;
    document.getElementById('mSell').textContent=cntSell;
    document.getElementById('mAvg').textContent=scores.length?Math.round(scores.reduce((a,b)=>a+b,0)/scores.length)+'%':'—';
    const now=new Date();
    const ts=String(now.getUTCHours()).padStart(2,'0')+':'+String(now.getUTCMinutes()).padStart(2,'0')+' UTC';
    document.getElementById('mLast').textContent=ts;
    const lsEl=document.getElementById('lastScanEl'); if(lsEl) lsEl.textContent='Last: '+ts;
    // New signals — sound + toast
    allSigs.forEach(s=>{
      if(s.direction==='WAIT') return;
      const id=s.pair+'_'+s.direction+'_'+s.entry;
      if(!lastSigIds[s.pair]||lastSigIds[s.pair]!==id){
        if(s.trade_status==='new'||!lastSigIds[s.pair]){
          playBeep(s.direction==='BUY'?880:440);
          showToast(`${s.direction==='BUY'?'📈':'📉'} NEW: ${s.pair.slice(0,3)+'/'+s.pair.slice(3)} ${s.direction} — ${s.score_pct}%`,s.direction==='BUY'?'tb':'ts');
          sendPushNotif(`SMC Signal: ${s.pair} ${s.direction}`,`Score: ${s.score_pct}% | Entry: ${s.entry} | Grade: ${s.grade||'C'}`);
          addLog(`${s.direction} | ${s.score_pct}% | ${(s.conf||[]).join(', ')}`,s.pair,s.direction==='BUY'?'buy':'sell');
          lastSigIds[s.pair]=id;
        }
      }
    });
    // Update active pair view
    const active=allSigs.find(s=>s.pair===pair);
    if(active){ updateSignal(active); loadChart(pair); }
    addLog(`Scan done — ${tradeSigs.length} signals active`,'SYS','info');
  }catch(e){ clearInterval(iv); addLog('Scan failed: '+e.message,'ERR','warn'); showToast('Scan failed — check server','tw'); }
  finally{ document.getElementById('spinner').classList.remove('show'); btn.disabled=false; }
}

// ── Pair select ───────────────────────────────────────────────
function selPair(btn,p){
  pair=p;
  document.querySelectorAll('.pb').forEach(b=>b.classList.remove('on'));
  if(btn) btn.classList.add('on');
  else document.querySelectorAll('.pb').forEach(b=>{if(b.textContent.trim().replace('/','')===p.slice(0,3)+p.slice(3)) b.classList.add('on');});
  const sig=allSigs.find(s=>s.pair===p);
  if(sig) updateSignal(sig);
  loadChart(p); calcRisk();
}

// ── Signal UI ─────────────────────────────────────────────────
function updateSignal(sig){
  const dp=DIGITS[sig.pair]||5; const dir=sig.direction||'WAIT';
  document.getElementById('sigDir').textContent=dir;
  document.getElementById('sigDir').className='sig-dir '+(dir==='BUY'?'sd-buy':dir==='SELL'?'sd-sell':'sd-wait');
  document.getElementById('bigScore').textContent=sig.score_pct+'%';
  document.getElementById('bigScore').className='big-score '+(dir==='BUY'?'bs-bull':dir==='SELL'?'bs-bear':'bs-wait');
  const fill=document.getElementById('barFill');
  fill.style.width=sig.score_pct+'%';
  fill.style.background=dir==='BUY'?'var(--g)':dir==='SELL'?'var(--r)':'var(--a)';
  const f=v=>v?v.toFixed(dp):'—';
  document.getElementById('lvE').textContent=dir!=='WAIT'?f(sig.entry):'—';
  document.getElementById('lvS').textContent=dir!=='WAIT'?f(sig.sl)+'  (30p)':'—';
  document.getElementById('lvT1').textContent=dir!=='WAIT'?f(sig.tp1):'—';
  document.getElementById('lvT2').textContent=dir!=='WAIT'?f(sig.tp2):'—';
  document.getElementById('lvRR').textContent=dir!=='WAIT'?'1:'+sig.rr:'—';
  // Age
  const ageEl=document.getElementById('lvAge');
  const age=sig.signal_age_min||0;
  ageEl.textContent=age<1?'Just now':age<60?age+'m ago':Math.floor(age/60)+'h ago';
  // Grade badge
  showGrade(sig.grade||'C', dir);
  // ADR display
  const adrEl=document.getElementById('lvADR');
  const adrPctEl=document.getElementById('lvADRpct');
  if(adrEl) adrEl.textContent=sig.adr_pips?sig.adr_pips+'p':'—';
  if(adrPctEl){
    const pct=sig.adr_pct||0;
    adrPctEl.textContent=pct+'%';
    adrPctEl.style.color=pct>80?'var(--r)':pct>60?'var(--a)':'var(--t2)';
  }
  // Cache tag
  const ct=document.getElementById('cacheTag');
  if(sig.demo){ct.textContent='DEMO';ct.className='cache-tag ct-demo';}
  else if(sig.cached){ct.textContent='CACHED';ct.className='cache-tag ct-cached';}
  else{ct.textContent='LIVE';ct.className='cache-tag ct-live';}
  // Correlation warning in confluences
  const corrConf=(sig.conf||[]).filter(c=>c.startsWith('Corr'));
  if(corrConf.length){
    const tags=document.getElementById('ctags');
    corrConf.forEach(c=>{
      const span=document.createElement('span');
      span.className='ctag neutral';
      span.style.color='var(--a)';
      span.textContent=c;
      tags.appendChild(span);
    });
  }
  // Trade status badge
  const tsb=document.getElementById('tradeStatusBadge');
  const ts=sig.trade_status;
  if(ts==='active') tsb.innerHTML='<span class="trade-status ts-active">● SIGNAL ACTIVE — watching TP/SL</span>';
  else if(ts==='tp1_hit') tsb.innerHTML='<span class="trade-status ts-tp1">▲ TP1 HIT — SL at breakeven</span>';
  else if(ts==='new') tsb.innerHTML='<span class="trade-status ts-active">★ NEW SIGNAL</span>';
  else tsb.innerHTML='';
  // Live P&L
  const pnl=sig.live_pnl_pips||0;
  const pEl=document.getElementById('lvPnl');
  pEl.textContent=(pnl>=0?'+':'')+pnl.toFixed(1)+' pips';
  pEl.style.color=pnl>0?'var(--g)':pnl<0?'var(--r)':'var(--t2)';
  // Multi-TF bias
  const trendCls=t=>t==='bullish'?'trend-bull':t==='bearish'?'trend-bear':'trend-range';
  const trendLbl=t=>t==='bullish'?'↑ BULL':t==='bearish'?'↓ BEAR':'→ RANGE';
  document.getElementById('tfH4').textContent=trendLbl(sig.htf_trend||'ranging');
  document.getElementById('tfH4').className='tf-trend '+trendCls(sig.htf_trend||'ranging');
  document.getElementById('tfH1').textContent=trendLbl(sig.mtf_trend||'ranging');
  document.getElementById('tfH1').className='tf-trend '+trendCls(sig.mtf_trend||'ranging');
  document.getElementById('tfM15').textContent=trendLbl(sig.ltf_trend||'ranging');
  document.getElementById('tfM15').className='tf-trend '+trendCls(sig.ltf_trend||'ranging');
  // ICT context indicators (SMC + ICT only — no RSI/MACD/BB/ATR/Fib/Candle)
  const kzEl=document.getElementById('infoKZ');
  if(kzEl){kzEl.textContent=sig.killzone||'None';kzEl.style.color=sig.killzone?'var(--g)':'var(--t3)';}
  const judEl=document.getElementById('infoJudas');
  if(judEl){judEl.textContent=sig.judas?sig.judas.replace('_',' ').toUpperCase():'None';judEl.style.color=sig.judas?'var(--a)':'var(--t3)';}
  // Confluence breakdown
  document.getElementById('bsBull').textContent='B:'+sig.buy_score+'%';
  document.getElementById('bsBear').textContent='S:'+sig.sell_score+'%';
  buildBreakdown(sig.conf_breakdown||{});
  // Chart meta
  document.getElementById('csym').textContent=sig.pair.slice(0,3)+'/'+sig.pair.slice(3);
  document.getElementById('cBias').textContent='Bias: '+(sig.trend||'—').toUpperCase();
  const struct=sig.choch?`CHoCH(${sig.choch_dir})`:sig.bos?`BOS(${sig.bos_dir})`:'None';
  document.getElementById('cStruct').textContent='Structure: '+struct;
  // Confluences
  const tags=document.getElementById('ctags');
  tags.innerHTML=sig.conf&&sig.conf.length
    ?sig.conf.map(c=>`<span class="ctag${dir==='SELL'?' bear':''}">${c}</span>`).join('')
    :'<span class="ctag neutral">No confluence</span>';
  // Zones
  buildZones(sig.zones||[],dp);
  // Risk calculator update
  calcRisk();
}

function buildBreakdown(bd){
  const el=document.getElementById('breakdownList');
  const items=Object.entries(bd);
  if(!items.length){el.innerHTML='<div style="font-family:var(--mono);font-size:9px;color:var(--t3)">No data</div>';return;}
  const maxW=Math.max(...items.map(([,v])=>v));
  el.innerHTML=items.map(([name,val])=>`
    <div class="brow">
      <span class="brow-name">${name}</span>
      <div class="brow-bar"><div class="brow-fill${val===0?' zero':''}" style="width:${maxW>0?Math.round(val/maxW*100):0}%"></div></div>
      <span class="brow-val">${val>0?Math.round(val*100)+'%':'—'}</span>
    </div>`).join('');
}

function buildZones(zones,dp){
  const zl=document.getElementById('zlist');
  if(!zones.length){zl.innerHTML='<div class="zrow"><span class="zname" style="color:var(--t3)">No zones</span></div>';return;}
  const cm={green:'zg',red:'zr',amber:'za',blue:'zb'};
  const sm={Tapped:'zst-g',Untouched:'zst-x',Unfilled:'zst-a',Target:'zst-x'};
  zl.innerHTML=zones.slice(0,6).map(z=>{
    const px=z.price?z.price.toFixed(dp):(z.lo&&z.hi?z.lo.toFixed(dp)+'–'+z.hi.toFixed(dp):'—');
    return `<div class="zrow"><div class="zdot ${cm[z.color]||'zb'}"></div><span class="zname">${z.type}</span><span class="zprice">${px}</span><span class="zst ${sm[z.status]||'zst-x'}">${z.status}</span></div>`;
  }).join('');
}

// ── Table ─────────────────────────────────────────────────────
function buildTable(sigs){
  const tb=document.getElementById('tbody');
  if(!sigs.length){tb.innerHTML='<tr><td colspan="11" style="color:var(--t3);text-align:center;padding:1rem">No signals yet</td></tr>';return;}
  tb.innerHTML=sigs.map(s=>{
    const dp=DIGITS[s.pair]||5;
    const dc=s.direction==='BUY'?'db':s.direction==='SELL'?'ds':'dw';
    const pc=s.score_pct>70?'sph':s.score_pct>=50?'spm':'spl';
    const f=v=>v&&v!==0?v.toFixed(dp):'—';
    const age=s.signal_age_min||0;
    const ageStr=age<1?'now':age<60?age+'m':Math.floor(age/60)+'h';
    const htfCls=s.htf_trend==='bullish'?'style="color:var(--g)"':s.htf_trend==='bearish'?'style="color:var(--r)"':'style="color:var(--a)"';
    const grCfg={A:'background:rgba(0,255,179,.15);color:var(--g)',B:'background:rgba(61,159,255,.12);color:var(--bl)',C:'background:rgba(255,184,0,.1);color:var(--a)'};
    const grSt=grCfg[s.grade||'C']||grCfg.C;
    return `<tr>
      <td style="color:var(--t);font-weight:500">${s.pair.slice(0,3)+'/'+s.pair.slice(3)}</td>
      <td class="${dc}">${s.direction}</td>
      <td><span style="font-family:var(--mono);font-size:9px;padding:1px 6px;border-radius:3px;${grSt}">${s.grade||'C'}</span></td>
      <td><span class="sp ${pc}">${s.score_pct}%</span></td>
      <td ${htfCls}>${(s.htf_trend||'—').slice(0,4).toUpperCase()}</td>
      <td>${s.direction!=='WAIT'?f(s.entry):'—'}</td>
      <td>${s.direction!=='WAIT'?f(s.sl):'—'}</td>
      <td>${s.direction!=='WAIT'?f(s.tp1):'—'}</td>
      <td>${s.direction!=='WAIT'?f(s.tp2):'—'}</td>
      <td>${s.direction!=='WAIT'?'1:'+s.rr:'—'}</td>
      <td style="color:var(--t3)">${ageStr}</td>
      <td style="color:var(--t2);font-size:9px;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${(s.conf||[]).join(', ')}">${(s.conf||[]).join(' · ')||'—'}</td>
    </tr>`;
  }).join('');
}

// ── Chart ─────────────────────────────────────────────────────
async function loadChart(p, tf){
  tf = tf || currentTF || 'H1';
  const limit = tf==='M15'?100:tf==='H1'?80:tf==='H4'?80:60;
  try{
    const d=await(await fetch(`/api/bars/${p}?tf=${tf}&limit=${limit}`,{signal:AbortSignal.timeout(15000)})).json();
    if(d.bars&&d.bars.length) buildChart(p,d.bars,d.demo,tf);
  }catch(e){}
}
function buildChart(p, bars, isDemo, tf){
  const dp   = DIGITS[p] || 5;
  const sig  = allSigs.find(s => s.pair === p);
  const canvas = document.getElementById('chart');
  if(!canvas) return;
  const ctx  = canvas.getContext('2d');

  // Update price display
  const last = bars[bars.length-1];
  if(last){
    document.getElementById('cpx').textContent = last.close.toFixed(dp);
    document.getElementById('cpx').className = 'cpx ' + (last.close >= bars[0].close ? 'cup' : 'cdn');
  }
  const srcEl = document.getElementById('cpxSrc');
  if(srcEl) srcEl.textContent = (isDemo?'DEMO':'LIVE')+' · '+(tf||'H1')+' · yfinance';
  // Update killzone display
  const kz=getKillzoneLocal();
  const kzEl=document.getElementById('cKZ');
  if(kzEl){
    if(kz.name) kzEl.textContent='⚡ '+kz.name+' Killzone ACTIVE';
    else kzEl.textContent=kz.next_in_mins?'Next KZ in '+kz.next_in_mins+'m':'';
  }

  // Destroy previous Chart.js instance if exists
  if(chart){ chart.destroy(); chart = null; }

  // ── Canvas-based candlestick renderer ──────────────────────────────
  // We draw candles directly on canvas via a Chart.js plugin for full control
  const N = bars.length;
  if(N < 2) return;

  // Price range with padding
  const allH = bars.map(b=>b.high);
  const allL = bars.map(b=>b.low);
  let yMax = Math.max(...allH);
  let yMin = Math.min(...allL);

  // Include signal levels in range
  if(sig && sig.direction !== 'WAIT'){
    yMax = Math.max(yMax, sig.tp2 || 0);
    yMin = Math.min(yMin, sig.sl  || yMin);
  }

  const pad  = (yMax - yMin) * 0.08;
  yMax += pad; yMin -= pad;

  // Draw function helpers
  function toY(price, h){ return h - ((price - yMin) / (yMax - yMin)) * h; }

  // Build overlay lines dataset (zones, entry, SL, TP)
  const overlayLines = [];

  if(sig){
    // OB zones
    (sig.zones || []).forEach(z => {
      if(z.color === 'green' && z.hi && z.lo){
        overlayLines.push({y: z.hi, color:'rgba(0,255,179,.5)', dash:[6,3], width:1, label:'OB'});
        overlayLines.push({y: z.lo, color:'rgba(0,255,179,.25)', dash:[3,4], width:1});
      }
      if(z.color === 'red' && z.hi && z.lo){
        overlayLines.push({y: z.hi, color:'rgba(255,61,90,.5)', dash:[6,3], width:1, label:'OB'});
        overlayLines.push({y: z.lo, color:'rgba(255,61,90,.25)', dash:[3,4], width:1});
      }
      if(z.color === 'amber' && z.hi && z.lo){
        const mid=(z.hi+z.lo)/2;
        overlayLines.push({y: mid, color:'rgba(255,184,0,.5)', dash:[4,3], width:1, label:'FVG'});
      }
      if(z.color === 'blue' && z.price){
        overlayLines.push({y: z.price, color:'rgba(61,159,255,.4)', dash:[4,4], width:1, label: z.type});
      }
    });
    // Signal levels
    if(sig.direction !== 'WAIT'){
      overlayLines.push({y:sig.entry, color:'rgba(61,159,255,.9)',  dash:[],    width:1.5, label:'Entry'});
      overlayLines.push({y:sig.sl,    color:'rgba(255,61,90,.9)',   dash:[5,3], width:1.5, label:'SL'});
      overlayLines.push({y:sig.tp1,   color:'rgba(0,255,179,.7)',   dash:[5,3], width:1.5, label:'TP1'});
      overlayLines.push({y:sig.tp2,   color:'rgba(0,255,179,.95)',  dash:[],    width:1.5, label:'TP2'});
    }
  }

  // Custom candlestick plugin
  const candlePlugin = {
    id: 'candles',
    beforeDraw(chartInst){
      const {ctx: c, chartArea: {left,right,top,bottom}, scales} = chartInst;
      const W = right - left;
      const H = bottom - top;
      const candleW = Math.max(2, Math.floor(W / N * 0.7));
      const gap     = W / N;

      c.save();
      c.beginPath();
      c.rect(left, top, W, H);
      c.clip();

      // Draw overlay lines behind candles
      overlayLines.forEach(line => {
        const ly = scales.y.getPixelForValue(line.y);
        if(ly < top || ly > bottom) return;
        c.beginPath();
        c.strokeStyle = line.color;
        c.lineWidth   = line.width || 1;
        c.setLineDash(line.dash || []);
        c.moveTo(left, ly);
        c.lineTo(right, ly);
        c.stroke();
        // Label on right edge
        if(line.label){
          c.setLineDash([]);
          c.fillStyle = line.color;
          c.font = '9px JetBrains Mono, monospace';
          c.textAlign = 'right';
          c.fillText(line.label, right - 2, ly - 3);
        }
      });

      // Draw candles
      bars.forEach((b, i) => {
        const x  = left + i * gap + gap / 2;
        const yO = scales.y.getPixelForValue(b.open);
        const yC = scales.y.getPixelForValue(b.close);
        const yH = scales.y.getPixelForValue(b.high);
        const yL = scales.y.getPixelForValue(b.low);
        const isBull = b.close >= b.open;
        const col    = isBull ? '#00ffb3' : '#ff3d5a';
        const bodyTop    = Math.min(yO, yC);
        const bodyBottom = Math.max(yO, yC);
        const bodyH      = Math.max(1, bodyBottom - bodyTop);

        // Wick
        c.beginPath();
        c.strokeStyle = isBull ? 'rgba(0,255,179,.6)' : 'rgba(255,61,90,.6)';
        c.lineWidth = 1;
        c.setLineDash([]);
        c.moveTo(x, yH);
        c.lineTo(x, yL);
        c.stroke();

        // Body
        c.fillStyle = isBull ? 'rgba(0,255,179,.85)' : 'rgba(255,61,90,.85)';
        c.strokeStyle = col;
        c.lineWidth = 0.5;
        c.beginPath();
        c.rect(x - candleW/2, bodyTop, candleW, bodyH);
        c.fill();
        c.stroke();
      });

      // ── Volume bars at bottom (10% of chart height) ──────────────
      const volH   = H * 0.12;
      const volTop = bottom - volH;
      const vols   = bars.map(b=>b.volume||0);
      const maxVol = Math.max(...vols, 1);
      c.globalAlpha = 0.45;
      bars.forEach((b,i)=>{
        const x   = left + i*gap + gap/2;
        const vh  = (b.volume||0) / maxVol * volH;
        const col = b.close >= b.open ? 'rgba(0,255,179,.6)' : 'rgba(255,61,90,.6)';
        c.fillStyle = col;
        c.fillRect(x - candleW/2, volTop + volH - vh, candleW, vh);
      });
      c.globalAlpha = 1.0;

      // Volume axis label
      c.fillStyle = '#2d404f';
      c.font = '8px JetBrains Mono, monospace';
      c.textAlign = 'left';
      c.fillText('Vol', left + 2, volTop + 10);

      c.restore();
    }
  };

  // Invisible dataset to set up the axes
  const pricePoints = bars.map((b,i) => ({x: i, y: (b.high+b.low)/2}));

  chart = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [{
        data: pricePoints,
        pointRadius: 0,
        showLine: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend:  { display: false },
        tooltip: { enabled: false },
        candles: {},
      },
      scales: {
        x: {
          type: 'linear',
          display: false,
          min: 0,
          max: N - 1,
        },
        y: {
          min: yMin,
          max: yMax,
          grid: { color: 'rgba(255,255,255,.04)', lineWidth: 0.5 },
          border: { color: 'rgba(255,255,255,.06)' },
          ticks: {
            color: '#2d404f',
            font: { size: 9, family: 'JetBrains Mono, monospace' },
            maxTicksLimit: 6,
            callback: v => typeof v === 'number' ? v.toFixed(dp > 3 ? 4 : dp < 3 ? 1 : 2) : v,
          }
        }
      }
    },
    plugins: [candlePlugin]
  });
}


// ── P&L History ───────────────────────────────────────────────
function buildHistory(history){
  const hl=document.getElementById('histList');
  if(!history||!history.length){
    hl.innerHTML='<div style="font-family:var(--mono);font-size:9px;color:var(--t3);padding:3px 0">No completed signals yet</div>';
    document.getElementById('hTotal').textContent='0';
    document.getElementById('hWR').textContent='—';
    document.getElementById('hNet').textContent='0';
    return;
  }
  const wins=history.filter(h=>h.result==='win').length;
  const total=wins+history.filter(h=>h.result==='loss').length;
  const net=history.reduce((a,h)=>a+(h.pnl_pips||0),0);
  const wr=total>0?Math.round(wins/total*100):0;
  document.getElementById('hTotal').textContent=total;
  const wrEl=document.getElementById('hWR'); wrEl.textContent=wr+'%'; wrEl.style.color=wr>=50?'var(--g)':'var(--r)';
  const nEl=document.getElementById('hNet'); nEl.textContent=(net>=0?'+':'')+net.toFixed(1)+'p'; nEl.style.color=net>=0?'var(--g)':'var(--r)';
  hl.innerHTML=history.map(h=>{
    const dp=DIGITS[h.pair]||5;
    const icon={tp2_hit:'✓✓',tp1_hit:'✓',sl_hit:'✗'}[h.status]||'?';
    const pnl=h.pnl_pips||0; const col=h.status==='sl_hit'?'var(--r)':pnl>0?'var(--g)':'var(--t2)';
    const t=h.hit_at?new Date(h.hit_at).toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit',hour12:false}):'—';
    return `<div style="display:flex;align-items:center;gap:5px;padding:4px 7px;border-radius:3px;border:1px solid var(--b);background:var(--bg2);font-family:var(--mono);font-size:9px">
      <span style="color:${h.direction==='BUY'?'var(--g)':'var(--r)'};font-weight:700;min-width:9px">${h.direction==='BUY'?'B':'S'}</span>
      <span style="color:var(--t);min-width:50px">${h.pair.slice(0,3)+'/'+h.pair.slice(3)}</span>
      <span style="color:${col};min-width:20px">${icon}</span>
      <span style="color:${col};flex:1;font-weight:500">${pnl>=0?'+':''}${pnl.toFixed(1)}p</span>
      <span style="color:var(--t3);font-size:8px">${t}</span>
    </div>`;
  }).join('');
}

// ── Grade display ────────────────────────────────────────────
function showGrade(grade, dir){
  const el=document.getElementById('gradeTag');
  if(!el||dir==='WAIT'){el&&(el.style.display='none');return;}
  el.style.display='block';
  const cfg={
    A:{bg:'rgba(0,255,179,.15)',color:'var(--g)',border:'rgba(0,255,179,.4)',label:'A'},
    B:{bg:'rgba(61,159,255,.12)',color:'var(--bl)',border:'rgba(61,159,255,.3)',label:'B'},
    C:{bg:'rgba(255,184,0,.1)',color:'var(--a)',border:'rgba(255,184,0,.25)',label:'C'},
  };
  const c=cfg[grade]||cfg.C;
  el.textContent=c.label;
  el.style.background=c.bg; el.style.color=c.color;
  el.style.border=`1px solid ${c.border}`;
}

// ── Stats polling ─────────────────────────────────────────────
async function pollStats(){
  try{
    const d=await(await fetch('/api/stats',{signal:AbortSignal.timeout(5000)})).json();
    const s=d.stats||{};
    // Pause banner
    const pb=document.getElementById('pauseBanner');
    const pr=document.getElementById('pauseReason');
    if(d.paused&&pb){
      pb.style.display='block';
      if(pr) pr.textContent=d.pause_reason||'Risk limit hit';
    } else if(pb){
      pb.style.display='none';
    }
    // Stats grid
    const wr=s.win_rate||0;
    const wrEl=document.getElementById('stWR');
    if(wrEl){wrEl.textContent=wr+'%';wrEl.style.color=wr>=50?'var(--g)':'var(--r)';}
    const net=s.net_pips||0;
    const netEl=document.getElementById('stNet');
    if(netEl){netEl.textContent=(net>=0?'+':'')+net+'p';netEl.style.color=net>=0?'var(--g)':'var(--r)';}
    const daily=s.daily_pnl_pips||0;
    const dailyEl=document.getElementById('stDaily');
    if(dailyEl){dailyEl.textContent=(daily>=0?'+':'')+daily.toFixed(1)+'p';dailyEl.style.color=daily>=0?'var(--g)':'var(--r)';}
    const cl=document.getElementById('stCL');
    if(cl){cl.textContent=s.consecutive_losses||0;cl.style.color=(s.consecutive_losses||0)>=2?'var(--r)':'var(--t2)';}
    const best=document.getElementById('stBest');
    if(best) best.textContent=(s.best_trade_pips||0)>0?'+'+(s.best_trade_pips||0).toFixed(1)+'p':'—';
    const worst=document.getElementById('stWorst');
    if(worst) worst.textContent=(s.worst_trade_pips||0)<0?(s.worst_trade_pips||0).toFixed(1)+'p':'—';
    // Also update P&L history from stats
    if(d.history_count) buildHistory(await getHistory());
    // Weekly performance
    buildWeekly(d.weekly||[]);
  }catch(e){}
}

async function getHistory(){
  try{
    const d=await(await fetch('/api/trade-state',{signal:AbortSignal.timeout(5000)})).json();
    return d.history||[];
  }catch(e){return[];}
}

function buildWeekly(weeks){
  const el=document.getElementById('weeklyList');
  if(!el) return;
  if(!weeks.length){el.innerHTML='<div style="color:var(--t3)">No weekly data yet</div>';return;}
  el.innerHTML=weeks.map(w=>{
    const pips=w.pips||0;
    const col=pips>=0?'var(--g)':'var(--r)';
    const wr=w.wins+w.losses>0?Math.round(w.wins/(w.wins+w.losses)*100):0;
    return `<div style="display:flex;align-items:center;gap:5px;padding:3px 0;border-bottom:1px solid var(--b)">
      <span style="color:var(--t3);min-width:60px">${w.week}</span>
      <span style="color:var(--t2);min-width:30px">${w.wins}W/${w.losses}L</span>
      <span style="color:var(--t2);min-width:28px">${wr}%</span>
      <span style="color:${col};flex:1;text-align:right;font-weight:500">${pips>=0?'+':''}${pips}p</span>
    </div>`;
  }).join('');
}

async function resetPause(){
  try{
    await fetch('/api/reset-pause',{method:'POST'});
    document.getElementById('pauseBanner').style.display='none';
    addLog('Trading pause reset manually','SYS','warn');
    showToast('Trading resumed','tb');
  }catch(e){}
}

setInterval(pollStats, 30000);

// ── Trade state polling ───────────────────────────────────────
async function pollTradeState(){
  try{
    const d=await(await fetch('/api/trade-state',{signal:AbortSignal.timeout(5000)})).json();
    buildHistory(d.history||[]);
    const st=d.active[pair];
    if(st){
      const sig=allSigs.find(s=>s.pair===pair);
      if(sig&&sig.live_pnl_pips!==undefined){
        const pnl=sig.live_pnl_pips;
        const el=document.getElementById('lvPnl');
        if(el){el.textContent=(pnl>=0?'+':'')+pnl.toFixed(1)+' pips';el.style.color=pnl>0?'var(--g)':pnl<0?'var(--r)':'var(--t2)';}
      }
    }
  }catch(e){}
}
setInterval(pollTradeState,15000);

// ── CSV export ────────────────────────────────────────────────
function exportCSV(){
  window.open('/api/export-history','_blank');
  addLog('Downloading signal history CSV...','SYS','info');
}

// ── Auto-scan countdown ───────────────────────────────────────
const AUTO_INTERVAL=300; let nextScanIn=AUTO_INTERVAL;
function updateCountdown(){
  nextScanIn=Math.max(0,nextScanIn-1);
  const m=Math.floor(nextScanIn/60), s=String(nextScanIn%60).padStart(2,'0');
  const el=document.getElementById('countdownEl'); if(el) el.textContent=`Next: ${m}:${s}`;
  if(nextScanIn===0){ nextScanIn=AUTO_INTERVAL; fetchLatestSignals(); }
}
setInterval(updateCountdown,1000);

async function fetchLatestSignals(){
  try{
    const d=await(await fetch('/api/scan',{signal:AbortSignal.timeout(30000)})).json();
    allSigs=d.signals||[]; buildTable(allSigs);
    allSigs.forEach(s=>{
      const dot=document.getElementById('sig_'+s.pair); if(dot) dot.className='pb-sig '+(s.direction==='BUY'?'pb-buy':s.direction==='SELL'?'pb-sell':'pb-wait');
    });
    const active=allSigs.find(s=>s.pair===pair);
    if(active){updateSignal(active);loadChart(pair);}
    const newSigs=allSigs.filter(s=>s.direction!=='WAIT'&&s.trade_status==='new');
    newSigs.forEach(s=>{
      playBeep(s.direction==='BUY'?880:440);
      showToast(`${s.direction==='BUY'?'📈':'📉'} NEW: ${s.pair.slice(0,3)+'/'+s.pair.slice(3)} ${s.direction} — ${s.score_pct}%`,s.direction==='BUY'?'tb':'ts');
      addLog(`AUTO: ${s.direction} ${s.score_pct}% | ${(s.conf||[]).join(', ')}`,s.pair,s.direction==='BUY'?'buy':'sell');
    });
    const ts=new Date(); const tsStr=String(ts.getUTCHours()).padStart(2,'0')+':'+String(ts.getUTCMinutes()).padStart(2,'0')+' UTC';
    document.getElementById('mLast').textContent=tsStr;
    const lsEl=document.getElementById('lastScanEl'); if(lsEl) lsEl.textContent='Last: '+tsStr;
    const tradeSigs=allSigs.filter(s=>s.direction!=='WAIT');
    document.getElementById('mSig').textContent=tradeSigs.length;
    document.getElementById('mBuy').textContent=allSigs.filter(s=>s.direction==='BUY').length;
    document.getElementById('mSell').textContent=allSigs.filter(s=>s.direction==='SELL').length;
  }catch(e){ addLog('Auto-fetch error: '+e.message,'SYS','warn'); }
}

// ── Log & Toast ───────────────────────────────────────────────
function addLog(msg,pair,type){
  const la=document.getElementById('logEl');
  const n=new Date(); const t=String(n.getUTCHours()).padStart(2,'0')+':'+String(n.getUTCMinutes()).padStart(2,'0');
  const pc=type==='buy'?'lg':type==='sell'?'lr2':type==='warn'?'la':'lb';
  const tm={buy:'ltb',sell:'lts',info:'lti',warn:'ltw'};
  const row=document.createElement('div'); row.className='lr';
  row.innerHTML=`<span class="lt">${t}</span><span class="lp ${pc}">${pair||'SYS'}</span><span class="lmsg">${msg}</span>${type!=='info'?`<span class="ltag ${tm[type]}">${type.toUpperCase()}</span>`:''}`;
  la.insertBefore(row,la.firstChild);
  if(la.children.length>80) la.removeChild(la.lastChild);
}
function showToast(msg,cls){
  const t=document.getElementById('toast'); t.textContent=msg; t.className=`toast show ${cls||''}`;
  setTimeout(()=>t.classList.remove('show'),5000);
}

// ── Timeframe switcher ───────────────────────────────────────
let currentTF = 'H1';
function switchTF(btn, tf){
  currentTF = tf;
  document.querySelectorAll('.tf-btn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  loadChart(pair, tf);
}

// ── Volume bar renderer (added to candlestick plugin) ─────────
// Integrated into buildChart below — uses canvas bottom area

// ── Pip value & lot size reference table ─────────────────────
const PIP_VALUES = {
  EURUSD:{std:10,mini:1,micro:0.1,unit:'USD'},
  GBPUSD:{std:10,mini:1,micro:0.1,unit:'USD'},
  USDJPY:{std:9.3,mini:0.93,micro:0.093,unit:'USD'},
  GBPJPY:{std:7.8,mini:0.78,micro:0.078,unit:'USD'},
  AUDUSD:{std:10,mini:1,micro:0.1,unit:'USD'},
  USDCAD:{std:7.7,mini:0.77,micro:0.077,unit:'USD'},
  XAUUSD:{std:10,mini:1,micro:0.1,unit:'USD'},
};

function calcRisk(){
  const bal=parseFloat(document.getElementById('rcBalance').value)||10000;
  const riskPct=parseFloat(document.getElementById('rcRisk').value)||1;
  const riskUSD=bal*riskPct/100;
  const pv=PIP_VALUES[pair]||{std:10};
  const slPips=parseInt(document.getElementById('rcSLPips')?.value||30);
  const lots=Math.max(0.01,Math.min((riskUSD/(slPips*pv.std)).toFixed(2),100));
  document.getElementById('rcDollar').textContent='$'+riskUSD.toFixed(2);
  document.getElementById('rcLots').textContent=lots;
  document.getElementById('rcSL').textContent='$'+(lots*slPips*pv.std).toFixed(2);
  document.getElementById('rcTP1').textContent='$'+(lots*slPips*1.5*pv.std).toFixed(2);
  // pip value display
  const pvEl=document.getElementById('rcPipVal');
  if(pvEl) pvEl.textContent='$'+pv.std+'/pip (std lot)';
}

// ── Local timezone display ────────────────────────────────────
function updateLocalTime(){
  const el=document.getElementById('localTimeEl');
  if(!el) return;
  const n=new Date();
  el.textContent='Local: '+n.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
}
setInterval(updateLocalTime,1000);

// ── Killzone local time helper ────────────────────────────────
function getKillzoneLocal(){
  const now=new Date();
  const h=now.getUTCHours();
  const m=now.getUTCMinutes();
  const t=h*60+m;
  if(t>=420&&t<600) return {name:'London',ends:600};
  if(t>=720&&t<900) return {name:'New York',ends:900};
  if(t>=1320||t<60) return {name:'Asian',ends:60};
  // Calculate time to next killzone
  const next=[420,720,1320];
  const diff=next.map(s=>s>t?s-t:1440-t+s);
  const mins=Math.min(...diff);
  return {name:null,next_in_mins:mins};
}

// ── Browser push notification permission ──────────────────────
async function requestNotifyPermission(){
  if(!('Notification' in window)) return;
  if(Notification.permission==='default'){
    const p=await Notification.requestPermission();
    if(p==='granted') addLog('Push notifications enabled','SYS','info');
  }
}

function sendPushNotif(title, body){
  if(!('Notification' in window)||Notification.permission!=='granted') return;
  new Notification(title,{body,icon:'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="16" fill="%2300ffb3"/><text x="50%" y="50%" text-anchor="middle" dy=".35em" font-size="16">📈</text></svg>'});
}

// ── Init ──────────────────────────────────────────────────────
(async function(){
  addLog('Dashboard ready — auto-scan every 5 min','SYS','info');
  requestNotifyPermission();
  updateLocalTime();
  loadPrices();
  const st=await checkStatus();
  if(st&&st.is_open){
    addLog('Market OPEN — running initial scan...','SYS','info');
    loadChart('EURUSD');
    await doScan(false);
  } else if(st&&st.is_closed){
    addLog('Market closed: '+st.reason,'SYS','warn');
    loadChart('EURUSD');
  }
  pollTradeState();
  pollStats();
})();
setInterval(checkStatus,60000);
setInterval(loadPrices,60000);
</script>
</body>
</html>
"""

def open_browser():
    """Only opens browser when running locally, not on Render."""
    if os.environ.get("RENDER"):
        return
    time.sleep(1.2)
    webbrowser.open("http://localhost:5000")


AUTO_SCAN_INTERVAL = 300   # seconds — auto scan every 5 minutes

def auto_scan_loop():
    """Background thread: runs a scan every AUTO_SCAN_INTERVAL seconds."""
    log.info(f"Auto-scan started — every {AUTO_SCAN_INTERVAL}s")
    time.sleep(15)   # wait for server to fully start first
    while True:
        try:
            ms = market_status()
            log.info(f"[AUTO-SCAN] Running... market={'OPEN' if ms['is_open'] else 'CLOSED'}")
            run_scan(force=False)
            log.info(f"[AUTO-SCAN] Done. Next in {AUTO_SCAN_INTERVAL}s")
        except Exception as e:
            log.error(f"[AUTO-SCAN] Error: {e}")
        time.sleep(AUTO_SCAN_INTERVAL)


if __name__ == "__main__":
    ms = market_status()
    print("""
╔══════════════════════════════════════════╗
║     SMC Signal Dashboard  — v4.0        ║
║     Signals only · No MT5 trading       ║
╚══════════════════════════════════════════╝""")
    print(f"\n  Status  : {'✓ OPEN' if ms['is_open'] else '✗ CLOSED — '+ms['reason']}")
    print(f"  Sessions : {', '.join(ms['sessions']) or 'None active'}")
    print(f"  Time UTC : {ms['time_utc']}")
    print(f"\n  Opening  : http://localhost:5000\n")
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    # Start auto-scan background thread
    scanner = threading.Thread(target=auto_scan_loop, daemon=True)
    scanner.start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
