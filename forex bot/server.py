"""
SMC Signal Dashboard — All-in-One Server
==========================================
Just run:  python server.py
Then open: http://localhost:5000

No separate frontend files needed.
Everything is embedded in this single file.

Install deps first (one time only):
    pip install flask flask-cors yfinance pandas numpy
"""

import time
import logging
import webbrowser
import threading
from datetime import datetime, timezone
from typing import Optional

# ── Dependency check ─────────────────────────────────────────────────
try:
    from flask import Flask, jsonify, request, Response
    from flask_cors import CORS
    import yfinance as yf
    import pandas as pd
    import numpy as np
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    missing = str(e).replace("No module named ", "")
    print(f"\n[ERROR] Missing: {missing}")
    print("Run this first:\n  pip install flask flask-cors yfinance pandas numpy\n")
    exit(1)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("SMC")

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "XAUUSD": "GC=F",
}

PIP = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01,   "GBPJPY": 0.01,   "EURJPY": 0.01,
    "XAUUSD": 0.1,
}

DIGITS = {
    "EURUSD": 5, "GBPUSD": 5, "AUDUSD": 5, "USDCAD": 5,
    "USDJPY": 3, "GBPJPY": 3, "EURJPY": 3,
    "XAUUSD": 2,
}

TF_INTERVAL = {"M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}
TF_PERIOD   = {"M15": "5d",  "M30": "7d",  "H1": "30d","H4": "60d","D1": "1y"}

_cache: dict = {}
CACHE_TTL = 300  # seconds

# ══════════════════════════════════════════════════════════════════════
#  MARKET HOURS
# ══════════════════════════════════════════════════════════════════════

SESSIONS = {
    "Sydney":   (21, 6),
    "Tokyo":    (0,  9),
    "London":   (7,  16),
    "New York": (12, 21),
}

def market_status() -> dict:
    now     = datetime.now(timezone.utc)
    weekday = now.weekday()   # 0=Mon, 6=Sun
    hour    = now.hour

    closed = False
    reason = ""

    if weekday == 4 and hour >= 21:
        closed = True
        reason = "Forex closed — Weekend starts Friday 21:00 UTC"
    elif weekday == 5:
        closed = True
        reason = "Forex closed — Saturday"
    elif weekday == 6 and hour < 21:
        closed = True
        reason = "Forex closed — Sunday (opens 21:00 UTC)"

    if (now.month, now.day) in [(12, 25), (1, 1)]:
        closed = True
        reason = "Forex closed — Public holiday"

    active = []
    for name, (o, c) in SESSIONS.items():
        if o < c:
            if o <= hour < c: active.append(name)
        else:
            if hour >= o or hour < c: active.append(name)

    hours_to_open = ""
    if closed:
        if weekday == 6:
            h = 21 - hour if hour < 21 else 0
            hours_to_open = f"Opens in {h}h {now.minute}m"
        elif weekday == 5:
            h = (24 - hour) + 21
            hours_to_open = f"Opens in ~{h}h (Sunday 21:00 UTC)"
        elif weekday == 4:
            h = (24 - hour) + 45
            hours_to_open = f"Opens in ~{h}h (Sunday 21:00 UTC)"

    return {
        "is_open":          not closed,
        "is_closed":        closed,
        "reason":           reason,
        "active_sessions":  active,
        "next_open":        hours_to_open,
        "server_time_utc":  now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "weekday":          now.strftime("%A"),
        "hour_utc":         hour,
    }

# ══════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════

def fetch_bars(pair: str, tf: str = "H1") -> Optional[list]:
    key = f"{pair}_{tf}"
    cached = _cache.get(key)
    if cached and time.time() - cached["ts"] < CACHE_TTL:
        return cached["bars"]

    ticker   = PAIRS.get(pair)
    interval = TF_INTERVAL.get(tf, "1h")
    period   = TF_PERIOD.get(tf, "30d")

    if not ticker:
        return None

    try:
        log.info(f"Fetching {pair} ({ticker}) {interval} / {period}")
        df = yf.download(ticker, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()
        bars = []
        for ts, row in df.iterrows():
            bars.append({
                "time":   int(pd.Timestamp(ts).timestamp()),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(row.get("Volume", 0)),
            })
        _cache[key] = {"bars": bars, "ts": time.time()}
        log.info(f"  Got {len(bars)} bars for {pair} {tf}")
        return bars
    except Exception as e:
        log.error(f"Fetch error {pair}: {e}")
        return None


# ── Live price cache ──────────────────────────────────────────────────
_price_cache: dict = {}
_price_cache_ts: float = 0.0
PRICE_TTL = 60

def fetch_live_prices() -> dict:
    global _price_cache, _price_cache_ts
    if _price_cache and time.time() - _price_cache_ts < PRICE_TTL:
        return _price_cache
    prices = {}
    for pair, ticker in PAIRS.items():
        try:
            df = yf.download(ticker, period="2d", interval="1h",
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if df.empty:
                continue
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            prices[pair] = {
                "price":      round(float(last["Close"]), 5),
                "open":       round(float(last["Open"]), 5),
                "high":       round(float(last["High"]), 5),
                "low":        round(float(last["Low"]), 5),
                "prev_close": round(float(prev["Close"]), 5),
                "change_pct": round((float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100, 3) if float(prev["Close"]) != 0 else 0,
            }
            log.info(f"  Live {pair}: {prices[pair]['price']}")
        except Exception as e:
            log.warning(f"  Price fail {pair}: {e}")
    if prices:
        _price_cache = prices
        _price_cache_ts = time.time()
    return prices


def fetch_demo_bars(pair: str, n: int = 80) -> list:
    import random
    prices = fetch_live_prices()
    base = prices.get(pair, {}).get("price", 0)
    fallback = {"EURUSD":1.0855,"GBPUSD":1.2685,"USDJPY":149.82,"GBPJPY":192.30,"AUDUSD":0.6512,"USDCAD":1.3810,"XAUUSD":2321.0}
    if base == 0:
        base = fallback.get(pair, 1.0)
    vol_map = {"EURUSD":0.0008,"GBPUSD":0.0012,"USDJPY":0.12,"GBPJPY":0.18,"AUDUSD":0.0007,"USDCAD":0.0009,"XAUUSD":2.8}
    vol = vol_map.get(pair, base * 0.001)
    rng = random.Random(int(time.time() / 3600))
    bars = []
    price = base * (1 + rng.uniform(-0.003, 0.003))
    for i in range(n):
        drift = (base - price) * 0.03
        o = price
        c = o + drift + rng.gauss(0, vol)
        h = max(o, c) + abs(rng.gauss(0, vol * 0.4))
        l = min(o, c) - abs(rng.gauss(0, vol * 0.4))
        bars.append({"time": int(time.time()) - (n - i) * 3600,
                     "open": round(o, 5), "high": round(h, 5),
                     "low": round(l, 5), "close": round(c, 5),
                     "volume": rng.randint(500, 5000)})
        price = c
    if bars:
        bars[-1]["close"] = base
        bars[-1]["high"]  = max(bars[-1]["high"], base)
        bars[-1]["low"]   = min(bars[-1]["low"],  base)
    return bars


# ══════════════════════════════════════════════════════════════════════
#  SMC ENGINE
# ══════════════════════════════════════════════════════════════════════

def detect_swings(bars: list, n: int = 5) -> list:
    swings = []
    prev_sh = prev_sl = None
    for i in range(n, len(bars) - n):
        h = bars[i]["high"]
        l = bars[i]["low"]
        is_h = all(bars[j]["high"] < h for j in range(i-n, i+n+1) if j != i)
        is_l = all(bars[j]["low"]  > l for j in range(i-n, i+n+1) if j != i)
        if is_h:
            kind = "HH" if (prev_sh is None or h > prev_sh) else "LH"
            swings.append({"idx": i, "price": h, "kind": kind, "is_high": True})
            prev_sh = h
        if is_l:
            kind = "HL" if (prev_sl is None or l > prev_sl) else "LL"
            swings.append({"idx": i, "price": l, "kind": kind, "is_high": False})
            prev_sl = l
    return sorted(swings, key=lambda s: s["idx"])

def get_trend(swings: list) -> str:
    if len(swings) < 4: return "ranging"
    r = swings[-8:]
    bull = sum(1 for s in r if s["kind"] in ("HH","HL"))
    bear = sum(1 for s in r if s["kind"] in ("LH","LL"))
    if bull > bear + 1: return "bullish"
    if bear > bull + 1: return "bearish"
    return "ranging"

def get_structure(swings: list, trend: str) -> dict:
    choch = bos = False
    choch_dir = bos_dir = ""
    for s in swings[-6:]:
        k = s["kind"]
        if k == "HH" and trend == "bearish": choch = True; choch_dir = "bullish"
        if k == "LL" and trend == "bullish": choch = True; choch_dir = "bearish"
        if k == "HH" and trend == "bullish": bos   = True; bos_dir   = "bullish"
        if k == "LL" and trend == "bearish": bos   = True; bos_dir   = "bearish"
    return {"choch": choch, "choch_dir": choch_dir, "bos": bos, "bos_dir": bos_dir}

def detect_obs(bars: list, min_body=0.55, lookback=50) -> list:
    obs = []
    lim = min(lookback, len(bars) - 4)
    for i in range(2, lim):
        b    = bars[i]
        body = abs(b["close"] - b["open"])
        rng  = b["high"] - b["low"]
        if rng == 0 or body / rng < min_body: continue
        after = bars[max(i-3, 0):i]
        if b["close"] < b["open"]:
            mx = max((x["close"] for x in after), default=0)
            if rng > 0 and (mx - b["close"]) / rng >= 1.5:
                obs.append({"hi": b["high"], "lo": b["low"], "mid": (b["high"]+b["low"])/2,
                             "bull": True, "tapped": False})
        elif b["close"] > b["open"]:
            mn = min((x["close"] for x in after), default=float("inf"))
            if rng > 0 and (b["close"] - mn) / rng >= 1.5:
                obs.append({"hi": b["high"], "lo": b["low"], "mid": (b["high"]+b["low"])/2,
                             "bull": False, "tapped": False})
    return obs[-8:]

def detect_fvgs(bars: list, pip=0.0001) -> list:
    fvgs = []
    lim  = min(60, len(bars) - 2)
    for i in range(2, lim):
        if bars[i-2]["low"] > bars[i]["high"]:
            gap = bars[i-2]["low"] - bars[i]["high"]
            if gap / pip >= 2.0:
                fvgs.append({"hi": bars[i-2]["low"], "lo": bars[i]["high"],
                              "mid": (bars[i-2]["low"]+bars[i]["high"])/2, "bull": True})
        if bars[i-2]["high"] < bars[i]["low"]:
            gap = bars[i]["low"] - bars[i-2]["high"]
            if gap / pip >= 2.0:
                fvgs.append({"hi": bars[i]["low"], "lo": bars[i-2]["high"],
                              "mid": (bars[i]["low"]+bars[i-2]["high"])/2, "bull": False})
    return fvgs[-8:]

def smc_analyse(bars: list, pair: str) -> dict:
    if not bars or len(bars) < 50:
        return {}
    pip  = PIP.get(pair, 0.0001)
    rev  = list(reversed(bars))
    price = rev[0]["close"]
    lb   = min(100, len(rev))
    r_hi = max(b["high"] for b in rev[:lb])
    r_lo = min(b["low"]  for b in rev[:lb])
    mid  = (r_hi + r_lo) / 2
    swings  = detect_swings(rev)
    trend   = get_trend(swings)
    struct  = get_structure(swings, trend)
    obs     = detect_obs(rev)
    for ob in obs:
        if ob["lo"] <= price <= ob["hi"]:
            ob["tapped"] = True
    fvgs    = detect_fvgs(rev, pip=pip)
    near_fvgs = [f for f in fvgs if abs(f["mid"] - price) / pip < 100]
    liq = [{"price": s["price"], "type": "BSL" if s["is_high"] else "SSL"}
           for s in swings[-12:]]
    return {
        "price": price, "trend": trend, "mid": mid,
        "in_discount": price < mid, "in_premium": price > mid,
        "choch": struct["choch"], "choch_dir": struct["choch_dir"],
        "bos":   struct["bos"],   "bos_dir":   struct["bos_dir"],
        "obs": obs, "tapped_obs": [o for o in obs if o["tapped"]],
        "fvgs": fvgs, "near_fvgs": near_fvgs, "liq": liq,
    }

def score_dir(buy: bool, htf: dict, mtf: dict, ltf: dict) -> tuple:
    s = 0.0; conf = []
    if buy  and htf.get("trend") == "bullish": s += 0.20; conf.append("HTF↑")
    if not buy and htf.get("trend") == "bearish": s += 0.20; conf.append("HTF↓")
    if any(o["bull"] == buy for o in mtf.get("tapped_obs", [])): s += 0.20; conf.append("OB_tap")
    if any(f["bull"] == buy for f in mtf.get("near_fvgs",  [])): s += 0.15; conf.append("FVG")
    if mtf.get("choch") and mtf["choch_dir"] == ("bullish" if buy else "bearish"):
        s += 0.20; conf.append("CHoCH")
    elif ltf.get("choch") and ltf["choch_dir"] == ("bullish" if buy else "bearish"):
        s += 0.14; conf.append("LTF_CHoCH")
    if mtf.get("bos") and mtf["bos_dir"] == ("bullish" if buy else "bearish"):
        s += 0.10; conf.append("BOS")
    if any(l["type"] == ("SSL" if buy else "BSL") for l in mtf.get("liq", [])):
        s += 0.10; conf.append("Liq_pool")
    if buy  and mtf.get("in_discount"): s += 0.10; conf.append("Discount")
    if not buy and mtf.get("in_premium"):  s += 0.10; conf.append("Premium")
    if ltf.get("choch") and ltf["choch_dir"] == ("bullish" if buy else "bearish"):
        s += 0.10; conf.append("LTF↑" if buy else "LTF↓")
    return min(s, 1.0), conf

def make_signal(pair: str, htf: dict, mtf: dict, ltf: dict) -> dict:
    pip    = PIP.get(pair, 0.0001)
    dp     = DIGITS.get(pair, 5)
    price  = mtf.get("price", 0)
    bs, bc = score_dir(True,  htf, mtf, ltf)
    ss, sc = score_dir(False, htf, mtf, ltf)
    if bs >= ss and bs >= 0.50:        direction, score, conf = "BUY",  bs, bc
    elif ss > bs and ss >= 0.50:       direction, score, conf = "SELL", ss, sc
    else:                              direction, score, conf = "WAIT", max(bs,ss), []
    sl_d  = pip * 20
    entry = round(price, dp)
    sl    = round(price - sl_d if direction == "BUY" else price + sl_d, dp)
    tp1   = round(price + sl_d*1.5 if direction == "BUY" else price - sl_d*1.5, dp)
    tp2   = round(price + sl_d*3.0 if direction == "BUY" else price - sl_d*3.0, dp)
    rr    = round(abs(tp1-entry)/abs(entry-sl), 2) if abs(entry-sl) > 0 else 0

    # Zones for frontend
    zones = []
    for ob in mtf.get("obs", [])[-3:]:
        zones.append({"type": "Bull OB" if ob["bull"] else "Bear OB",
                      "hi": round(ob["hi"],dp), "lo": round(ob["lo"],dp),
                      "mid": round(ob["mid"],dp),
                      "status": "Tapped" if ob["tapped"] else "Untouched",
                      "color": "green" if ob["bull"] else "red"})
    for fvg in mtf.get("near_fvgs", [])[:2]:
        zones.append({"type": "Bull FVG" if fvg["bull"] else "Bear FVG",
                      "hi": round(fvg["hi"],dp), "lo": round(fvg["lo"],dp),
                      "mid": round(fvg["mid"],dp), "status": "Unfilled", "color": "amber"})
    for lq in mtf.get("liq", [])[-2:]:
        zones.append({"type": lq["type"], "price": round(lq["price"],dp),
                      "status": "Target", "color": "blue"})

    return {
        "pair": pair, "direction": direction,
        "score": round(score,3), "score_pct": round(score*100),
        "conf": conf, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr,
        "valid": direction != "WAIT" and rr >= 2.0,
        "buy_score": round(bs,3), "sell_score": round(ss,3),
        "trend": mtf.get("trend","ranging"),
        "choch": mtf.get("choch",False), "choch_dir": mtf.get("choch_dir",""),
        "bos":   mtf.get("bos",  False), "bos_dir":   mtf.get("bos_dir",  ""),
        "in_discount": mtf.get("in_discount",False),
        "in_premium":  mtf.get("in_premium", False),
        "ob_count":  len(mtf.get("obs",[])),
        "fvg_count": len(mtf.get("near_fvgs",[])),
        "zones": zones,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


@app.route("/api/live-prices")
def api_live_prices():
    """Returns real current prices for all pairs — works even when market is closed."""
    prices = fetch_live_prices()
    ms = market_status()
    return jsonify({
        "prices": prices,
        "market_open": ms["is_open"],
        "source": "yfinance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/market-status")
def api_market_status():
    return jsonify(market_status())

@app.route("/api/bars/<pair>")
def api_bars(pair: str):
    tf    = request.args.get("tf", "H1")
    limit = int(request.args.get("limit", 100))
    bars  = fetch_bars(pair.upper(), tf)
    if bars is None:
        return jsonify({"error": f"No data for {pair}"}), 503
    return jsonify({"pair": pair, "tf": tf, "bars": bars[-limit:]})

@app.route("/api/scan")
def api_scan():
    ms = market_status()
    if ms["is_closed"]:
        # Market closed — fetch real latest prices and run SMC on demo bars
        log.info("Market closed — fetching real latest prices for demo signals")
        signals = []
        for pair in PAIRS:
            try:
                bars = fetch_demo_bars(pair, n=100)
                if not bars:
                    continue
                mtf = smc_analyse(bars, pair)
                htf = smc_analyse(bars, pair)
                ltf = smc_analyse(bars, pair)
                sig = make_signal(pair, htf, mtf, ltf)
                sig["demo"] = True
                signals.append(sig)
                log.info(f"  Demo {pair}: {sig['direction']} {sig['score_pct']}% price={sig['price']}")
            except Exception as e:
                log.error(f"  Demo {pair}: {e}")
        return jsonify({
            "market_closed": True,
            "demo_mode": True,
            "reason": ms["reason"],
            "next_open": ms["next_open"],
            "active_sessions": ms["active_sessions"],
            "server_time_utc": ms["server_time_utc"],
            "weekday": ms["weekday"],
            "signals": signals,
        })
    signals = []
    for pair in PAIRS:
        try:
            bh = fetch_bars(pair, "H4")
            bm = fetch_bars(pair, "H1")
            bl = fetch_bars(pair, "M15")
            if not bm: continue
            htf = smc_analyse(bh or bm, pair)
            mtf = smc_analyse(bm, pair)
            ltf = smc_analyse(bl or bm, pair)
            sig = make_signal(pair, htf, mtf, ltf)
            signals.append(sig)
            log.info(f"  {pair}: {sig['direction']} {sig['score_pct']}% | {sig['conf']}")
        except Exception as e:
            log.error(f"  {pair} error: {e}")
    return jsonify({
        "market_closed": False,
        "active_sessions": ms["active_sessions"],
        "server_time_utc": ms["server_time_utc"],
        "signals": signals,
        "scan_count": len(signals),
    })

@app.route("/api/signal/<pair>")
def api_signal(pair: str):
    pair = pair.upper()
    ms   = market_status()
    if ms["is_closed"]:
        return jsonify({"pair": pair, "market_closed": True,
                        "reason": ms["reason"], "next_open": ms["next_open"]})
    bh = fetch_bars(pair, "H4")
    bm = fetch_bars(pair, "H1")
    bl = fetch_bars(pair, "M15")
    if not bm:
        return jsonify({"error": f"No data for {pair}"}), 503
    htf = smc_analyse(bh or bm, pair)
    mtf = smc_analyse(bm, pair)
    ltf = smc_analyse(bl or bm, pair)
    return jsonify(make_signal(pair, htf, mtf, ltf))

# ══════════════════════════════════════════════════════════════════════
#  FRONTEND — served at /
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

# ── The entire frontend HTML (API calls go to /api/*) ────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMC Signal Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg0:#050709;--bg1:#090d12;--bg2:#0d1219;--bg3:#111a24;
  --border:rgba(255,255,255,0.06);--border2:rgba(255,255,255,0.12);
  --green:#00ffb3;--red:#ff3d5a;--amber:#ffb800;--blue:#3d9fff;
  --green-a:rgba(0,255,179,0.08);--red-a:rgba(255,61,90,0.08);
  --amber-a:rgba(255,184,0,0.08);--blue-a:rgba(61,159,255,0.08);
  --text:#dde6f0;--text2:#6b8099;--text3:#2d404f;
  --mono:'JetBrains Mono',monospace;
  --display:'Bebas Neue',sans-serif;
  --body:'DM Sans',sans-serif;
}
html{scroll-behavior:smooth}
body{background:var(--bg0);color:var(--text);font-family:var(--body);font-size:14px;overflow-x:hidden;min-height:100vh}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.025) 2px,rgba(0,0,0,0.025) 4px);pointer-events:none;z-index:0}
::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
.shell{position:relative;z-index:1;display:grid;grid-template-rows:56px 1fr;min-height:100vh}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;background:rgba(5,7,9,.92);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:50}
.logo{font-family:var(--display);font-size:22px;letter-spacing:.05em}
.logo em{color:var(--green);font-style:normal}
.top-right{display:flex;align-items:center;gap:1rem}
.status-badge{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:10px;letter-spacing:.1em;padding:4px 12px;border-radius:3px;border:1px solid}
.badge-open{color:var(--green);border-color:rgba(0,255,179,.25);background:var(--green-a)}
.badge-closed{color:var(--amber);border-color:rgba(255,184,0,.25);background:var(--amber-a)}
.badge-dot{width:5px;height:5px;border-radius:50%;animation:blink 1.2s infinite}
.badge-open .badge-dot{background:var(--green)}.badge-closed .badge-dot{background:var(--amber)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.clock{font-family:var(--mono);font-size:11px;color:var(--text2)}
.scan-btn{font-family:var(--mono);font-size:10px;letter-spacing:.08em;padding:6px 18px;border-radius:3px;cursor:pointer;background:var(--green);color:#000;border:none;font-weight:700;transition:opacity .15s;text-transform:uppercase}
.scan-btn:hover{opacity:.8}.scan-btn:disabled{opacity:.35;cursor:default}
.main{display:grid;grid-template-columns:320px 1fr;height:calc(100vh - 56px)}
@media(max-width:860px){.main{grid-template-columns:1fr;height:auto}}
.left{border-right:1px solid var(--border);overflow-y:auto;background:var(--bg1);display:flex;flex-direction:column}
.sect{padding:.9rem 1.1rem;border-bottom:1px solid var(--border)}
.sect-label{font-family:var(--mono);font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--text3);margin-bottom:.6rem}
.pair-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}
.pair-btn{font-family:var(--mono);font-size:11px;font-weight:500;padding:7px 4px;border-radius:4px;border:1px solid var(--border);background:var(--bg2);color:var(--text2);cursor:pointer;text-align:center;transition:all .15s}
.pair-btn:hover{color:var(--text);border-color:var(--border2)}
.pair-btn.active{background:var(--green-a);border-color:rgba(0,255,179,.3);color:var(--green)}
.score-num{font-family:var(--display);font-size:52px;line-height:1;letter-spacing:.02em;margin-bottom:.3rem}
.score-num.bull{color:var(--green)}.score-num.bear{color:var(--red)}.score-num.wait{color:var(--amber)}
.bar-bg{height:5px;background:var(--bg3);border-radius:3px;overflow:hidden;margin-bottom:.4rem}
.bar-fill{height:100%;border-radius:3px;transition:width .5s ease,background .3s}
.bar-labels{display:flex;justify-content:space-between;font-family:var(--mono);font-size:8px;color:var(--text3)}
.sig-dir{font-family:var(--display);font-size:36px;letter-spacing:.05em;margin-bottom:.5rem}
.sig-dir.BUY{color:var(--green)}.sig-dir.SELL{color:var(--red)}.sig-dir.WAIT{color:var(--amber)}
.lvl-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-bottom:.7rem}
.lvl{background:var(--bg3);border-radius:4px;padding:7px 9px}
.lvl-k{font-family:var(--mono);font-size:8px;color:var(--text3);letter-spacing:.1em;text-transform:uppercase;margin-bottom:2px}
.lvl-v{font-family:var(--mono);font-size:13px;font-weight:500}
.lv-e{color:var(--blue)}.lv-s{color:var(--red)}.lv-t{color:var(--green)}
.ctags{display:flex;flex-wrap:wrap;gap:4px}
.ctag{font-family:var(--mono);font-size:9px;letter-spacing:.06em;padding:2px 7px;border-radius:3px;background:var(--green-a);color:var(--green);border:1px solid rgba(0,255,179,.2)}
.ctag.bear{background:var(--red-a);color:var(--red);border-color:rgba(255,61,90,.2)}
.ctag.neutral{background:var(--bg3);color:var(--text2);border-color:var(--border)}
.zone-list{display:flex;flex-direction:column;gap:4px}
.zone-row{display:flex;align-items:center;gap:7px;padding:6px 9px;border-radius:4px;border:1px solid var(--border);background:var(--bg2)}
.zd{width:7px;height:7px;border-radius:2px;flex-shrink:0}
.zd-g{background:var(--green)}.zd-r{background:var(--red)}.zd-a{background:var(--amber)}.zd-b{background:var(--blue)}
.zn{font-family:var(--mono);font-size:10px;font-weight:500;min-width:72px;color:var(--text)}
.zp{font-family:var(--mono);font-size:10px;color:var(--text2);flex:1}
.zs{font-family:var(--mono);font-size:9px}
.zs-g{color:var(--green)}.zs-a{color:var(--amber)}.zs-x{color:var(--text3)}
.met-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}
.met{background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:.55rem .75rem}
.met-k{font-family:var(--mono);font-size:8px;letter-spacing:.12em;text-transform:uppercase;color:var(--text3);margin-bottom:2px}
.met-v{font-family:var(--mono);font-size:15px;font-weight:500}
.mv-g{color:var(--green)}.mv-r{color:var(--red)}.mv-a{color:var(--amber)}
.right{display:flex;flex-direction:column;background:var(--bg0);overflow:hidden}
.chart-area{flex:1;padding:1.1rem 1.4rem .5rem;display:flex;flex-direction:column;min-height:0}
.chart-top{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.9rem}
.chart-sym{font-family:var(--display);font-size:30px;letter-spacing:.04em}
.chart-meta{font-family:var(--mono);font-size:10px;color:var(--text2);display:flex;gap:.8rem;margin-top:2px;flex-wrap:wrap}
.chart-px{font-family:var(--mono);font-size:14px;font-weight:500;text-align:right}
.px-up{color:var(--green)}.px-dn{color:var(--red)}
.chart-wrap{flex:1;position:relative;min-height:220px}
.tbl-wrap{padding:.5rem 1.4rem .8rem;border-top:1px solid var(--border)}
.tbl-hdr{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--text3);padding:.5rem 0 .4rem}
table{width:100%;border-collapse:collapse}
th{font-family:var(--mono);font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--text3);padding:5px 7px;text-align:left;border-bottom:1px solid var(--border)}
td{font-family:var(--mono);font-size:11px;padding:7px 7px;border-bottom:1px solid var(--border);color:var(--text2)}
tr:last-child td{border-bottom:none}tr:hover td{background:var(--bg2)}
.d-buy{color:var(--green);font-weight:700}.d-sell{color:var(--red);font-weight:700}.d-wait{color:var(--amber)}
.sp{font-size:9px;padding:2px 6px;border-radius:2px}
.sp-h{background:var(--green-a);color:var(--green)}.sp-m{background:var(--amber-a);color:var(--amber)}.sp-l{background:var(--red-a);color:var(--red)}
.log-area{height:160px;border-top:1px solid var(--border);overflow-y:auto;padding:.65rem 1.4rem;background:var(--bg1);display:flex;flex-direction:column;gap:3px}
.lr{display:flex;gap:9px;font-family:var(--mono);font-size:10px;line-height:1.7;align-items:baseline}
.lt{color:var(--text3);min-width:42px}.lp{min-width:50px;font-weight:500}
.lp-g{color:var(--green)}.lp-r{color:var(--red)}.lp-b{color:var(--blue)}.lp-a{color:var(--amber)}
.lm{color:var(--text2);flex:1}
.ltag{font-size:9px;padding:1px 5px;border-radius:2px;margin-left:3px}
.ltag-b{background:var(--green-a);color:var(--green)}.ltag-s{background:var(--red-a);color:var(--red)}
.ltag-i{background:var(--blue-a);color:var(--blue)}.ltag-w{background:var(--amber-a);color:var(--amber)}
/* Market Closed */
.closed{display:none;position:fixed;inset:0;z-index:100;background:var(--bg0);flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:1.25rem}
.closed.show{display:flex}
.closed-icon{font-size:56px}
.closed-title{font-family:var(--display);font-size:52px;letter-spacing:.05em;color:var(--amber)}
.closed-reason{font-family:var(--mono);font-size:12px;color:var(--text2);max-width:440px;line-height:1.9}
.closed-next{font-family:var(--mono);font-size:12px;color:var(--green);padding:8px 22px;border:1px solid rgba(0,255,179,.3);border-radius:4px;background:var(--green-a)}
.closed-time{font-family:var(--mono);font-size:10px;color:var(--text3)}
.sess-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;max-width:580px}
.sess-card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:.7rem 1rem}
.sc-name{font-family:var(--display);font-size:15px;letter-spacing:.04em;margin-bottom:2px}
.sc-time{font-family:var(--mono);font-size:9px;color:var(--text2)}
.sc-st{font-family:var(--mono);font-size:9px;margin-top:4px}
.sc-open{color:var(--green)}.sc-cls{color:var(--text3)}
.override{font-family:var(--mono);font-size:10px;padding:5px 14px;border-radius:3px;border:1px solid var(--border2);background:transparent;color:var(--text2);cursor:pointer;transition:all .15s;margin-top:.25rem}
.override:hover{background:var(--bg2);color:var(--text)}
/* Spinner */
.spin{display:none;position:fixed;inset:0;z-index:200;background:rgba(5,7,9,.82);backdrop-filter:blur(8px);flex-direction:column;align-items:center;justify-content:center;gap:1rem}
.spin.show{display:flex}
.spinner{width:44px;height:44px;border:2px solid var(--border2);border-top-color:var(--green);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.spin-txt{font-family:var(--display);font-size:24px;letter-spacing:.1em;color:var(--green)}
.spin-sub{font-family:var(--mono);font-size:11px;color:var(--text2)}
/* Toast */
.toast{position:fixed;bottom:1.5rem;right:1.5rem;z-index:300;font-family:var(--mono);font-size:11px;padding:.7rem 1rem;border-radius:5px;background:var(--bg2);border:1px solid var(--border2);color:var(--text);max-width:300px;transform:translateY(80px);opacity:0;transition:all .3s;pointer-events:none}
.toast.show{transform:translateY(0);opacity:1}
.toast.tb{border-color:rgba(0,255,179,.3);background:var(--green-a)}
.toast.ts{border-color:rgba(255,61,90,.3);background:var(--red-a)}
.toast.tw{border-color:rgba(255,184,0,.3);background:var(--amber-a)}
</style>
</head>
<body>
<div class="shell">
<nav class="topbar">
  <div class="logo">SMC<em>FX</em> · LIVE SIGNALS</div>
  <div class="top-right">
    <div id="mktBadge" class="status-badge badge-closed"><div class="badge-dot"></div><span id="mktLabel">CHECKING...</span></div>
    <div class="clock" id="clockEl">--:--:-- UTC</div>
    <button class="scan-btn" id="scanBtn" onclick="runScan()">⟳ SCAN ALL</button>
  </div>
</nav>
<div style="background:var(--bg1);border-bottom:1px solid var(--border);overflow:hidden;height:28px;display:flex;align-items:center">
  <div style="display:flex;gap:2rem;padding:0 1.5rem;font-family:var(--mono);font-size:10px;white-space:nowrap;animation:none" id="priceStrip">
    <span id="tick_EURUSD" style="color:var(--text2)">EUR/USD <span style="color:var(--text3)">loading...</span></span>
    <span id="tick_GBPUSD" style="color:var(--text2)">GBP/USD</span>
    <span id="tick_USDJPY" style="color:var(--text2)">USD/JPY</span>
    <span id="tick_GBPJPY" style="color:var(--text2)">GBP/JPY</span>
    <span id="tick_AUDUSD" style="color:var(--text2)">AUD/USD</span>
    <span id="tick_USDCAD" style="color:var(--text2)">USD/CAD</span>
    <span id="tick_XAUUSD" style="color:var(--text2)">XAU/USD</span>
  </div>
</div>

<!-- Market Closed -->
<div class="closed" id="closedScreen">
  <div class="closed-icon">🔒</div>
  <div class="closed-title">MARKET CLOSED</div>
  <div class="closed-reason" id="closedReason">Forex market is currently closed.</div>
  <div class="closed-next" id="closedNext"></div>
  <div class="closed-time" id="closedTime"></div>
  <div class="sess-grid">
    <div class="sess-card"><div class="sc-name">SYDNEY</div><div class="sc-time">21:00–06:00 UTC</div><div class="sc-st" id="ss0">—</div></div>
    <div class="sess-card"><div class="sc-name">TOKYO</div><div class="sc-time">00:00–09:00 UTC</div><div class="sc-st" id="ss1">—</div></div>
    <div class="sess-card"><div class="sc-name">LONDON</div><div class="sc-time">07:00–16:00 UTC</div><div class="sc-st" id="ss2">—</div></div>
    <div class="sess-card"><div class="sc-name">NEW YORK</div><div class="sc-time">12:00–21:00 UTC</div><div class="sc-st" id="ss3">—</div></div>
  </div>
  <button class="override" onclick="demoMode()">View demo signals anyway</button>
</div>

<!-- Main -->
<div class="main" id="mainDash" style="display:none">
  <div class="left">
    <div class="sect">
      <div class="sect-label">Select Pair</div>
      <div class="pair-grid">
        <button class="pair-btn active" onclick="selectPair(this,'EURUSD')">EUR/USD</button>
        <button class="pair-btn" onclick="selectPair(this,'GBPUSD')">GBP/USD</button>
        <button class="pair-btn" onclick="selectPair(this,'USDJPY')">USD/JPY</button>
        <button class="pair-btn" onclick="selectPair(this,'GBPJPY')">GBP/JPY</button>
        <button class="pair-btn" onclick="selectPair(this,'AUDUSD')">AUD/USD</button>
        <button class="pair-btn" onclick="selectPair(this,'XAUUSD')">XAU/USD</button>
      </div>
    </div>
    <div class="sect">
      <div class="sect-label">SMC Confluence Score</div>
      <div class="score-num wait" id="scoreNum">—</div>
      <div class="bar-bg"><div class="bar-fill" id="barFill" style="width:0%;background:var(--amber)"></div></div>
      <div class="bar-labels"><span>0</span><span>THRESHOLD 50%</span><span>100%</span></div>
    </div>
    <div class="sect">
      <div class="sect-label">Active Signal</div>
      <div class="sig-dir WAIT" id="sigDir">WAIT</div>
      <div class="lvl-grid">
        <div class="lvl"><div class="lvl-k">Entry</div><div class="lvl-v lv-e" id="lvE">—</div></div>
        <div class="lvl"><div class="lvl-k">Stop Loss</div><div class="lvl-v lv-s" id="lvS">—</div></div>
        <div class="lvl"><div class="lvl-k">TP1 (1.5R)</div><div class="lvl-v lv-t" id="lvT1">—</div></div>
        <div class="lvl"><div class="lvl-k">TP2 (3R)</div><div class="lvl-v lv-t" id="lvT2">—</div></div>
        <div class="lvl"><div class="lvl-k">R:R</div><div class="lvl-v" id="lvRR" style="color:var(--blue)">—</div></div>
        <div class="lvl"><div class="lvl-k">Trend</div><div class="lvl-v" id="lvTr" style="color:var(--text)">—</div></div>
      </div>
      <div class="ctags" id="ctags"><span class="ctag neutral">Click SCAN ALL</span></div>
    </div>
    <div class="sect">
      <div class="sect-label">Detected SMC Zones</div>
      <div class="zone-list" id="zoneList"><div class="zone-row"><span class="zn" style="color:var(--text3)">Scan to detect zones</span></div></div>
    </div>
    <div class="sect">
      <div class="sect-label">Session Metrics</div>
      <div class="met-grid">
        <div class="met"><div class="met-k">Signals</div><div class="met-v" id="mSig">0</div></div>
        <div class="met"><div class="met-k">Buy</div><div class="met-v mv-g" id="mBuy">0</div></div>
        <div class="met"><div class="met-k">Sell</div><div class="met-v mv-r" id="mSell">0</div></div>
        <div class="met"><div class="met-k">Avg Score</div><div class="met-v mv-a" id="mAvg">—</div></div>
        <div class="met"><div class="met-k">Data</div><div class="met-v" id="mData" style="font-size:10px;color:var(--blue)">yfinance</div></div>
        <div class="met"><div class="met-k">Mode</div><div class="met-v mv-g" id="mMode" style="font-size:11px">LIVE</div></div>
      </div>
    </div>
  </div>

  <div class="right">
    <div class="chart-area">
      <div class="chart-top">
        <div>
          <div class="chart-sym" id="chartSym">—/—</div>
          <div class="chart-meta">
            <span id="cBias">Bias: —</span>
            <span id="cStruct">Structure: —</span>
            <span id="cSess">Session: —</span>
          </div>
        </div>
        <div>
          <div class="chart-px px-up" id="chartPx">—</div>
          <div style="font-family:var(--mono);font-size:9px;color:var(--text3);text-align:right" id="pxSrc">—</div>
        </div>
      </div>
      <div class="chart-wrap"><canvas id="chart" role="img" aria-label="SMC price chart"></canvas></div>
    </div>
    <div class="tbl-wrap">
      <div class="tbl-hdr">All Pairs — Scan Results</div>
      <table><thead><tr>
        <th>Pair</th><th>Signal</th><th>Entry</th><th>SL</th><th>TP1</th><th>TP2</th><th>R:R</th><th>Score</th><th>Confluences</th>
      </tr></thead><tbody id="tblBody">
        <tr><td colspan="9" style="color:var(--text3);text-align:center;padding:1.2rem">Click SCAN ALL — fetches real market data</td></tr>
      </tbody></table>
    </div>
    <div class="log-area" id="logArea"></div>
  </div>
</div>
</div>

<div class="spin" id="spinner">
  <div class="spinner"></div>
  <div class="spin-txt">SCANNING</div>
  <div class="spin-sub" id="spinSub">Connecting to market data...</div>
</div>
<div class="toast" id="toast"></div>

<script>
const API = '';
const DIGITS = {EURUSD:5,GBPUSD:5,USDJPY:3,GBPJPY:3,AUDUSD:5,USDCAD:5,XAUUSD:2};
const PIPS   = {EURUSD:0.0001,GBPUSD:0.0001,USDJPY:0.01,GBPJPY:0.01,AUDUSD:0.0001,USDCAD:0.0001,XAUUSD:0.1};

// Real prices fetched from backend — updated every 60s
let LIVE_PRICES = {
  EURUSD:{price:0},GBPUSD:{price:0},USDJPY:{price:0},
  GBPJPY:{price:0},AUDUSD:{price:0},USDCAD:{price:0},XAUUSD:{price:0}
};

let activePair='EURUSD', allSignals=[], chart=null;
let isDemoMode=false, buys=0, sells=0, scores=[];

async function refreshLivePrices(){
  try{
    const r=await fetch('/api/live-prices',{signal:AbortSignal.timeout(10000)});
    const d=await r.json();
    if(d.prices) LIVE_PRICES=d.prices;
    // Update ticker strip
    Object.entries(d.prices||{}).forEach(([pair,info])=>{
      const el=document.getElementById('tick_'+pair);
      if(el){
        const dp=DIGITS[pair]||5;
        const chg=info.change_pct||0;
        el.innerHTML='<span style="color:var(--text);font-weight:500">'+pair.slice(0,3)+'/'+pair.slice(3)+'</span> '+
          '<span style="color:var(--text2)">'+info.price.toFixed(dp)+'</span> '+
          '<span style="color:'+(chg>=0?'var(--green)':'var(--red)')+'">'+
          (chg>=0?'+':'')+chg.toFixed(2)+'%</span>';
      }
    });
    log.info&&addLog('Prices updated from yfinance','SYS','info');
  } catch(e){ /* silent fail */ }
}

// Clock
function tick(){
  const n=new Date();
  const pad=v=>String(v).padStart(2,'0');
  document.getElementById('clockEl').textContent=pad(n.getUTCHours())+':'+pad(n.getUTCMinutes())+':'+pad(n.getUTCSeconds())+' UTC';
}
tick(); setInterval(tick,1000);

// Market status
async function checkStatus(){
  try{
    const r=await fetch('/api/market-status',{signal:AbortSignal.timeout(5000)});
    const d=await r.json();
    const badge=document.getElementById('mktBadge');
    const label=document.getElementById('mktLabel');
    if(d.is_open){
      badge.className='status-badge badge-open';
      label.textContent=(d.active_sessions[0]||'OPEN')+' SESSION';
    } else {
      badge.className='status-badge badge-closed';
      label.textContent='MARKET CLOSED';
    }
    // Session cards
    const sNames=['Sydney','Tokyo','London','New York'];
    sNames.forEach((n,i)=>{
      const el=document.getElementById('ss'+i);
      if(!el) return;
      const on=d.active_sessions.includes(n);
      el.textContent=on?'● ACTIVE':'○ CLOSED';
      el.className='sc-st '+(on?'sc-open':'sc-cls');
    });
    if(d.is_closed && !isDemoMode){
      document.getElementById('closedReason').textContent=d.reason;
      document.getElementById('closedNext').textContent=d.next_open||'';
      document.getElementById('closedTime').textContent=d.server_time_utc;
      document.getElementById('closedScreen').classList.add('show');
      document.getElementById('mainDash').style.display='none';
    } else {
      document.getElementById('closedScreen').classList.remove('show');
      document.getElementById('mainDash').style.display='grid';
      document.getElementById('cSess').textContent='Session: '+(d.active_sessions.join(' + ')||'—');
    }
    return d;
  } catch(e){
    // Server error — still show dashboard
    document.getElementById('closedScreen').classList.remove('show');
    document.getElementById('mainDash').style.display='grid';
    return null;
  }
}

function demoMode(){
  isDemoMode=true;
  document.getElementById('closedScreen').classList.remove('show');
  document.getElementById('mainDash').style.display='grid';
  document.getElementById('mMode').textContent='DEMO';
  document.getElementById('mMode').className='met-v mv-a';
  addLog('Demo mode — market closed, showing simulated signals','SYS','warn');
  buildDemoChart(activePair);
}

// Scan
async function runScan(){
  const btn=document.getElementById('scanBtn');
  btn.disabled=true;
  document.getElementById('spinner').classList.add('show');
  const steps=['Checking market hours...','Fetching OHLCV from yfinance...','Detecting Order Blocks...','Scanning FVG zones...','Analysing CHoCH & BOS...','Scoring confluences...','Building signals...'];
  let si=0;
  const iv=setInterval(()=>{ document.getElementById('spinSub').textContent=steps[Math.min(si++,steps.length-1)]; },700);
  try{
    const r=await fetch('/api/scan',{signal:AbortSignal.timeout(90000)});
    const d=await r.json();
    clearInterval(iv);
    // Backend returns signals even when closed (using real latest prices)
    if(d.market_closed && !d.signals.length){
      showToast('⚠ '+d.reason,'tw');
      document.getElementById('closedReason').textContent=d.reason;
      document.getElementById('closedNext').textContent=d.next_open||'';
      document.getElementById('closedTime').textContent=d.server_time_utc;
      document.getElementById('closedScreen').classList.add('show');
      document.getElementById('mainDash').style.display='none';
    } else {
      if(d.market_closed){
        document.getElementById('mMode').textContent='DEMO';
        document.getElementById('mMode').className='met-v mv-a';
        document.getElementById('mData').textContent='real prices';
        showToast('Market closed — showing signals on real latest prices','tw');
        addLog(d.reason+' — using real latest prices','MKT','warn');
      }
      handleResult(d);
    }
  } catch(e){
    clearInterval(iv);
    addLog('Backend error: '+e.message,'ERR','warn');
    showToast('⚠ Cannot reach backend — is server.py running?','tw');
  } finally {
    document.getElementById('spinner').classList.remove('show');
    btn.disabled=false;
  }
}

function handleResult(d){
  allSignals=d.signals||[];
  updateTable(allSignals);
  allSignals.forEach(s=>{
    const t=s.direction==='BUY'?'buy':s.direction==='SELL'?'sell':'wait';
    addLog(s.direction+' | '+s.score_pct+'% | '+(s.conf||[]).join(',')||'—', s.pair, t);
    if(s.direction==='BUY'){buys++;scores.push(s.score);showToast('📈 '+s.pair+' BUY — '+s.score_pct+'%','tb');}
    if(s.direction==='SELL'){sells++;scores.push(s.score);showToast('📉 '+s.pair+' SELL — '+s.score_pct+'%','ts');}
  });
  document.getElementById('mSig').textContent=allSignals.length;
  document.getElementById('mBuy').textContent=buys;
  document.getElementById('mSell').textContent=sells;
  document.getElementById('mAvg').textContent=scores.length?Math.round(scores.reduce((a,b)=>a+b,0)/scores.length*100)+'%':'—';
  const active=allSignals.find(s=>s.pair===activePair);
  if(active) updateSignalUI(active);
  fetchChart(activePair);
  addLog('Scan done — '+allSignals.length+' pairs | '+(d.server_time_utc||''),'SYS','info');
}

// Pair select
function selectPair(btn,pair){
  document.querySelectorAll('.pair-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  activePair=pair;
  const s=allSignals.find(x=>x.pair===pair);
  if(s) updateSignalUI(s);
  fetchChart(pair);
}

// Chart
async function fetchChart(pair){
  try{
    if(isDemoMode){ buildDemoChart(pair); return; }
    const r=await fetch('/api/bars/'+pair+'?tf=H1&limit=80',{signal:AbortSignal.timeout(15000)});
    const d=await r.json();
    if(d.bars && d.bars.length) buildChart(pair,d.bars);
    else buildDemoChart(pair);
  } catch(e){ buildDemoChart(pair); }
}

function buildChart(pair,bars){
  const dp=DIGITS[pair]||5;
  const labels=bars.map((_,i)=>i);
  const colors=bars.map(b=>b.close>=b.open?'rgba(0,255,179,0.75)':'rgba(255,61,90,0.75)');
  const sig=allSignals.find(s=>s.pair===pair);
  const ds=[{type:'bar',label:'P',data:bars.map((b,i)=>({x:i,y:[b.open,b.close]})),backgroundColor:colors,borderWidth:0,barPercentage:0.55}];
  if(sig&&sig.zones){
    sig.zones.filter(z=>z.color==='green'&&z.lo).forEach(z=>{
      ds.push({type:'line',data:labels.map(()=>z.lo),borderColor:'rgba(0,255,179,.4)',borderWidth:1,borderDash:[4,3],pointRadius:0,tension:0});
      ds.push({type:'line',data:labels.map(()=>z.hi),borderColor:'rgba(0,255,179,.15)',borderWidth:1,borderDash:[2,5],pointRadius:0,tension:0});
    });
    sig.zones.filter(z=>z.color==='red'&&z.hi).forEach(z=>{
      ds.push({type:'line',data:labels.map(()=>z.hi),borderColor:'rgba(255,61,90,.4)',borderWidth:1,borderDash:[4,3],pointRadius:0,tension:0});
    });
    sig.zones.filter(z=>z.color==='amber'&&z.mid).forEach(z=>{
      ds.push({type:'line',data:labels.map(()=>z.mid),borderColor:'rgba(255,184,0,.4)',borderWidth:1,borderDash:[2,4],pointRadius:0,tension:0});
    });
  }
  const last=bars[bars.length-1]?.close||0;
  document.getElementById('chartSym').textContent=pair.slice(0,3)+'/'+pair.slice(3);
  document.getElementById('chartPx').textContent=last.toFixed(dp);
  document.getElementById('chartPx').className='chart-px px-up';
  document.getElementById('pxSrc').textContent=isDemoMode?'SIMULATED':'LIVE · yfinance';
  if(sig){
    document.getElementById('cBias').textContent='Bias: '+(sig.trend||'—').toUpperCase();
    document.getElementById('cStruct').textContent='Structure: '+(sig.choch?'CHoCH('+sig.choch_dir+')':sig.bos?'BOS('+sig.bos_dir+')':'None');
  }
  if(chart) chart.destroy();
  chart=new Chart(document.getElementById('chart'),{
    type:'bar',data:{labels,datasets:ds},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:350},
      plugins:{legend:{display:false},tooltip:{enabled:false}},
      scales:{x:{display:false},y:{grid:{color:'rgba(255,255,255,0.04)'},
        ticks:{color:'#2d404f',font:{size:9,family:'JetBrains Mono'},maxTicksLimit:6,
          callback:v=>typeof v==='number'?v.toFixed(dp>3?4:dp<3?1:2):v}}}}
  });
}

async function buildDemoChart(pair){
  // Try to get real bars from backend first
  try{
    const r=await fetch('/api/bars/'+pair+'?tf=H1&limit=80',{signal:AbortSignal.timeout(10000)});
    const d=await r.json();
    if(d.bars&&d.bars.length){ buildChart(pair,d.bars); return; }
  } catch(e){ /* fallback to generated */ }
  // Generate demo bars anchored to real price
  const lp=LIVE_PRICES[pair]||{};
  const base=lp.price||0;
  if(base>0){
    const bars=genDemoBarsFromPrice(pair,base,80);
    buildChart(pair,bars);
  } else {
    const bars=genDemoBars(pair,80);
    buildChart(pair,bars);
  }
}

// Signal UI
function updateSignalUI(sig){
  const dp=DIGITS[sig.pair]||5;
  const dir=sig.direction||'WAIT';
  document.getElementById('sigDir').textContent=dir;
  document.getElementById('sigDir').className='sig-dir '+dir;
  document.getElementById('scoreNum').textContent=sig.score_pct+'%';
  document.getElementById('scoreNum').className='score-num '+(dir==='BUY'?'bull':dir==='SELL'?'bear':'wait');
  const fill=document.getElementById('barFill');
  fill.style.width=sig.score_pct+'%';
  fill.style.background=dir==='BUY'?'var(--green)':dir==='SELL'?'var(--red)':'var(--amber)';
  const f=v=>v&&v!==0?v.toFixed(dp):'—';
  document.getElementById('lvE').textContent=dir!=='WAIT'?f(sig.entry):'—';
  document.getElementById('lvS').textContent=dir!=='WAIT'?f(sig.sl):'—';
  document.getElementById('lvT1').textContent=dir!=='WAIT'?f(sig.tp1):'—';
  document.getElementById('lvT2').textContent=dir!=='WAIT'?f(sig.tp2):'—';
  document.getElementById('lvRR').textContent=dir!=='WAIT'?'1:'+sig.rr:'—';
  document.getElementById('lvTr').textContent=(sig.trend||'—').toUpperCase();
  const tags=document.getElementById('ctags');
  tags.innerHTML=sig.conf&&sig.conf.length?sig.conf.map(c=>'<span class="ctag'+(dir==='SELL'?' bear':'')+'">'+c+'</span>').join(''):'<span class="ctag neutral">No confluence</span>';
  updateZones(sig.zones||[],dp);
}

function updateZones(zones,dp){
  const zl=document.getElementById('zoneList');
  if(!zones.length){zl.innerHTML='<div class="zone-row"><span class="zn" style="color:var(--text3)">No zones detected</span></div>';return;}
  const cm={green:'zd-g',red:'zd-r',amber:'zd-a',blue:'zd-b'};
  const sm={Tapped:'zs-g',Target:'zs-x',Untouched:'zs-x',Unfilled:'zs-a'};
  zl.innerHTML=zones.slice(0,6).map(z=>{
    const px=z.price?z.price.toFixed(dp):(z.lo&&z.hi?z.lo.toFixed(dp)+'–'+z.hi.toFixed(dp):'—');
    return '<div class="zone-row"><div class="zd '+(cm[z.color]||'zd-b')+'"></div><span class="zn">'+z.type+'</span><span class="zp">'+px+'</span><span class="zs '+(sm[z.status]||'zs-x')+'">'+z.status+'</span></div>';
  }).join('');
}

function updateTable(sigs){
  const tb=document.getElementById('tblBody');
  if(!sigs.length){tb.innerHTML='<tr><td colspan="9" style="color:var(--text3);text-align:center;padding:1.2rem">No signals</td></tr>';return;}
  tb.innerHTML=sigs.map(sig=>{
    const dp=DIGITS[sig.pair]||5;
    const dc=sig.direction==='BUY'?'d-buy':sig.direction==='SELL'?'d-sell':'d-wait';
    const pc=sig.score_pct>=70?'sp-h':sig.score_pct>=50?'sp-m':'sp-l';
    const f=v=>v&&v!==0?v.toFixed(dp):'—';
    return '<tr><td style="color:var(--text);font-weight:500">'+sig.pair.slice(0,3)+'/'+sig.pair.slice(3)+'</td><td class="'+dc+'">'+sig.direction+'</td><td>'+f(sig.entry)+'</td><td>'+f(sig.sl)+'</td><td>'+f(sig.tp1)+'</td><td>'+f(sig.tp2)+'</td><td>'+(sig.direction!=='WAIT'?'1:'+sig.rr:'—')+'</td><td><span class="sp '+pc+'">'+sig.score_pct+'%</span></td><td style="color:var(--text2);font-size:10px">'+((sig.conf||[]).join(' · ')||'—')+'</td></tr>';
  }).join('');
}

// Log & Toast
function addLog(msg,pair,type){
  const la=document.getElementById('logArea');
  const now=new Date();
  const t=String(now.getUTCHours()).padStart(2,'0')+':'+String(now.getUTCMinutes()).padStart(2,'0');
  const pc=type==='buy'?'lp-g':type==='sell'?'lp-r':type==='warn'?'lp-a':'lp-b';
  const tm={buy:'ltag-b',sell:'ltag-s',info:'ltag-i',warn:'ltag-w'};
  const row=document.createElement('div');
  row.className='lr';
  row.innerHTML='<span class="lt">'+t+'</span><span class="lp '+pc+'">'+(pair||'SYS')+'</span><span class="lm">'+msg+'</span>'+(type!=='info'?'<span class="ltag '+tm[type]+'">'+type.toUpperCase()+'</span>':'');
  la.insertBefore(row,la.firstChild);
  if(la.children.length>80) la.removeChild(la.lastChild);
}
function showToast(msg,cls){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='toast show '+(cls||'');
  setTimeout(()=>t.classList.remove('show'),4000);
}

// Demo data
function seededR(seed){let s=seed;return()=>{s^=s<<13;s^=s>>17;s^=s<<5;return(s>>>0)/4294967296;};}
const VOL_MAP={EURUSD:0.0008,GBPUSD:0.0012,USDJPY:0.12,GBPJPY:0.18,AUDUSD:0.0007,USDCAD:0.0009,XAUUSD:2.8};

function genDemoBarsFromPrice(pair,basePrice,n=80){
  const vol=VOL_MAP[pair]||basePrice*0.001;
  // Walk back from base price so last bar ends at real price
  const bars=[];
  let p=basePrice*(1+((Math.random()-0.5)*0.005));
  for(let i=0;i<n;i++){
    const drift=(basePrice-p)*0.04;
    const o=p,c=o+drift+(Math.random()-0.5)*vol*2;
    const h=Math.max(o,c)+Math.abs((Math.random()-0.5)*vol*0.6);
    const l=Math.min(o,c)-Math.abs((Math.random()-0.5)*vol*0.6);
    bars.push({time:Math.floor(Date.now()/1000)-(n-i)*3600,
              open:parseFloat(o.toFixed(5)),high:parseFloat(h.toFixed(5)),
              low:parseFloat(l.toFixed(5)),close:parseFloat(c.toFixed(5)),
              volume:Math.floor(Math.random()*4500+500)});
    p=c;
  }
  // Anchor last bar to real price
  if(bars.length){ bars[bars.length-1].close=basePrice; bars[bars.length-1].high=Math.max(bars[bars.length-1].high,basePrice); bars[bars.length-1].low=Math.min(bars[bars.length-1].low,basePrice); }
  return bars;
}

function genDemoBars(pair,n=80){
  const lp=LIVE_PRICES[pair]||{};
  const base=lp.price||0;
  if(base>0) return genDemoBarsFromPrice(pair,base,n);
  // Pure fallback (should rarely happen)
  const vol=VOL_MAP[pair]||0.001;
  const fallback={EURUSD:1.0855,GBPUSD:1.2685,USDJPY:149.82,GBPJPY:192.30,AUDUSD:0.6512,USDCAD:1.3810,XAUUSD:2321.0};
  let p=fallback[pair]||1;
  const bars=[];
  for(let i=0;i<n;i++){const o=p,c=o+(Math.random()-0.48)*vol*3;bars.push({time:Math.floor(Date.now()/1000)-n*3600+i*3600,open:o,high:Math.max(o,c)+Math.random()*vol*.5,low:Math.min(o,c)-Math.random()*vol*.5,close:c,volume:Math.floor(Math.random()*5000+500)});p=c;}
  return bars;
}
function demoSignals(){
  const dirs=['BUY','SELL','WAIT','BUY','SELL','WAIT'];
  const cfs=[['HTF↑','OB_tap','FVG','CHoCH','Discount'],['HTF↓','OB_tap','BOS','Premium'],['HTF↑','FVG'],['CHoCH','OB_tap','Liq_pool','Discount'],['HTF↓','CHoCH','FVG','Premium','BOS'],['OB_tap']];
  return Object.keys(BASES).map((pair,i)=>{
    const dir=dirs[i];const score=dir==='WAIT'?0.35+Math.random()*.15:0.52+Math.random()*.38;
    const p=BASES[pair];const pip=PIPS[pair];const slD=pip*20;const dp=DIGITS[pair]||5;
    return{pair,direction:dir,score:parseFloat(score.toFixed(3)),score_pct:Math.round(score*100),conf:dir==='WAIT'?[]:cfs[i],
      entry:parseFloat(p.toFixed(dp)),sl:parseFloat((dir==='BUY'?p-slD:p+slD).toFixed(dp)),
      tp1:parseFloat((dir==='BUY'?p+slD*1.5:p-slD*1.5).toFixed(dp)),tp2:parseFloat((dir==='BUY'?p+slD*3:p-slD*3).toFixed(dp)),rr:2.5,
      trend:i%3===0?'bullish':i%3===1?'bearish':'ranging',choch:i%2===0,choch_dir:i%2===0?'bullish':'bearish',
      bos:i%3===1,bos_dir:'bullish',in_discount:i%2===0,in_premium:i%2===1,
      zones:[{type:'Bull OB',hi:parseFloat((p+pip*15).toFixed(dp)),lo:parseFloat((p-pip*5).toFixed(dp)),mid:p,status:'Tapped',color:'green'},
             {type:'Bear FVG',hi:parseFloat((p+pip*40).toFixed(dp)),lo:parseFloat((p+pip*25).toFixed(dp)),mid:parseFloat((p+pip*32).toFixed(dp)),status:'Unfilled',color:'amber'},
             {type:'BSL',price:parseFloat((p+pip*60).toFixed(dp)),status:'Target',color:'blue'}],
      timestamp:new Date().toISOString()};
  });
}

// Init
(async function(){
  addLog('Server started — fetching live prices...','SYS','info');
  // Fetch real prices first (works even on weekends)
  await refreshLivePrices();
  addLog('Live prices loaded from yfinance','SYS','info');
  const st=await checkStatus();
  buildDemoChart(activePair);
  if(st&&st.is_open){
    addLog('Market OPEN — click SCAN ALL for real signals','SYS','info');
  } else {
    addLog('Market closed — real prices loaded, click SCAN ALL for SMC signals','SYS','warn');
  }
})();
setInterval(checkStatus,60000);
setInterval(refreshLivePrices,60000);  // refresh prices every minute
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def open_browser():
    """Open browser after a short delay so Flask is ready."""
    time.sleep(1.2)
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    ms = market_status()

    print("""
╔════════════════════════════════════════════════╗
║        SMC Signal Dashboard — v3.0            ║
║   Smart Money Concepts · Real Market Data      ║
╚════════════════════════════════════════════════╝""")
    print(f"\n  Market Status : {'✓ OPEN' if ms['is_open'] else '✗ CLOSED'}")
    if ms["is_closed"]:
        print(f"  Reason        : {ms['reason']}")
        print(f"  Next Open     : {ms['next_open']}")
    else:
        print(f"  Session       : {', '.join(ms['active_sessions']) or 'None'}")
    print(f"  Server Time   : {ms['server_time_utc']}")
    print(f"\n  Opening       : http://localhost:5000\n")

    # Auto-open browser in background thread
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
