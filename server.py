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
            out[pair] = {
                "price": round(float(last["Close"]),5),
                "change_pct": round((float(last["Close"])-float(prev["Close"]))/float(prev["Close"])*100,3)
                              if float(prev["Close"])!=0 else 0
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

def analyse(bars, pair):
    if not bars or len(bars)<50: return {}
    p=PIP.get(pair,0.0001); rev=list(reversed(bars)); price=rev[0]["close"]
    lb=min(100,len(rev)); rhi=max(b["high"] for b in rev[:lb]); rlo=min(b["low"] for b in rev[:lb]); mid=(rhi+rlo)/2
    sw=detect_swings(rev); trend=get_trend(sw); choch,cd,bos,bd=get_structure(sw,trend)
    obs=detect_obs(rev)
    for ob in obs:
        if ob["lo"]<=price<=ob["hi"]: ob["tapped"]=True
    fvgs=detect_fvgs(rev,p)
    near_fvgs=[f for f in fvgs if abs(f["mid"]-price)/p<100]
    liq=[{"price":s["price"],"type":"BSL" if s["hi"] else "SSL"} for s in sw[-12:]]
    return {"price":price,"trend":trend,"mid":mid,
            "disc":price<mid,"prem":price>mid,
            "choch":choch,"cd":cd,"bos":bos,"bd":bd,
            "obs":obs,"tapped":[o for o in obs if o["tapped"]],
            "fvgs":fvgs,"near":near_fvgs,"liq":liq}

def score(buy, htf, mtf, ltf):
    s=0.0; c=[]
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
    return round(min(s,1.0),3),c

def make_signal(pair, htf, mtf, ltf):
    p=PIP.get(pair,0.0001); dp=DIGITS.get(pair,5); price=mtf.get("price",0)
    bs,bc=score(True, htf,mtf,ltf); ss,sc=score(False,htf,mtf,ltf)
    if bs>=ss and bs>0.70: direction,sc_val,conf="BUY", bs,bc
    elif ss>bs and ss>0.70: direction,sc_val,conf="SELL",ss,sc
    else: direction,sc_val,conf="WAIT",max(bs,ss),[]
    sl_d=p*30; entry=round(price,dp)
    sl  =round(price-sl_d if direction=="BUY" else price+sl_d,dp)
    tp1 =round(price+sl_d*1.5 if direction=="BUY" else price-sl_d*1.5,dp)
    tp2 =round(price+sl_d*3.0 if direction=="BUY" else price-sl_d*3.0,dp)
    rr  =round(abs(tp1-entry)/abs(entry-sl),2) if abs(entry-sl)>0 else 0
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
    return {"pair":pair,"direction":direction,"score":sc_val,
            "score_pct":round(sc_val*100),"conf":conf,
            "entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":rr,
            "trend":mtf.get("trend","ranging"),
            "choch":mtf.get("choch",False),"choch_dir":mtf.get("cd",""),
            "bos":mtf.get("bos",False),"bos_dir":mtf.get("bd",""),
            "disc":mtf.get("disc",False),"prem":mtf.get("prem",False),
            "zones":zones,"ts":datetime.now(timezone.utc).isoformat()}

def run_scan(force=False):
    global _signal_cache
    ms=market_status(); prices=fetch_prices(); out=[]
    for pair in PAIRS:
        cur_price=prices.get(pair,{}).get("price",0)
        # Use cache if price hasn't moved more than 0.03%
        if not force and pair in _signal_cache:
            cached=_signal_cache[pair]; lp=cached.get("entry",0)
            if lp>0 and abs(cur_price-lp)/lp<0.0003:
                cached["cached"]=True; out.append(cached); continue
        try:
            if ms["is_open"]:
                bh=fetch_bars(pair,"H4"); bm=fetch_bars(pair,"H1"); bl=fetch_bars(pair,"M15")
                if not bm: raise Exception("no data")
            else:
                bh=bm=bl=demo_bars(pair)
            htf=analyse(bh or bm,pair); mtf=analyse(bm,pair); ltf=analyse(bl or bm,pair)
            sig=make_signal(pair,htf,mtf,ltf); sig["cached"]=False; sig["demo"]=not ms["is_open"]
            _signal_cache[pair]=sig; out.append(sig)
            log.info(f"  {pair}: {sig['direction']} {sig['score_pct']}%")
        except Exception as e:
            log.warning(f"  {pair}: {e}")
            if pair in _signal_cache: out.append(_signal_cache[pair])
    return out

# ── API routes ────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    return jsonify(market_status())

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
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500&display=swap" rel="stylesheet">
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

/* Layout */
.app{position:relative;z-index:1;display:grid;grid-template-rows:52px 26px 1fr;height:100vh;overflow:hidden}

/* Nav */
nav{display:flex;align-items:center;justify-content:space-between;padding:0 1.25rem;
    background:rgba(5,8,11,.95);border-bottom:1px solid var(--b);backdrop-filter:blur(12px)}
.logo{font-family:var(--disp);font-size:20px;letter-spacing:.05em}
.logo em{color:var(--g);font-style:normal}
.nav-r{display:flex;align-items:center;gap:.75rem}
.mkt-badge{display:flex;align-items:center;gap:5px;font-family:var(--mono);font-size:9px;
           padding:3px 10px;border-radius:3px;letter-spacing:.1em;border:1px solid}
.open-b{color:var(--g);border-color:rgba(0,255,179,.25);background:var(--ga)}
.closed-b{color:var(--a);border-color:rgba(255,184,0,.25);background:var(--aa)}
.bd{width:5px;height:5px;border-radius:50%}
.open-b .bd{background:var(--g);animation:pulse 1.2s infinite}
.closed-b .bd{background:var(--a)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.15}}
.clk{font-family:var(--mono);font-size:10px;color:var(--t3)}
.scan-btn{font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.06em;
          padding:6px 16px;border-radius:3px;background:var(--g);color:#000;
          border:none;cursor:pointer;transition:opacity .15s;text-transform:uppercase}
.scan-btn:hover{opacity:.8}.scan-btn:disabled{opacity:.35;cursor:default}
.force-btn{font-family:var(--mono);font-size:10px;padding:6px 12px;border-radius:3px;
           background:transparent;color:var(--a);border:1px solid rgba(255,184,0,.3);cursor:pointer}
.force-btn:hover{background:var(--aa)}

/* Ticker */
.ticker{background:var(--bg1);border-bottom:1px solid var(--b);overflow:hidden;
        display:flex;align-items:center;padding:0 1rem;gap:2rem;font-family:var(--mono);font-size:9px}
.tick{display:flex;gap:6px;align-items:center;white-space:nowrap;cursor:pointer;padding:4px 0}
.tick:hover .tsym{color:var(--g)}
.tsym{color:var(--t);font-weight:500}.tpx{color:var(--t2)}
.tup{color:var(--g)}.tdn{color:var(--r)}

/* Main grid */
.main{display:grid;grid-template-columns:300px 1fr;overflow:hidden}

/* Left panel */
.left{border-right:1px solid var(--b);background:var(--bg1);overflow-y:auto;display:flex;flex-direction:column}
.sect{padding:.85rem 1rem;border-bottom:1px solid var(--b)}
.slbl{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--t3);margin-bottom:.55rem}

/* Pair buttons */
.pgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.pb{font-family:var(--mono);font-size:11px;font-weight:500;padding:7px 3px;border-radius:4px;
    border:1px solid var(--b);background:var(--bg2);color:var(--t2);cursor:pointer;text-align:center;transition:all .15s}
.pb:hover{color:var(--t);border-color:var(--b2)}
.pb.on{background:var(--ga);border-color:rgba(0,255,179,.3);color:var(--g)}

/* Score */
.big-score{font-family:var(--disp);font-size:56px;line-height:1;letter-spacing:.02em;margin-bottom:.3rem}
.bs-bull{color:var(--g)}.bs-bear{color:var(--r)}.bs-wait{color:var(--a)}
.bar-bg{height:4px;background:var(--bg3);border-radius:2px;overflow:hidden;margin-bottom:.35rem}
.bar-fill{height:100%;border-radius:2px;transition:width .5s,background .3s}
.blbl{display:flex;justify-content:space-between;font-family:var(--mono);font-size:8px;color:var(--t3)}

/* Signal */
.sig-dir{font-family:var(--disp);font-size:38px;letter-spacing:.05em}
.sd-buy{color:var(--g)}.sd-sell{color:var(--r)}.sd-wait{color:var(--a)}
.cache-tag{font-family:var(--mono);font-size:9px;padding:2px 7px;border-radius:3px}
.ct-live{background:var(--ga);color:var(--g);border:1px solid rgba(0,255,179,.2)}
.ct-cached{background:var(--bla);color:var(--bl);border:1px solid rgba(61,159,255,.2)}
.ct-demo{background:var(--aa);color:var(--a);border:1px solid rgba(255,184,0,.2)}
.lvgrid{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin:.6rem 0}
.lv{background:var(--bg3);border-radius:4px;padding:6px 8px}
.lvk{font-family:var(--mono);font-size:8px;color:var(--t3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px}
.lvv{font-family:var(--mono);font-size:12px;font-weight:500}
.lv-e{color:var(--bl)}.lv-s{color:var(--r)}.lv-t{color:var(--g)}.lv-rr{color:var(--t)}
.ctags{display:flex;flex-wrap:wrap;gap:3px;margin-top:.4rem}
.ctag{font-family:var(--mono);font-size:9px;padding:2px 6px;border-radius:3px;
      background:var(--ga);color:var(--g);border:1px solid rgba(0,255,179,.15)}
.ctag.bear{background:var(--ra);color:var(--r);border-color:rgba(255,61,90,.15)}
.ctag.neutral{background:var(--bg3);color:var(--t2);border-color:var(--b)}

/* Zones */
.zlist{display:flex;flex-direction:column;gap:3px}
.zrow{display:flex;align-items:center;gap:6px;padding:5px 8px;border-radius:3px;border:1px solid var(--b);background:var(--bg2)}
.zdot{width:6px;height:6px;border-radius:1px;flex-shrink:0}
.zg{background:var(--g)}.zr{background:var(--r)}.za{background:var(--a)}.zb{background:var(--bl)}
.zname{font-family:var(--mono);font-size:9px;font-weight:500;min-width:68px;color:var(--t)}
.zprice{font-family:var(--mono);font-size:9px;color:var(--t2);flex:1}
.zst{font-family:var(--mono);font-size:8px}
.zst-g{color:var(--g)}.zst-a{color:var(--a)}.zst-x{color:var(--t3)}

/* Metrics */
.mgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.mc{background:var(--bg2);border:1px solid var(--b);border-radius:3px;padding:.5rem .65rem}
.mck{font-family:var(--mono);font-size:8px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);margin-bottom:2px}
.mcv{font-family:var(--mono);font-size:14px;font-weight:500}
.mcg{color:var(--g)}.mcr{color:var(--r)}.mca{color:var(--a)}

/* Right panel */
.right{display:flex;flex-direction:column;overflow:hidden;background:var(--bg)}

/* Chart */
.chart-area{flex:1;padding:1rem 1.25rem .5rem;display:flex;flex-direction:column;min-height:0}
.chart-hdr{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.75rem}
.csym{font-family:var(--disp);font-size:28px;letter-spacing:.04em;color:var(--t)}
.cmeta{font-family:var(--mono);font-size:9px;color:var(--t2);display:flex;gap:.75rem;margin-top:2px;flex-wrap:wrap}
.cpx{font-family:var(--mono);font-size:13px;font-weight:500;text-align:right}
.cup{color:var(--g)}.cdn{color:var(--r)}
.chart-wrap{flex:1;position:relative;min-height:0}

/* Table */
.tbl-wrap{padding:.5rem 1.25rem .75rem;border-top:1px solid var(--b);overflow-x:auto}
.tlbl{font-family:var(--mono);font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);padding:.4rem 0}
table{width:100%;border-collapse:collapse;min-width:560px}
th{font-family:var(--mono);font-size:8px;letter-spacing:.1em;text-transform:uppercase;
   color:var(--t3);padding:4px 7px;text-align:left;border-bottom:1px solid var(--b)}
td{font-family:var(--mono);font-size:10px;padding:6px 7px;border-bottom:1px solid var(--b);color:var(--t2)}
tr:last-child td{border-bottom:none}tr:hover td{background:var(--bg2)}
.db{color:var(--g);font-weight:700}.ds{color:var(--r);font-weight:700}.dw{color:var(--a)}
.sp{font-size:8px;padding:1px 5px;border-radius:2px}
.sph{background:var(--ga);color:var(--g)}.spm{background:var(--aa);color:var(--a)}.spl{background:var(--ra);color:var(--r)}

/* Log */
.log{height:130px;border-top:1px solid var(--b);overflow-y:auto;padding:.6rem 1.25rem;
     background:var(--bg1);display:flex;flex-direction:column;gap:2px}
.lr{display:flex;gap:8px;font-family:var(--mono);font-size:9px;line-height:1.8;align-items:baseline}
.lt{color:var(--t3);min-width:38px}.lp{min-width:46px;font-weight:500}
.lg{color:var(--g)}.lr2{color:var(--r)}.lb{color:var(--bl)}.la{color:var(--a)}
.lmsg{color:var(--t2);flex:1}
.ltag{font-size:8px;padding:1px 4px;border-radius:2px;margin-left:2px}
.ltb{background:var(--ga);color:var(--g)}.lts{background:var(--ra);color:var(--r)}
.lti{background:var(--bla);color:var(--bl)}.ltw{background:var(--aa);color:var(--a)}

/* Closed screen */
.closed-screen{display:none;position:fixed;inset:0;z-index:50;background:var(--bg);
               flex-direction:column;align-items:center;justify-content:center;gap:1rem;text-align:center}
.closed-screen.show{display:flex}
.cs-icon{font-size:48px}
.cs-title{font-family:var(--disp);font-size:48px;letter-spacing:.05em;color:var(--a)}
.cs-reason{font-family:var(--mono);font-size:11px;color:var(--t2);max-width:400px;line-height:1.8}
.cs-next{font-family:var(--mono);font-size:11px;color:var(--g);padding:7px 18px;
         border:1px solid rgba(0,255,179,.3);border-radius:3px;background:var(--ga)}
.cs-time{font-family:var(--mono);font-size:9px;color:var(--t3)}
.sess-row{display:flex;gap:8px}
.sess-card{background:var(--bg2);border:1px solid var(--b);border-radius:5px;padding:.6rem .9rem;text-align:left}
.sc-name{font-family:var(--disp);font-size:14px;letter-spacing:.04em;margin-bottom:2px}
.sc-time{font-family:var(--mono);font-size:8px;color:var(--t2)}
.sc-st{font-family:var(--mono);font-size:8px;margin-top:3px}
.sc-open{color:var(--g)}.sc-cls{color:var(--t3)}
.demo-btn{font-family:var(--mono);font-size:10px;padding:6px 14px;border-radius:3px;
          border:1px solid var(--b2);background:transparent;color:var(--t2);cursor:pointer;margin-top:.5rem}
.demo-btn:hover{background:var(--bg2);color:var(--t)}

/* Spinner */
.spin-overlay{display:none;position:fixed;inset:0;z-index:100;background:rgba(5,8,11,.8);
              backdrop-filter:blur(8px);flex-direction:column;align-items:center;justify-content:center;gap:1rem}
.spin-overlay.show{display:flex}
.spinner{width:40px;height:40px;border:2px solid var(--b2);border-top-color:var(--g);
         border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.spin-txt{font-family:var(--disp);font-size:22px;letter-spacing:.1em;color:var(--g)}
.spin-sub{font-family:var(--mono);font-size:10px;color:var(--t2)}

/* Toast */
.toast{position:fixed;bottom:1.25rem;right:1.25rem;z-index:200;font-family:var(--mono);font-size:10px;
       padding:.6rem .9rem;border-radius:4px;background:var(--bg2);border:1px solid var(--b2);
       color:var(--t);max-width:280px;transform:translateY(60px);opacity:0;transition:all .25s;pointer-events:none}
.toast.show{transform:translateY(0);opacity:1}
.tb{border-color:rgba(0,255,179,.3);background:var(--ga)}
.ts{border-color:rgba(255,61,90,.3);background:var(--ra)}
.tw{border-color:rgba(255,184,0,.3);background:var(--aa)}

@media(max-width:800px){
  .main{grid-template-columns:1fr}
  .right{display:none}
  body{overflow:auto}
  .app{height:auto;overflow:visible}
}
</style>
</head>
<body>
<div class="app">

<nav>
  <div class="logo">SMC<em>FX</em> · SIGNALS</div>
  <div class="nav-r">
    <div id="mktBadge" class="mkt-badge open-b"><div class="bd"></div><span id="mktLbl">CONNECTING</span></div>
    <div class="clk" id="clk">--:--:-- UTC</div>
    <button class="force-btn" onclick="doScan(true)">↺ FORCE</button>
    <button class="scan-btn" id="scanBtn" onclick="doScan(false)">⟳ SCAN ALL</button>
  </div>
</nav>

<div class="ticker" id="tickerStrip">
  <span class="tick" id="tick_EURUSD" onclick="selPair(null,'EURUSD')"><span class="tsym">EUR/USD</span><span class="tpx">—</span></span>
  <span class="tick" id="tick_GBPUSD" onclick="selPair(null,'GBPUSD')"><span class="tsym">GBP/USD</span><span class="tpx">—</span></span>
  <span class="tick" id="tick_USDJPY" onclick="selPair(null,'USDJPY')"><span class="tsym">USD/JPY</span><span class="tpx">—</span></span>
  <span class="tick" id="tick_GBPJPY" onclick="selPair(null,'GBPJPY')"><span class="tsym">GBP/JPY</span><span class="tpx">—</span></span>
  <span class="tick" id="tick_AUDUSD" onclick="selPair(null,'AUDUSD')"><span class="tsym">AUD/USD</span><span class="tpx">—</span></span>
  <span class="tick" id="tick_USDCAD" onclick="selPair(null,'USDCAD')"><span class="tsym">USD/CAD</span><span class="tpx">—</span></span>
  <span class="tick" id="tick_XAUUSD" onclick="selPair(null,'XAUUSD')"><span class="tsym">XAU/USD</span><span class="tpx">—</span></span>
</div>

<div class="main">
  <div class="left">
    <div class="sect">
      <div class="slbl">Select Pair</div>
      <div class="pgrid">
        <button class="pb on"  onclick="selPair(this,'EURUSD')">EUR/USD</button>
        <button class="pb"     onclick="selPair(this,'GBPUSD')">GBP/USD</button>
        <button class="pb"     onclick="selPair(this,'USDJPY')">USD/JPY</button>
        <button class="pb"     onclick="selPair(this,'GBPJPY')">GBP/JPY</button>
        <button class="pb"     onclick="selPair(this,'AUDUSD')">AUD/USD</button>
        <button class="pb"     onclick="selPair(this,'XAUUSD')">XAU/USD</button>
      </div>
    </div>

    <div class="sect">
      <div class="slbl">SMC Confluence Score</div>
      <div class="big-score bs-wait" id="bigScore">—</div>
      <div class="bar-bg"><div class="bar-fill" id="barFill" style="width:0%;background:var(--a)"></div></div>
      <div class="blbl"><span>0</span><span>MIN 70%</span><span>100%</span></div>
    </div>

    <div class="sect">
      <div class="slbl">Active Signal</div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.4rem">
        <div class="sig-dir sd-wait" id="sigDir">WAIT</div>
        <span class="cache-tag ct-demo" id="cacheTag">—</span>
      </div>
      <div class="lvgrid">
        <div class="lv"><div class="lvk">Entry</div><div class="lvv lv-e" id="lvE">—</div></div>
        <div class="lv"><div class="lvk">Stop Loss</div><div class="lvv lv-s" id="lvS">—</div></div>
        <div class="lv"><div class="lvk">TP1 · 1.5R</div><div class="lvv lv-t" id="lvT1">—</div></div>
        <div class="lv"><div class="lvk">TP2 · 3.0R</div><div class="lvv lv-t" id="lvT2">—</div></div>
        <div class="lv"><div class="lvk">R:R</div><div class="lvv lv-rr" id="lvRR">—</div></div>
        <div class="lv"><div class="lvk">Trend</div><div class="lvv" id="lvTr" style="color:var(--t)">—</div></div>
      </div>
      <div class="ctags" id="ctags"><span class="ctag neutral">Run scan first</span></div>
    </div>

    <div class="sect">
      <div class="slbl">SMC Zones</div>
      <div class="zlist" id="zlist"><div class="zrow"><span class="zname" style="color:var(--t3)">Scan to detect</span></div></div>
    </div>

    <div class="sect">
      <div class="slbl">Session Stats</div>
      <div class="mgrid">
        <div class="mc"><div class="mck">Signals</div><div class="mcv" id="mSig">0</div></div>
        <div class="mc"><div class="mck">Buy</div><div class="mcv mcg" id="mBuy">0</div></div>
        <div class="mc"><div class="mck">Sell</div><div class="mcv mcr" id="mSell">0</div></div>
        <div class="mc"><div class="mck">Avg Score</div><div class="mcv mca" id="mAvg">—</div></div>
        <div class="mc"><div class="mck">Last Scan</div><div class="mcv" id="mLast" style="font-size:10px;color:var(--t2)">—</div></div>
        <div class="mc"><div class="mck">Mode</div><div class="mcv mcg" id="mMode" style="font-size:10px">LIVE</div></div>
      </div>
    </div>
  </div>

  <div class="right">
    <div class="chart-area">
      <div class="chart-hdr">
        <div>
          <div class="csym" id="csym">EUR/USD</div>
          <div class="cmeta">
            <span id="cBias">Bias: —</span>
            <span id="cStruct">Structure: —</span>
            <span id="cSess">Session: —</span>
          </div>
        </div>
        <div>
          <div class="cpx cup" id="cpx">—</div>
          <div style="font-family:var(--mono);font-size:8px;color:var(--t3);text-align:right" id="cpxSrc">—</div>
        </div>
      </div>
      <div class="chart-wrap"><canvas id="chart" role="img" aria-label="SMC price chart"></canvas></div>
    </div>

    <div class="tbl-wrap">
      <div class="tlbl">All Pairs · Scan Results</div>
      <table><thead><tr>
        <th>Pair</th><th>Signal</th><th>Score</th><th>Entry</th>
        <th>SL (30p)</th><th>TP1 (1.5R)</th><th>TP2 (3R)</th><th>R:R</th><th>Confluences</th>
      </tr></thead>
      <tbody id="tbody"><tr><td colspan="9" style="color:var(--t3);text-align:center;padding:1rem">Click SCAN ALL to load signals</td></tr></tbody>
      </table>
    </div>

    <div class="log" id="logEl"></div>
  </div>
</div>
</div>

<!-- Closed screen -->
<div class="closed-screen" id="closedScr">
  <div class="cs-icon">🔒</div>
  <div class="cs-title">MARKET CLOSED</div>
  <div class="cs-reason" id="csReason">—</div>
  <div class="cs-next" id="csNext"></div>
  <div class="cs-time" id="csTime"></div>
  <div class="sess-row">
    <div class="sess-card"><div class="sc-name">SYDNEY</div><div class="sc-time">21:00–06:00 UTC</div><div class="sc-st" id="ss0">—</div></div>
    <div class="sess-card"><div class="sc-name">TOKYO</div><div class="sc-time">00:00–09:00 UTC</div><div class="sc-st" id="ss1">—</div></div>
    <div class="sess-card"><div class="sc-name">LONDON</div><div class="sc-time">07:00–16:00 UTC</div><div class="sc-st" id="ss2">—</div></div>
    <div class="sess-card"><div class="sc-name">NEW YORK</div><div class="sc-time">12:00–21:00 UTC</div><div class="sc-st" id="ss3">—</div></div>
  </div>
  <button class="demo-btn" onclick="showDemo()">View demo signals anyway</button>
</div>

<!-- Spinner -->
<div class="spin-overlay" id="spinner">
  <div class="spinner"></div>
  <div class="spin-txt">SCANNING</div>
  <div class="spin-sub" id="spinSub">Fetching market data...</div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
const DIGITS={EURUSD:5,GBPUSD:5,USDJPY:3,GBPJPY:3,AUDUSD:5,USDCAD:5,XAUUSD:2};
let pair='EURUSD', allSigs=[], chart=null, isDemoMode=false;
let cntBuy=0,cntSell=0,scores=[];

// ── Clock ──────────────────────────────────────────────────────
function tick(){
  const n=new Date(), p=v=>String(v).padStart(2,'0');
  document.getElementById('clk').textContent=p(n.getUTCHours())+':'+p(n.getUTCMinutes())+':'+p(n.getUTCSeconds())+' UTC';
}
tick(); setInterval(tick,1000);

// ── Market status ─────────────────────────────────────────────
async function checkStatus(){
  try{
    const d=await(await fetch('/api/status',{signal:AbortSignal.timeout(6000)})).json();
    const badge=document.getElementById('mktBadge');
    const lbl=document.getElementById('mktLbl');
    const sNames=['Sydney','Tokyo','London','New York'];
    sNames.forEach((n,i)=>{
      const el=document.getElementById('ss'+i);
      if(!el) return;
      const on=d.sessions.includes(n);
      el.textContent=on?'● ACTIVE':'○ CLOSED';
      el.className='sc-st '+(on?'sc-open':'sc-cls');
    });
    if(d.is_open){
      badge.className='mkt-badge open-b';
      lbl.textContent=(d.sessions[0]||'OPEN')+' SESSION';
      document.getElementById('cSess').textContent='Session: '+(d.sessions.join(' + ')||'—');
      if(!isDemoMode){
        document.getElementById('closedScr').classList.remove('show');
      }
    } else {
      badge.className='mkt-badge closed-b';
      lbl.textContent='MARKET CLOSED';
      document.getElementById('csReason').textContent=d.reason;
      document.getElementById('csNext').textContent=d.next_open||'';
      document.getElementById('csTime').textContent=d.time_utc;
      if(!isDemoMode) document.getElementById('closedScr').classList.add('show');
    }
    return d;
  } catch(e){ return null; }
}

function showDemo(){
  isDemoMode=true;
  document.getElementById('closedScr').classList.remove('show');
  document.getElementById('mMode').textContent='DEMO';
  document.getElementById('mMode').className='mcv mca';
  addLog('Demo mode — using real latest prices with simulated bars','SYS','warn');
  doScan(true);
}

// ── Prices ────────────────────────────────────────────────────
async function loadPrices(){
  try{
    const d=await(await fetch('/api/prices',{signal:AbortSignal.timeout(15000)})).json();
    Object.entries(d.prices||{}).forEach(([p,info])=>{
      const el=document.getElementById('tick_'+p);
      if(!el) return;
      const dp=DIGITS[p]||5; const chg=info.change_pct||0;
      el.innerHTML=`<span class="tsym">${p.slice(0,3)}/${p.slice(3)}</span>`+
        `<span class="tpx">${info.price.toFixed(dp)}</span>`+
        `<span class="${chg>=0?'tup':'tdn'}">${chg>=0?'+':''}${chg.toFixed(2)}%</span>`;
    });
  } catch(e){}
}

// ── Scan ──────────────────────────────────────────────────────
async function doScan(force){
  const btn=document.getElementById('scanBtn');
  btn.disabled=true;
  document.getElementById('spinner').classList.add('show');
  const steps=['Checking market hours...','Fetching OHLCV data...','Detecting Order Blocks...','Scanning FVG zones...','Analysing CHoCH & BOS...','Scoring confluences...','Building signals...'];
  let si=0;
  const iv=setInterval(()=>document.getElementById('spinSub').textContent=steps[Math.min(si++,steps.length-1)],700);
  try{
    const url='/api/scan'+(force?'?force=1':'');
    const d=await(await fetch(url,{signal:AbortSignal.timeout(120000)})).json();
    clearInterval(iv);
    allSigs=d.signals||[];
    buildTable(allSigs);
    allSigs.forEach(s=>{
      if(s.direction==='BUY'){cntBuy++;scores.push(s.score);}
      if(s.direction==='SELL'){cntSell++;scores.push(s.score);}
      const t=s.direction==='BUY'?'buy':s.direction==='SELL'?'sell':'wait';
      addLog(`${s.direction} | ${s.score_pct}% | ${(s.conf||[]).join(', ')||'—'}`,s.pair,t);
      if(s.direction!=='WAIT') showToast(`${s.direction==='BUY'?'📈':'📉'} ${s.pair} ${s.direction} — ${s.score_pct}%`,s.direction==='BUY'?'tb':'ts');
    });
    document.getElementById('mSig').textContent=allSigs.length;
    document.getElementById('mBuy').textContent=cntBuy;
    document.getElementById('mSell').textContent=cntSell;
    document.getElementById('mAvg').textContent=scores.length?Math.round(scores.reduce((a,b)=>a+b,0)/scores.length*100)+'%':'—';
    const now=new Date();
    document.getElementById('mLast').textContent=String(now.getUTCHours()).padStart(2,'0')+':'+String(now.getUTCMinutes()).padStart(2,'0')+' UTC';
    const active=allSigs.find(s=>s.pair===pair);
    if(active){ updateSignal(active); loadChart(pair); }
    addLog(`Scan complete — ${allSigs.length} pairs | ${d.ts}`.slice(0,80),'SYS','info');
  } catch(e){
    clearInterval(iv);
    addLog('Scan failed: '+e.message,'ERR','warn');
    showToast('Scan failed — check server is running','tw');
  } finally {
    document.getElementById('spinner').classList.remove('show');
    btn.disabled=false;
  }
}

// ── Pair select ───────────────────────────────────────────────
function selPair(btn,p){
  pair=p;
  document.querySelectorAll('.pb').forEach(b=>b.classList.remove('on'));
  if(btn) btn.classList.add('on');
  else{
    document.querySelectorAll('.pb').forEach(b=>{
      if(b.textContent.replace('/','')===p.slice(0,3)+p.slice(3)||b.onclick.toString().includes(p)) b.classList.add('on');
    });
  }
  const sig=allSigs.find(s=>s.pair===p);
  if(sig) updateSignal(sig);
  loadChart(p);
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
  document.getElementById('lvS').textContent=dir!=='WAIT'?f(sig.sl):'—';
  document.getElementById('lvT1').textContent=dir!=='WAIT'?f(sig.tp1):'—';
  document.getElementById('lvT2').textContent=dir!=='WAIT'?f(sig.tp2):'—';
  document.getElementById('lvRR').textContent=dir!=='WAIT'?'1:'+sig.rr:'—';
  document.getElementById('lvTr').textContent=(sig.trend||'—').toUpperCase();
  // Cache tag
  const ct=document.getElementById('cacheTag');
  if(sig.demo){ct.textContent='DEMO';ct.className='cache-tag ct-demo';}
  else if(sig.cached){ct.textContent='CACHED';ct.className='cache-tag ct-cached';}
  else{ct.textContent='LIVE';ct.className='cache-tag ct-live';}
  // Confluences
  const tags=document.getElementById('ctags');
  tags.innerHTML=sig.conf&&sig.conf.length
    ?sig.conf.map(c=>`<span class="ctag${dir==='SELL'?' bear':''}">${c}</span>`).join('')
    :'<span class="ctag neutral">No confluence</span>';
  // Chart meta
  document.getElementById('csym').textContent=sig.pair.slice(0,3)+'/'+sig.pair.slice(3);
  document.getElementById('cBias').textContent='Bias: '+(sig.trend||'—').toUpperCase();
  const struct=sig.choch?`CHoCH (${sig.choch_dir})`:sig.bos?`BOS (${sig.bos_dir})`:'None';
  document.getElementById('cStruct').textContent='Structure: '+struct;
  // Zones
  buildZones(sig.zones||[],dp);
}

function buildZones(zones,dp){
  const zl=document.getElementById('zlist');
  if(!zones.length){zl.innerHTML='<div class="zrow"><span class="zname" style="color:var(--t3)">No zones detected</span></div>';return;}
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
  if(!sigs.length){tb.innerHTML='<tr><td colspan="9" style="color:var(--t3);text-align:center;padding:1rem">No signals</td></tr>';return;}
  tb.innerHTML=sigs.map(s=>{
    const dp=DIGITS[s.pair]||5; const dc=s.direction==='BUY'?'db':s.direction==='SELL'?'ds':'dw';
    const pc=s.score_pct>70?'sph':s.score_pct>=50?'spm':'spl';
    const f=v=>v&&v!==0?v.toFixed(dp):'—';
    return `<tr>
      <td style="color:var(--t);font-weight:500">${s.pair.slice(0,3)+'/'+s.pair.slice(3)}</td>
      <td class="${dc}">${s.direction}</td>
      <td><span class="sp ${pc}">${s.score_pct}%</span></td>
      <td>${s.direction!=='WAIT'?f(s.entry):'—'}</td>
      <td>${s.direction!=='WAIT'?f(s.sl):'—'}</td>
      <td>${s.direction!=='WAIT'?f(s.tp1):'—'}</td>
      <td>${s.direction!=='WAIT'?f(s.tp2):'—'}</td>
      <td>${s.direction!=='WAIT'?'1:'+s.rr:'—'}</td>
      <td style="color:var(--t2);font-size:9px">${(s.conf||[]).join(' · ')||'—'}</td>
    </tr>`;
  }).join('');
}

// ── Chart ─────────────────────────────────────────────────────
async function loadChart(p){
  try{
    const d=await(await fetch(`/api/bars/${p}?tf=H1&limit=80`,{signal:AbortSignal.timeout(15000)})).json();
    if(d.bars&&d.bars.length) buildChart(p,d.bars,d.demo);
  } catch(e){}
}

function buildChart(p,bars,isDemo){
  const dp=DIGITS[p]||5;
  const labels=bars.map((_,i)=>i);
  const colors=bars.map(b=>b.close>=b.open?'rgba(0,255,179,.75)':'rgba(255,61,90,.75)');
  const sig=allSigs.find(s=>s.pair===p);
  const ds=[{type:'bar',data:bars.map((b,i)=>({x:i,y:[b.open,b.close]})),backgroundColor:colors,borderWidth:0,barPercentage:.55}];
  if(sig&&sig.zones){
    sig.zones.filter(z=>z.color==='green'&&z.lo).forEach(z=>{
      ds.push({type:'line',data:labels.map(()=>z.lo),borderColor:'rgba(0,255,179,.4)',borderWidth:1,borderDash:[4,3],pointRadius:0,tension:0});
      ds.push({type:'line',data:labels.map(()=>z.hi),borderColor:'rgba(0,255,179,.15)',borderWidth:1,borderDash:[2,5],pointRadius:0,tension:0});
    });
    sig.zones.filter(z=>z.color==='red'&&z.hi).forEach(z=>{
      ds.push({type:'line',data:labels.map(()=>z.hi),borderColor:'rgba(255,61,90,.4)',borderWidth:1,borderDash:[4,3],pointRadius:0,tension:0});
    });
    sig.zones.filter(z=>z.color==='amber'&&z.hi).forEach(z=>{
      ds.push({type:'line',data:labels.map(()=>(z.hi+z.lo)/2),borderColor:'rgba(255,184,0,.4)',borderWidth:1,borderDash:[2,4],pointRadius:0,tension:0});
    });
  }
  const last=bars[bars.length-1]?.close||0;
  document.getElementById('cpx').textContent=last.toFixed(dp);
  document.getElementById('cpxSrc').textContent=isDemo?'DEMO · real price':'LIVE · yfinance';
  if(chart) chart.destroy();
  chart=new Chart(document.getElementById('chart'),{
    type:'bar',data:{labels,datasets:ds},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:300},
      plugins:{legend:{display:false},tooltip:{enabled:false}},
      scales:{x:{display:false},y:{grid:{color:'rgba(255,255,255,.04)'},
        ticks:{color:'#2d404f',font:{size:9,family:'JetBrains Mono'},maxTicksLimit:6,
               callback:v=>typeof v==='number'?v.toFixed(dp>3?4:dp<3?1:2):v}}}}
  });
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
  setTimeout(()=>t.classList.remove('show'),4000);
}

// ── Init ──────────────────────────────────────────────────────
(async function(){
  addLog('Dashboard ready','SYS','info');
  // Load prices silently in background
  loadPrices();
  // Check market status
  const st=await checkStatus();
  if(st&&st.is_open){
    addLog('Market OPEN — click SCAN ALL for live signals','SYS','info');
    // Load chart immediately
    loadChart('EURUSD');
  } else if(st&&st.is_closed){
    addLog('Market closed: '+st.reason,'SYS','warn');
  }
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
        return  # skip on Render server
    time.sleep(1.2)
    webbrowser.open("http://localhost:5000")

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
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
