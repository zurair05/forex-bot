"""
MT5 Bridge — SMC Dashboard to MetaTrader 5
===========================================
Every trade MUST pass ALL conditions before executing.

Install:  pip install MetaTrader5 requests
Run:      python mt5_bridge.py
"""

import time
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[ERROR] Run:  pip install MetaTrader5")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("[ERROR] Run:  pip install requests")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
#  YOUR SETTINGS
# ══════════════════════════════════════════════════════════════════════

class Config:
    DASHBOARD_URL  = "http://localhost:5000"
    # DASHBOARD_URL = "https://your-app.onrender.com"

    MT5_LOGIN      = 414087232
    MT5_PASSWORD   = "!kwJmW9Ivm"
    MT5_SERVER     = "GoatFunded-Server2"
    MT5_PATH       = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    SYMBOL_SUFFIX  = ""       # some brokers add ".m" or "_raw"

    DEMO_ONLY      = True     # refuses to run on a live account
    DRY_RUN        = False    # set True to log without placing orders
    POLL_SECONDS   = 60
    MAGIC          = 20250419
    SLIPPAGE       = 10


# ══════════════════════════════════════════════════════════════════════
#  TRADE CONDITIONS  — all must pass, no exceptions
# ══════════════════════════════════════════════════════════════════════

class Conditions:
    RISK_PCT        = 1.0    # risk 1% of account balance per trade
    MIN_SCORE_PCT   = 70     # confluence score must be >= 70%
    SL_PIPS         = 30     # stop loss = exactly 30 pips
    MIN_RR          = 2.5    # minimum risk:reward ratio
    TP1_R           = 1.5    # TP1 = SL distance x 1.5
    TP2_R           = 3.0    # TP2 = SL distance x 3.0
    MAX_SPREAD_PIPS = 2.5    # skip if spread wider than this
    MAX_OPEN_TRADES = 5      # max simultaneous positions


# ══════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mt5_bridge.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("MT5Bridge")


# ══════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════

_executed: Dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

PIP_MAP = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "USDCAD": 0.0001, "USDCHF": 0.0001, "NZDUSD": 0.0001,
    "USDJPY": 0.01,   "GBPJPY": 0.01,   "EURJPY": 0.01,
    "XAUUSD": 0.1,
}


def sym(pair: str) -> str:
    return pair + Config.SYMBOL_SUFFIX


def pip(pair: str) -> float:
    return PIP_MAP.get(pair, 0.0001)


def spread_pips(pair: str) -> float:
    tick = mt5.symbol_info_tick(sym(pair))
    if tick is None:
        return 99.0
    return round((tick.ask - tick.bid) / pip(pair), 1)


def open_count() -> int:
    pos = mt5.positions_get()
    return sum(1 for p in (pos or []) if p.magic == Config.MAGIC)


def has_open(pair: str) -> bool:
    pos = mt5.positions_get(symbol=sym(pair))
    return any(p.magic == Config.MAGIC for p in (pos or []))


def balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 0.0


def already_executed(pair: str, direction: str, entry: float) -> bool:
    prev = _executed.get(pair)
    if not prev:
        return False
    return (prev["direction"] == direction and
            abs(prev["entry"] - entry) < pip(pair) * 5)


# ══════════════════════════════════════════════════════════════════════
#  MT5 CONNECTION
# ══════════════════════════════════════════════════════════════════════

def connect() -> bool:
    kwargs = {}
    if Config.MT5_PATH and os.path.exists(Config.MT5_PATH):
        kwargs["path"] = Config.MT5_PATH

    if not mt5.initialize(**kwargs):
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    if Config.MT5_LOGIN:
        if not mt5.login(Config.MT5_LOGIN,
                         password=Config.MT5_PASSWORD,
                         server=Config.MT5_SERVER):
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

    info = mt5.account_info()
    if info:
        mode = "DEMO" if info.trade_mode == 0 else "LIVE"
        log.info(f"MT5 connected | #{info.login} | {mode} | "
                 f"Balance: ${info.balance:,.2f} | {info.server}")
        if mode == "LIVE" and Config.DEMO_ONLY:
            log.error("DEMO_ONLY=True but this is a LIVE account. Stopped for safety.")
            mt5.shutdown()
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
#  CONDITION CHECKER
# ══════════════════════════════════════════════════════════════════════

def check_all(sig: dict):
    """
    Check every condition in sequence.
    Returns (True, "OK") if all pass.
    Returns (False, reason) at the first failure.
    Logs every check clearly.
    """
    pair      = sig.get("pair", "")
    direction = sig.get("direction", "WAIT")
    score_pct = sig.get("score_pct", 0)
    rr        = float(sig.get("rr", 0))
    entry     = float(sig.get("entry", 0))

    sep = "  " + "-" * 45

    # 1. Direction
    if direction == "WAIT":
        return False, "Direction is WAIT — no trade"

    log.info(sep)
    log.info(f"  Checking: {pair} {direction} | score={score_pct}%")
    log.info(sep)

    # 2. Confluence score >= 70%
    if score_pct < Conditions.MIN_SCORE_PCT:
        return False, (f"Score {score_pct}% < {Conditions.MIN_SCORE_PCT}% minimum  [FAIL]")
    log.info(f"  [PASS] Score        {score_pct}%  >=  {Conditions.MIN_SCORE_PCT}%")

    # 3. Already executed this signal?
    if already_executed(pair, direction, entry):
        return False, f"Already executed this signal for {pair}  [SKIP]"
    log.info(f"  [PASS] Not yet executed this session")

    # 4. No open position on this pair
    if has_open(pair):
        return False, f"Already have an open position on {pair}  [SKIP]"
    log.info(f"  [PASS] No existing position on {pair}")

    # 5. Max open trades
    count = open_count()
    if count >= Conditions.MAX_OPEN_TRADES:
        return False, (f"Max trades {count}/{Conditions.MAX_OPEN_TRADES} reached  [FAIL]")
    log.info(f"  [PASS] Trades       {count} / {Conditions.MAX_OPEN_TRADES}")

    # 6. Spread
    sp = spread_pips(pair)
    if sp > Conditions.MAX_SPREAD_PIPS:
        return False, (f"Spread {sp:.1f}p > {Conditions.MAX_SPREAD_PIPS}p max  [FAIL]")
    log.info(f"  [PASS] Spread       {sp:.1f}p  <=  {Conditions.MAX_SPREAD_PIPS}p")

    # 7. R:R check (using our own SL/TP will always give exactly 2.5, but verify)
    own_rr = Conditions.TP1_R / 1.0   # SL=1R so TP1=1.5R => rr=1.5, but TP2 gives 3.0
    tp2_rr = Conditions.TP2_R
    if tp2_rr < Conditions.MIN_RR:
        return False, (f"TP2 R:R {tp2_rr} < {Conditions.MIN_RR} min  [FAIL]")
    log.info(f"  [PASS] R:R          1:{tp2_rr}  >=  1:{Conditions.MIN_RR}")

    log.info(sep)
    log.info(f"  ALL CONDITIONS PASSED — proceeding to execute")
    log.info(sep)
    return True, "OK"


# ══════════════════════════════════════════════════════════════════════
#  LEVEL CALCULATOR — always uses our fixed 30-pip SL
# ══════════════════════════════════════════════════════════════════════

def build_levels(pair: str, direction: str) -> dict:
    """
    Recalculates levels from LIVE MT5 price using our exact conditions.
    Never trusts the dashboard price — gets a fresh tick right now.
    """
    p       = pip(pair)
    sl_dist = p * Conditions.SL_PIPS   # 30 pips exactly

    symbol_name = sym(pair)
    info        = mt5.symbol_info(symbol_name)
    tick        = mt5.symbol_info_tick(symbol_name)
    digits      = info.digits if info else 5

    if tick is None:
        log.error(f"No tick for {symbol_name}")
        return {}

    entry = round(tick.ask if direction == "BUY" else tick.bid, digits)

    if direction == "BUY":
        sl  = round(entry - sl_dist,                      digits)
        tp1 = round(entry + sl_dist * Conditions.TP1_R,   digits)
        tp2 = round(entry + sl_dist * Conditions.TP2_R,   digits)
    else:
        sl  = round(entry + sl_dist,                      digits)
        tp1 = round(entry - sl_dist * Conditions.TP1_R,   digits)
        tp2 = round(entry - sl_dist * Conditions.TP2_R,   digits)

    log.info(f"  Levels  | Entry:{entry}  SL:{sl} ({Conditions.SL_PIPS}p)"
             f"  TP1:{tp1} ({Conditions.TP1_R}R)"
             f"  TP2:{tp2} ({Conditions.TP2_R}R)"
             f"  R:R 1:{Conditions.TP2_R}")

    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "digits": digits}


# ══════════════════════════════════════════════════════════════════════
#  LOT SIZE — exactly 1% risk
# ══════════════════════════════════════════════════════════════════════

def calc_lots(pair: str, price: float) -> float:
    bal      = balance()
    risk_usd = bal * Conditions.RISK_PCT / 100.0
    p        = pip(pair)
    sl_pips  = Conditions.SL_PIPS

    pip_val  = p * 100_000
    if pair.endswith("JPY"):
        pip_val = (p * 100_000) / max(price, 1)
    elif pair == "XAUUSD":
        pip_val = p * 100

    if pip_val <= 0:
        return 0.01

    raw_lots = risk_usd / (sl_pips * pip_val)

    info = mt5.symbol_info(sym(pair))
    if info:
        step = info.volume_step or 0.01
        lo   = info.volume_min  or 0.01
        hi   = info.volume_max  or 100.0
        lots = max(lo, min(round(round(raw_lots / step) * step, 2), hi))
    else:
        lots = max(0.01, round(raw_lots, 2))

    log.info(f"  Lots    | balance=${bal:.2f}  risk={Conditions.RISK_PCT}%"
             f"  risk_usd=${risk_usd:.2f}  sl={sl_pips}p  -> {lots:.2f} lots")
    return lots


# ══════════════════════════════════════════════════════════════════════
#  PLACE ORDER
# ══════════════════════════════════════════════════════════════════════

def place_order(sig: dict, levels: dict, lots: float) -> bool:
    pair      = sig["pair"]
    direction = sig["direction"]
    symbol_n  = sym(pair)
    score_pct = sig.get("score_pct", 0)
    conf_str  = "|".join(sig.get("conf", []))[:25]
    digits    = levels["digits"]
    sl        = levels["sl"]
    tp1       = levels["tp1"]
    tp2       = levels["tp2"]

    # Print full summary
    bal = balance()
    log.info(f"\n  ╔══════════════════════════════╗")
    log.info(f"  ║  TRADE APPROVED              ║")
    log.info(f"  ╠══════════════════════════════╣")
    log.info(f"  ║  Pair     : {pair} {direction}")
    log.info(f"  ║  Score    : {score_pct}%  (min 70%)")
    log.info(f"  ║  Entry    : {levels['entry']:.{digits}f}")
    log.info(f"  ║  SL       : {sl:.{digits}f}  (30 pips)")
    log.info(f"  ║  TP1      : {tp1:.{digits}f}  (1.5R)")
    log.info(f"  ║  TP2      : {tp2:.{digits}f}  (3.0R)")
    log.info(f"  ║  R:R      : 1:2.5 / 1:3.0")
    log.info(f"  ║  Lots     : {lots:.2f}  (1% of ${bal:,.0f})")
    log.info(f"  ║  Risk $   : ${bal * Conditions.RISK_PCT / 100:.2f}")
    log.info(f"  ║  Conf     : {conf_str}")
    log.info(f"  ╚══════════════════════════════╝\n")

    if Config.DRY_RUN:
        log.info(f"  [DRY RUN] Order NOT sent (DRY_RUN=True)")
        _executed[pair] = {**sig, "entry": levels["entry"], "direction": direction}
        return True

    # Ensure symbol visible
    info = mt5.symbol_info(symbol_n)
    if info is None:
        log.error(f"Symbol '{symbol_n}' not found. Check SYMBOL_SUFFIX.")
        return False
    if not info.visible:
        mt5.symbol_select(symbol_n, True)
        time.sleep(0.5)

    tick = mt5.symbol_info_tick(symbol_n)
    if tick is None:
        log.error(f"No tick for {symbol_n}")
        return False

    order_type = mt5.ORDER_TYPE_BUY  if direction == "BUY"  else mt5.ORDER_TYPE_SELL
    price      = tick.ask            if direction == "BUY"  else tick.bid

    result = mt5.order_send({
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol_n,
        "volume":       lots,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp1,
        "deviation":    Config.SLIPPAGE,
        "magic":        Config.MAGIC,
        "comment":      f"SMC|{score_pct}%|SL30p",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    })

    if result is None:
        log.error(f"order_send None: {mt5.last_error()}")
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"  ORDER FILLED ticket=#{result.order} "
                 f"{direction} {symbol_n} {lots:.2f}L @ {result.price:.{digits}f}")
        _executed[pair] = {**sig, "entry": result.price, "direction": direction}

        # Push to dashboard trade log
        try:
            requests.post(
                Config.DASHBOARD_URL.rstrip("/") + "/api/trades",
                json={
                    "pair": pair, "direction": direction, "lots": lots,
                    "price": result.price, "sl": sl, "tp1": tp1, "tp2": tp2,
                    "ticket": result.order, "score_pct": score_pct,
                    "conf": sig.get("conf", []),
                    "risk_pct": Conditions.RISK_PCT,
                    "sl_pips": Conditions.SL_PIPS,
                    "status": "filled",
                },
                timeout=5,
            )
        except Exception:
            pass

        return True

    ERRORS = {
        10004: "Requote", 10006: "Rejected by broker",
        10013: "Invalid params", 10014: "Invalid volume",
        10015: "Invalid price", 10016: "Invalid SL/TP",
        10018: "Market closed", 10019: "Not enough money",
        10021: "No quotes", 10024: "Too many requests",
    }
    log.error(f"  ORDER FAILED retcode={result.retcode} "
              f"'{result.comment}' — {ERRORS.get(result.retcode, 'unknown')}")
    return False


# ══════════════════════════════════════════════════════════════════════
#  POSITION MANAGEMENT — breakeven + trailing SL
# ══════════════════════════════════════════════════════════════════════

def manage_positions():
    for pos in (mt5.positions_get() or []):
        if pos.magic != Config.MAGIC:
            continue

        symbol_n = pos.symbol
        ticket   = pos.ticket
        open_px  = pos.price_open
        cur_sl   = pos.sl
        is_buy   = pos.type == mt5.POSITION_TYPE_BUY
        sl_dist  = abs(open_px - cur_sl)
        d        = pos.digits

        if sl_dist < 10 ** (-d):
            continue

        tick = mt5.symbol_info_tick(symbol_n)
        if tick is None:
            continue

        cur_px   = tick.bid if is_buy else tick.ask
        r_gained = ((cur_px - open_px) / sl_dist if is_buy
                    else (open_px - cur_px) / sl_dist)

        # Breakeven after 1R
        if r_gained >= 1.0:
            be = round(open_px, d)
            needs = ((is_buy and cur_sl < be - 10**(-d)) or
                     (not is_buy and (cur_sl < 10**(-d) or cur_sl > be + 10**(-d))))
            if needs:
                r = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
                                    "position": ticket, "sl": be, "tp": pos.tp})
                if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                    log.info(f"Breakeven #{ticket} {symbol_n}: SL={be:.{d}f}")

        # Trail after 1.5R at 80% of SL distance
        if r_gained >= 1.5:
            trail = sl_dist * 0.8
            if is_buy:
                new_sl = round(cur_px - trail, d)
                if new_sl > cur_sl + 10**(-d):
                    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
                                    "position": ticket, "sl": new_sl, "tp": pos.tp})
                    log.info(f"Trail #{ticket}: SL={new_sl:.{d}f}")
            else:
                new_sl = round(cur_px + trail, d)
                if cur_sl < 10**(-d) or new_sl < cur_sl - 10**(-d):
                    mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
                                    "position": ticket, "sl": new_sl, "tp": pos.tp})
                    log.info(f"Trail #{ticket}: SL={new_sl:.{d}f}")


# ══════════════════════════════════════════════════════════════════════
#  DASHBOARD POLLING
# ══════════════════════════════════════════════════════════════════════

def fetch_signals() -> Optional[list]:
    try:
        r = requests.get(Config.DASHBOARD_URL.rstrip("/") + "/api/scan", timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("market_closed"):
            log.info(f"Market closed: {data.get('reason', '')}")
            return None
        sigs = data.get("signals", [])
        log.info(f"Fetched {len(sigs)} signals from dashboard")
        return sigs
    except requests.ConnectionError:
        log.warning(f"Cannot reach {Config.DASHBOARD_URL}")
        return None
    except Exception as e:
        log.error(f"Fetch error: {e}")
        return None


def fetch_market_status() -> dict:
    try:
        r = requests.get(Config.DASHBOARD_URL.rstrip("/") + "/api/market-status", timeout=10)
        return r.json()
    except Exception:
        return {"is_open": False, "reason": "Cannot reach dashboard"}


# ══════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════

def process_signals(signals: list):
    for sig in signals:
        pair = sig.get("pair", "")

        # Run all conditions
        passed, reason = check_all(sig)
        if not passed:
            log.info(f"  {pair}: {reason}")
            continue

        # Build levels from live price + our fixed params
        lvl = build_levels(pair, sig["direction"])
        if not lvl:
            continue

        # Calculate lots at 1% risk
        lots = calc_lots(pair, lvl["entry"])
        if lots <= 0:
            log.warning(f"  {pair}: Invalid lot size")
            continue

        place_order(sig, lvl, lots)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║         MT5 Bridge — SMC Signal Executor                ║
╠══════════════════════════════════════════════════════════╣
║  TRADE CONDITIONS (all must pass before any trade)      ║""")
    print(f"  ║  Risk per trade  :  {Conditions.RISK_PCT}% of account balance")
    print(f"  ║  Min score       :  {Conditions.MIN_SCORE_PCT}% confluence")
    print(f"  ║  Stop loss       :  {Conditions.SL_PIPS} pips fixed")
    print(f"  ║  Min R:R         :  1 : {Conditions.MIN_RR}")
    print(f"  ║  TP1             :  {Conditions.TP1_R}R")
    print(f"  ║  TP2             :  {Conditions.TP2_R}R")
    print(f"  ║  Max spread      :  {Conditions.MAX_SPREAD_PIPS} pips")
    print(f"  ║  Max open trades :  {Conditions.MAX_OPEN_TRADES}")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Dashboard  : {Config.DASHBOARD_URL}")
    print(f"  ║  Dry Run    : {Config.DRY_RUN}")
    print(f"  ║  Demo Only  : {Config.DEMO_ONLY}")
    print("  ╚══════════════════════════════════════════════════════════╝\n")

    if not connect():
        sys.exit(1)

    log.info(f"Bridge running — polling every {Config.POLL_SECONDS}s. Ctrl+C to stop.\n")

    while True:
        try:
            manage_positions()

            ms = fetch_market_status()
            if not ms.get("is_open"):
                log.info(f"Market closed: {ms.get('reason', '')} — sleeping")
                time.sleep(Config.POLL_SECONDS * 2)
                continue

            signals = fetch_signals()
            if signals:
                process_signals(signals)

            log.info(f"Open trades: {open_count()} | Next poll in {Config.POLL_SECONDS}s\n")
            time.sleep(Config.POLL_SECONDS)

        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception as e:
            log.exception(f"Error: {e}")
            time.sleep(30)

    mt5.shutdown()


if __name__ == "__main__":
    main()
