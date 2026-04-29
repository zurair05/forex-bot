# OANDA Setup — 5-minute guide

The bot now prefers **OANDA** for live and historical forex data. The free
demo account is enough — no funding, no KYC, just an email signup.
yfinance still works as a fallback if you skip this; the server detects
the missing env vars and falls back silently.

## Why bother

- Cleaner data (no random gaps, fewer "delisted" errors than yfinance)
- Real **bid/ask spread** on every tick — dashboard surfaces it as
  `spread_pips` so you know the true cost of every signal
- Same feed your real OANDA account would see → backtests match live
- Stable & documented (vs scraping TradingView, which violates ToS)

## Step 1 — Sign up for the free demo

1. Go to <https://www.oanda.com/demo-account/> (pick your region).
2. Email + password — that's it.
3. Verify email and log in to <https://fxtrade.oanda.com/>.

## Step 2 — Generate a personal access token

1. While logged in, go to **Manage API Access** (search "API access" in
   the dashboard). Direct link (region-dependent):
   <https://www.oanda.com/account/tpa/personal_token>
2. Click **Generate** and copy the long token string. **You can only see
   it once** — paste it somewhere safe.

## Step 3 — Find your Account ID

In the same dashboard, your Account ID looks like
`101-001-12345678-001`. It's shown next to your demo account name.

## Step 4 — Set the environment variables

### Windows PowerShell (one-shot, expires when you close the window)

```powershell
$env:OANDA_API_KEY    = "<your-token>"
$env:OANDA_ACCOUNT_ID = "101-001-12345678-001"
$env:OANDA_ENV        = "practice"     # or "live" if you switch to a real account
python server.py
```

### Windows — make it permanent

```powershell
[System.Environment]::SetEnvironmentVariable("OANDA_API_KEY", "<your-token>", "User")
[System.Environment]::SetEnvironmentVariable("OANDA_ACCOUNT_ID", "101-001-12345678-001", "User")
[System.Environment]::SetEnvironmentVariable("OANDA_ENV", "practice", "User")
```

(open a new shell to pick them up)

### Linux / macOS

```bash
export OANDA_API_KEY=<your-token>
export OANDA_ACCOUNT_ID=101-001-12345678-001
export OANDA_ENV=practice
python server.py
```

### Render / production

Add the same three keys under **Environment Variables** in your Render
service settings.

## What you should see in the logs

```
OANDA fetch_bars(EURUSD,H1) ... OK
[AUTO-SCAN] EURUSD: BUY 78% [NEW]   spread=0.7p
```

If you instead see `OANDA fetch_bars(...) failed: 401`, your token is
wrong. `403` usually means wrong env (`practice` vs `live`). Anything
else and the server quietly falls back to yfinance — `/api/scan` will
still work, you just lose spread data.

## Going to a real account later

When you switch to a funded account, generate a *new* token from the live
dashboard (<https://www.oanda.com/account/tpa/personal_token>), update
`OANDA_ACCOUNT_ID` to the live one, and set `OANDA_ENV=live`. The bot
itself doesn't change.

## Rate limits

20 requests/sec, 100 connections. The bot caches bars for 5 min and
prices for 1 min, so real-world load is well under the limit even with
all 7 pairs.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `OANDA fetch_bars failed: 401` | Bad / expired token |
| `OANDA fetch_bars failed: 403` | Token is for a different env (live vs practice) |
| `OANDA fetch_prices failed` but bars work | Account ID missing or wrong |
| Server falls back to yfinance silently | `OANDA_API_KEY` not set, or `requests` not installed |

## Reverting to yfinance

Just unset the env vars (or remove the token). The fallback is automatic.
