# LLM Trader (Gemini)

This project includes an optional LLM-driven “trader” loop that can propose **PLACE / MODIFY / CANCEL** actions as strict JSON, which are then executed through the existing OMS (so paper-only enforcement + send-gates still apply).

## Safety model

- Paper-only is enforced by `IBKRBroker` (it refuses non-`DU…` accounts).
- All order execution goes through `OrderManager`, so IBKR sends still require:
  - `TRADING_DRY_RUN=false`
  - `TRADING_LIVE_ENABLED=true`
  - `TRADING_ORDER_TOKEN` set
  - `--confirm-token` must match `TRADING_ORDER_TOKEN`
- The LLM trader adds its own safety rails:
  - `LLM_ALLOWED_SYMBOLS` allowlist is **required**
  - `LLM_MAX_ORDERS_PER_TICK` and `LLM_MAX_QTY` caps

## Environment variables

- `LLM_ENABLED=true`
- `LLM_PROVIDER=gemini`
- `GEMINI_API_KEY=...`
- `GEMINI_MODEL=gemini-3` (override if you want)
- `LLM_USE_GOOGLE_SEARCH=false|true` (adds `tools:[{googleSearch:{}}]` to Gemini requests)
- `LLM_ALLOWED_KINDS=STK`
- `LLM_ALLOWED_SYMBOLS=AAPL,SPY,...`
- `LLM_MAX_ORDERS_PER_TICK=3`
- `LLM_MAX_QTY=10.0`

## Run (sim broker)

```bash
LLM_ENABLED=true LLM_PROVIDER=gemini GEMINI_API_KEY=... \
LLM_ALLOWED_SYMBOLS=AAPL \
python3 -m trading_algo.cli llm-run --broker sim --once
```

## Terminal chat (interactive)

```bash
LLM_ENABLED=true LLM_PROVIDER=gemini GEMINI_API_KEY=... \
LLM_ALLOWED_SYMBOLS=AAPL \
python3 -m trading_algo.cli chat --broker sim
```

## Run (IBKR paper)

```bash
export TRADING_DRY_RUN=false
export TRADING_LIVE_ENABLED=true
export TRADING_ORDER_TOKEN=I_UNDERSTAND_PAPER_TRADING

export LLM_ENABLED=true
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=...
export LLM_ALLOWED_SYMBOLS=AAPL

python3 -m trading_algo.cli --confirm-token "$TRADING_ORDER_TOKEN" llm-run --broker ibkr --once
```
