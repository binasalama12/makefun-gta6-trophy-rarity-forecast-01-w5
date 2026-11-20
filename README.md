# GTA VI Trophy Rarity Forecast — wallet w5 (Robust Conservative)

Forecasts each of GTA VI's 55 PS5 trophies' Exophase earn rate (%) at the
case's target time, as five quantiles (5%, 25%, 50%, 75%, 95%) on the logit
scale.

## Model

**Two branches, chosen per trophy:**

1. **Stale page** — that trophy's rates never moved *and* the
   players-tracked count never moved across the whole history. The page has
   not refreshed at all, so a refresh before the target is likely, and a
   refresh dilutes every rate (new players arrive faster than new earners).
   Centre on `last_rate * 0.35` with a wide `1.2`-logit spread.
2. **Live page** — centre on the **median of the last 6 hours** of reads
   (robust to a single noisy read), plus a damped trend: the logit change
   between the median of the first half and the median of the second half
   of the last 24 hours, extrapolated by `horizon_days` and capped to
   ±1.0 logit. Spread is
   `max(0.5*sqrt(horizon_days), observed 24h logit volatility *
   sqrt(horizon_days)) + 0.3`, floored at 0.35 and capped at 2.0.

Quantiles use normal z-scores (-1.645, -0.674, 0, 0.674, 1.645) on the logit
scale, mapped back to percent, clamped to [0.01, 99.99], rounded half-up to
4 decimals, and forced non-decreasing.

## Why this shape

Exophase refreshes a game's counts on its own schedule, not with every
player's trophy. On launch night the page sat unchanged (638 players
tracked, every rate identical) for hours, then refreshed to 3,868 players
with every rate but the 100% one lower. A model that forecasts a flat
history flat pays for the whole refresh; this one anticipates dilution and
widens its spread. Over a multi-day final horizon, the trend branch takes
over: earn rates drift as players progress, and the spread grows with both
the horizon and the volatility actually observed.

## Run

```sh
MAKEFUN_INPUT_DIR=<input dir> MAKEFUN_OUTPUT_DIR=<output dir> python3 main.py
```

Reads every `*.in` in the input directory (else `/input`) and writes
`<id>.out` per case to the output directory (else `/output`).

Python 3.13, standard library only, no network, no randomness — two runs
produce identical bytes. Runtime is well under the 8 s/case limit.

## Files

- `main.py` — the whole forecaster.
- `README.md` — this file.
- `LICENSE` — MIT.
