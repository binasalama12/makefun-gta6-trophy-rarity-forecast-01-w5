"""GTA VI PS5 trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Wallet w5 model: ROBUST CONSERVATIVE.

Idea
----
Exophase refreshes a game's counts on its own schedule, so the page a case's
history shows can be stale: every read identical, players tracked frozen.
A refresh between the checkpoint and the target dilutes every rate (new
players arrive faster than new earners), so a flat history must NOT be
forecast flat. When the history does move, extrapolate its recent trend,
damped, and let the spread follow the observed volatility.

Two branches, chosen per trophy:

* Stale page (that trophy's rates never moved AND the players-tracked count
  never moved across the whole history): the page has not refreshed at all,
  so a refresh before the target is likely. Centre on last_rate * 0.35
  (dilution) with a wide 1.2-logit spread.
* Live page: centre on the median of the last 6 hours of reads (robust to a
  single noisy read), plus a damped trend: the logit change between the
  median of the first half and the median of the second half of the last
  24 hours, extrapolated by horizon_days and capped to +/-1.0 logit.
  Spread = max(0.5 * sqrt(horizon_days),
               observed 24h logit volatility * sqrt(horizon_days)) + 0.3,
  floored at 0.35 and capped at 2.0.

Quantiles are normal z-scores (-1.645, -0.674, 0, 0.674, 1.645) applied on
the logit scale, mapped back to percent, clamped to [0.01, 99.99], rounded
half-up to 4 decimals, and forced non-decreasing.

Python 3.13, standard library only, no network, no randomness: two runs
produce identical bytes.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import median, pstdev

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in output order.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at those levels.
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Stale-page branch.
DILUTION_FACTOR = 0.35
FLAT_SPREAD = 1.2
# Live-page branch.
RECENT_WINDOW = timedelta(hours=6)
TREND_WINDOW = timedelta(hours=24)
TREND_CAP_LOGIT = 1.0
BASE_SIGMA = 0.5        # logit units per sqrt(day), random-walk part
SPREAD_PAD = 0.3        # extra logit spread for unmodelled refresh risk
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.0
# Fallback centre (percent) when a trophy has no usable rate at all.
FALLBACK_RATE = 1.0

PCT_MIN = 0.01
PCT_MAX = 99.99


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp (with 'Z' or an offset) to datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(value):
    return min(PCT_MAX, max(PCT_MIN, value))


def logit_of(pct):
    """logit of a percent in (0, 100)."""
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of(logit_value):
    """Inverse logit, as a percent."""
    return 100.0 / (1.0 + math.exp(-logit_value))


def round4(value):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    quantised = Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )
    number = float(quantised)
    return int(number) if number.is_integer() else number


def median_logit(rates):
    """Median of a list of rates, expressed on the logit scale."""
    return logit_of(median(rates))


def forecast_trophy(series, players_series, checkpoint, horizon_days):
    """Forecast one trophy.

    series: [(datetime, rate), ...] oldest first, null rates already dropped.
    players_series: [players_tracked, ...] one per history read.
    Returns [q05, q25, q50, q75, q95] in percent.
    """
    if not series:
        centre_logit = logit_of(FALLBACK_RATE)
        spread = FLAT_SPREAD
        return quantiles_from(centre_logit, spread)

    rates = [rate for _, rate in series]
    last_rate = rates[-1]

    page_stale = len(set(players_series)) <= 1
    trophy_flat = len(set(rates)) <= 1

    if page_stale and trophy_flat:
        # The page never refreshed during the whole history: anticipate a
        # refresh before the target, which dilutes every rate.
        centre_logit = logit_of(last_rate * DILUTION_FACTOR)
        spread = FLAT_SPREAD
        return quantiles_from(centre_logit, spread)

    # Live page: robust centre from the last 6 hours of reads.
    recent_cutoff = checkpoint - RECENT_WINDOW
    recent = [rate for when, rate in series if when >= recent_cutoff]
    centre_rate = median(recent) if recent else last_rate
    centre_logit = logit_of(centre_rate)

    # Damped trend from the last 24 hours: first-half vs second-half median.
    trend_cutoff = checkpoint - TREND_WINDOW
    window = [(when, rate) for when, rate in series if when >= trend_cutoff]
    if len(window) >= 4:
        half = len(window) // 2
        first = [rate for _, rate in window[:half]]
        second = [rate for _, rate in window[half:]]
        drift = median_logit(second) - median_logit(first)
        step = drift * horizon_days
        centre_logit += max(-TREND_CAP_LOGIT, min(TREND_CAP_LOGIT, step))

    # Spread: random-walk floor vs observed volatility, both grown with the
    # horizon, plus a pad for refresh risk; floored and capped.
    if len(window) >= 2:
        volatility = pstdev([logit_of(rate) for _, rate in window])
    else:
        volatility = 0.0
    grown = max(
        BASE_SIGMA * math.sqrt(horizon_days),
        volatility * math.sqrt(horizon_days),
    )
    spread = min(SPREAD_CAP, max(SPREAD_FLOOR, grown + SPREAD_PAD))

    return quantiles_from(centre_logit, spread)


def quantiles_from(centre_logit, spread):
    """Normal quantiles on the logit scale, as clamped, rounded percents."""
    values = []
    for z in Z_SCORES:
        value = round4(clamp_pct(pct_of(centre_logit + z * spread)))
        if values and value < values[-1]:
            value = values[-1]
        values.append(value)
    return values


def forecast_case(case, name):
    if len(case["levels"]) != len(LEVELS) or any(
        abs(a - b) > 1e-9 for a, b in zip(case["levels"], LEVELS)
    ):
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    checkpoint = parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    players_series = [read["players_tracked"] for read in case["history"]]

    rows = []
    for index, trophy in enumerate(case["trophies"]):
        series = []
        for read in case["history"]:
            rate = read["rates"][index]
            if rate is not None:
                series.append((parse_time(read["at"]), rate))
        rows.append(
            {
                "id": trophy["id"],
                "q": forecast_trophy(series, players_series, checkpoint, horizon_days),
            }
        )
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
