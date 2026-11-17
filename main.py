#!/usr/bin/env python3
"""Forecast GTA VI launch-day Twitch viewers, hour by hour (variant C).

Log-scale quantile forecaster anchored on the first hour's print.

Idea, in plain words: big game launches on Twitch follow a familiar
two-act shape over their release day.  Act one is the launch surge
itself: the audience that was already waiting floods in during the
first hours, then thins out as the small hours pass in Europe and the
Americas are still asleep.  Act two is the long day that follows: the
Americas wake up and unlock, Europe comes home in the evening, the
audience climbs again to a prime-time peak and only then eases off
towards midnight.  Where act one ends and act two begins is not
assumed here -- it is fitted, every run, from the past launches the
case itself carries.

How it works, per case:
  1. Anchor on the hour-1 print p1 (the only thing known for sure).
  2. For every comparable launch, build its release-day curve as
     log-ratios ln(viewers_h / viewers_1) for hours 2..24.
  3. Throw out curves that cannot be a same-day launch: a launch
     whose hour 1 sits before its real unlock explodes to tens or
     hundreds of times its hour-1 count later in the day, which no
     rolling-midnight launch does.  Curves whose largest ratio tops
     8x are dropped, so a pre-launch hour 1 cannot poison the fit.
  4. Weight the surviving curves by how close their hour-1 size is
     to this case's p1 (Gaussian kernel on log size, bandwidth 1.5):
     a 900k-viewer launch teaches more about a huge launch than a
     400-viewer one does.  Per hour, take the weighted median of the
     log-ratios and a weighted interquartile spread.
  5. Fit the two-act median: two straight lines in log space, joined
     at one transition hour T, chosen by least squares over T = 4..20
     against the per-hour medians.  Act one is the surge fading,
     act two the long climb to the evening peak and the slow ease
     after it; the slopes come from the data, not from a guess.
  6. Quantiles: q_tau = p1 * exp(median_h + z_tau * sigma_h) with the
     usual normal z-scores.  The spread sigma_h comes from the
     comparables' own disagreement that hour, so it widens naturally
     as the day goes on and the launches diverge.

Only the Python standard library is used.  No network, no random
numbers, no clock reads: the same input always yields the same bytes.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
HOURS = list(range(2, 25))

# A same-day launch never grows more than this many times its hour-1
# count; anything above it had its hour 1 before its real unlock.
MAX_RATIO = 8.0
# Gaussian kernel bandwidth (in log size) for weighting comparables
# by how close their hour-1 audience is to this case's.
SIZE_BANDWIDTH = 1.5
# Lowest spread allowed for any hour (log units); keeps the outer
# quantiles honest when the comparables happen to agree.
SIGMA_FLOOR = 0.30
# Fallback spread (house-style) when a case carries no usable curve.
FALLBACK_SIGMA0 = 0.15


def parse_time(text):
    """Parse an ISO-8601 timestamp, with or without a Z suffix."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def hour_index(day_start, stamp):
    """Whole-hour offset of `stamp` from `day_start`, 1-based."""
    delta = parse_time(stamp) - day_start
    return int(delta.total_seconds() // 3600) + 1


def release_day_curve(comp):
    """Hour -> viewers for one comparable's release day (hours 1..24).

    Returns None when the curve has no usable hour-1 print.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            h = hour_index(day_start, point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= h <= 24:
            hours[h] = float(viewers)
    if 1 not in hours:
        return None
    return hours


def usable_curves(case, anchor):
    """Log-ratio curves of the comparables, filtered and weighted.

    A curve is usable when it covers hour 1 plus at least 12 of the
    23 forecast hours, and its largest hourly ratio stays below
    MAX_RATIO (a same-day launch).  Each curve carries a weight from
    the size kernel against this case's anchor print.
    """
    curves = []
    for comp in case.get("comparables") or []:
        hours = release_day_curve(comp)
        if hours is None:
            continue
        p1 = hours[1]
        ratios = {}
        worst = 0.0
        for h in HOURS:
            if h in hours:
                ratio = hours[h] / p1
                ratios[h] = math.log(ratio)
                if ratio > worst:
                    worst = ratio
        if len(ratios) < 12 or worst > MAX_RATIO:
            continue
        size_gap = math.log(p1 / anchor) if anchor > 0 else 0.0
        weight = math.exp(-((size_gap / SIZE_BANDWIDTH) ** 2))
        curves.append((weight, ratios))
    return curves


def weighted_quantile(pairs, frac):
    """Weighted quantile of (value, weight) pairs, sorted by value."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= frac * total:
            return value
    return ordered[-1][0]


def hourly_medians(curves):
    """Per-hour weighted median log-ratio and spread (log units).

    The median is the point forecast's centre; the spread is the
    comparables' own interquartile range turned into a log sigma,
    floored so a lucky agreement never claims false certainty.
    Missing hours borrow the nearest available hour's figures.
    """
    medians = {}
    sigmas = {}
    for h in HOURS:
        pairs = [(ratios[h], weight) for weight, ratios in curves
                 if h in ratios]
        if len(pairs) < 3:
            continue
        medians[h] = weighted_quantile(pairs, 0.5)
        q25 = weighted_quantile(pairs, 0.25)
        q75 = weighted_quantile(pairs, 0.75)
        sigmas[h] = max((q75 - q25) / 1.349, SIGMA_FLOOR)
    for h in HOURS:
        if h not in medians:
            near = min(medians, key=lambda k: abs(k - h)) if medians else None
            medians[h] = medians[near] if near is not None else 0.0
            sigmas[h] = sigmas[near] if near is not None else SIGMA_FLOOR
    return medians, sigmas


def _solve_3x3(matrix, vector):
    """Solve a 3x3 linear system by Gauss elimination (no pivoting
    tricks needed: the normal equations here are well conditioned)."""
    rows = [matrix[i][:] + [vector[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if abs(rows[col][col]) < 1e-12:
            continue
        for r in range(3):
            if r == col:
                continue
            factor = rows[r][col] / rows[col][col]
            for c in range(col, 4):
                rows[r][c] -= factor * rows[col][c]
    out = []
    for i in range(3):
        if abs(rows[i][i]) < 1e-12:
            out.append(0.0)
        else:
            out.append(rows[i][3] / rows[i][i])
    return out


def two_phase_median(medians):
    """Fit the two-act median: two log-space lines joined at hour T.

    Act one covers hours 2..T, act two hours T+1..24, both lines
    passing through the same point at T so the forecast never jumps.
    T is chosen by least squares over 4..20 against the per-hour
    medians; the level and both slopes are fitted, never assumed.
    """
    best = None
    for trans in range(4, 21):
        rows = []
        target = []
        for h in HOURS:
            offset = h - trans
            rows.append([1.0,
                         float(offset) if h <= trans else 0.0,
                         float(offset) if h > trans else 0.0])
            target.append(medians[h])
        normal = [[sum(rows[i][j] * rows[i][k] for i in range(len(rows)))
                   for k in range(3)] for j in range(3)]
        rhs = [sum(rows[i][j] * target[i] for i in range(len(rows)))
               for j in range(3)]
        level, slope1, slope2 = _solve_3x3(normal, rhs)
        error = 0.0
        for h, row in zip(HOURS, rows):
            fitted = level + row[1] * slope1 + row[2] * slope2
            error += (medians[h] - fitted) ** 2
        if best is None or error < best[0]:
            best = (error, trans, level, slope1, slope2)
    _, trans, level, slope1, slope2 = best
    fitted = {}
    for h in HOURS:
        offset = h - trans
        fitted[h] = level + (slope1 if h <= trans else slope2) * offset
    return fitted


def forecast_case(case):
    """Return the 23 forecast rows for one case."""
    anchor = 1.0
    known = case.get("known") or []
    for row in known:
        if row.get("hour") == 1 and row.get("viewers"):
            anchor = max(1.0, float(row["viewers"]))
            break
    else:
        if known and known[0].get("viewers"):
            anchor = max(1.0, float(known[0]["viewers"]))

    curves = usable_curves(case, anchor)
    if len(curves) >= 5:
        medians, sigmas = hourly_medians(curves)
        centre = two_phase_median(medians)
        spread = sigmas
    else:
        # No usable past launches in this case: stay flat, widening
        # the spread over the day like the house baseline does.
        centre = {h: 0.0 for h in HOURS}
        spread = {h: FALLBACK_SIGMA0 * math.sqrt(h - 1) for h in HOURS}

    rows = []
    for h in HOURS:
        quantiles = []
        prev = 0.0
        for z in Z:
            value = anchor * math.exp(centre[h] + z * spread[h])
            if not math.isfinite(value) or value <= 0:
                value = anchor
            value = min(value, 1e9)
            value = round(value, 2)
            if value < prev:
                value = prev
            if value <= 0:
                value = prev if prev > 0 else 1.0
            quantiles.append(value)
            prev = value
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        name = os.path.basename(path)
        case_id = name[:-3] if name.endswith(".in") else name
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(result, separators=(",", ":")))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
