# GTA VI Twitch CCU forecaster — variant C

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (hours 2–24, UTC), as five quantiles an
hour, from the first hour's viewer count alone.

## The model, in one paragraph

A log-scale quantile forecaster anchored on the hour-1 print. Big
launches follow a two-act day: the launch surge floods in, thins out
through the small hours, then climbs again as the Americas wake and
unlock and Europe comes home for the evening, peaking in prime time
before easing towards midnight. The program learns that shape at run
time from the past launches the case itself carries: it keeps only
same-day launches (dropping any curve whose hour 1 sat before its real
unlock), weights them by how close their hour-1 audience is to this
case's, takes a per-hour weighted median and spread, and fits two
straight lines in log space joined at one transition hour — the point
where act one ends and act two begins is fitted, never assumed. The
outer quantiles use the comparables' own disagreement that hour, so
the spread widens naturally as the day goes on.

## Running it

```sh
python3 -P -s main.py
```

Reads `$MAKEFUN_INPUT_DIR` (or `/input`) and writes one
`/output/<id>.out` per `/input/<id>.in`. Python 3.13 standard library
only, no network, deterministic, well under the 8-second cap per case.

## Local validation

On 26 synthetic leave-one-out cases built from the comparables
(each launch held out, forecast from its hour-1 print alone) the
forecaster scores a mean pinball loss of **0.216**, against **0.444**
for a flat forecast that repeats the hour-1 print — winning 25 of 26
cases. On the four practice cases it produces valid 23-row outputs,
byte-identical across repeated runs.
