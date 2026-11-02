# Score v2 out-of-sample check

The thresholds in config were tuned by hand on all 100 clips, so the headline 32/40 falls and 1/60 false alarms is in-sample. Here the clips are split in two halves, stratified by type; the grid (speed 0.3 to 0.6, drop 0.1 to 0.25, ar_wide_max 0.7 to 1.0) picks thresholds on one half by `caught - 2 x false alarms`, the other half is scored with them, then the halves swap. Every clip is therefore scored once with thresholds it did not influence. Repeated over 5 random splits.

| split | half A picked (speed, drop, ar) → scores B | half B picked → scores A | pooled out-of-sample |
|---|---|---|---|
| 1 | (0.4, 0.15, 0.9) → 18/20 falls, 1/31 FA | (0.45, 0.15, 0.8) → 15/20 falls, 1/29 FA | **33/40 falls, 2/60 FA** |
| 2 | (0.45, 0.15, 0.9) → 15/20 falls, 1/30 FA | (0.35, 0.15, 0.8) → 17/20 falls, 1/30 FA | **32/40 falls, 2/60 FA** |
| 3 | (0.4, 0.15, 0.9) → 17/21 falls, 1/31 FA | (0.4, 0.25, 0.8) → 14/19 falls, 1/29 FA | **31/40 falls, 2/60 FA** |
| 4 | (0.4, 0.15, 1.0) → 17/19 falls, 2/31 FA | (0.35, 0.25, 0.8) → 15/21 falls, 1/29 FA | **32/40 falls, 3/60 FA** |
| 5 | (0.4, 0.15, 1.0) → 17/20 falls, 3/30 FA | (0.3, 0.25, 0.8) → 16/20 falls, 0/30 FA | **33/40 falls, 3/60 FA** |

**Out-of-sample over 5 splits: 80% of falls caught (range 78 to 82%), 2.4 false alarms per 60 clips (range 2 to 3). In-sample: 80% and 1.**

The hand-picked point (0.45, 0.15, 0.8) was chosen by the search in 1 of 10 half-fits; the others are its neighbours. The misses are the same slow falls every time: no threshold in the grid separates a fall from sitting or onto a bed from lying down on purpose.
