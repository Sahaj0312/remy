# Score v2 threshold sensitivity

Re-scored the committed SAM 3.1 tracks (100 clips: 40 falls, 60 non-falls). Each cell is falls caught / false alarms. The chosen operating point is in bold. The thresholds were tuned on this same set, so these numbers are in-sample; the table shows how flat the neighbourhood is, not how a new room will do.

## drop_speed_rel_min (rows) x drop_rel_min (columns), ar_wide_max = 0.8

| speed \ drop | 0.1 | 0.125 | 0.15 | 0.175 | 0.2 | 0.25 |
|---|---|---|---|---|---|---|
| 0.3 | 37/7 | 37/7 | 36/5 | 36/3 | 35/3 | 33/1 |
| 0.35 | 35/3 | 35/3 | 35/2 | 34/2 | 34/2 | 32/1 |
| 0.4 | 34/2 | 34/2 | 34/2 | 34/2 | 34/2 | 31/1 |
| 0.45 | 32/1 | 32/1 | **32/1** | 32/1 | 31/1 | 29/1 |
| 0.5 | 32/1 | 32/1 | 32/1 | 32/1 | 32/1 | 26/1 |
| 0.55 | 26/1 | 26/1 | 26/1 | 26/1 | 26/1 | 26/0 |
| 0.6 | 25/0 | 25/0 | 25/0 | 25/0 | 25/0 | 22/0 |

## ar_wide_max alone (speed 0.45, drop 0.15)

| ar_wide_max | 0.6 | 0.7 | 0.8 | 0.9 | 1 | 1.2 |
|---|---|---|---|---|---|---|
| falls / FA | 22/1 | 30/1 | **32/1** | 33/1 | 34/2 | 34/2 |

Read across a row: a looser drop threshold adds few falls and few false alarms, because the misses are slow falls that no drop threshold separates from lying down. Read down a column: the speed threshold is the knob that trades false alarms for the slow falls.
