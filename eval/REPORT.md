# Fall-detection feasibility test, round 2

100 clips: 40 falls, 60 non-falls (42 UR Fall, 58 GMDCSA24), 10 fps, max width 640. Lighting: {'day': 84, 'night': 9, 'lights-off': 7}.

## Before / after tuning on the first-round 40 clips

| tracks | score | falls caught | false alarms | flagged with hold unknown |
|---|---|---|---|---|
| 5 fps, v1 (round 1) | | 18/20 | 0/20 | 0 |
| 5 fps, v2 | | 20/20 | 1/20 | 14 |
| 10 fps, v1 | | 14/20 | 0/20 | 0 |
| 10 fps, v2 (round 2) | | 19/20 | 1/20 | 13 |

v1 = absolute drop + tall-before + fixed 3 s low hold. v2 = drop in body lengths, ends-wide, no get-up within no_getup_s (hold 'unknown' when the clip ends first, still flagged).

## Per-type results on the full set (SAM 3.1 tracks, score v2)

| type | clips | flagged by score | hold unknown | note |
|---|---|---|---|---|
| fall:bed | 7 | 5/7 | 4 | caught |
| fall:chair | 11 | 11/11 | 10 | caught |
| fall:floor-sit | 3 | 0/3 | 0 | caught |
| fall:onto-bed | 2 | 0/2 | 0 | caught |
| fall:slow | 1 | 1/1 | 0 | caught |
| fall:standing | 16 | 15/16 | 9 | caught |
| adl:bend-pick | 15 | 0/15 | 0 | false alarms |
| adl:exercise-floor | 4 | 1/4 | 1 | false alarms |
| adl:exercise-stand | 4 | 0/4 | 0 | false alarms |
| adl:get-up | 5 | 0/5 | 0 | false alarms |
| adl:lie-bed | 20 | 0/20 | 0 | false alarms |
| adl:lie-floor | 6 | 0/6 | 0 | false alarms |
| adl:sit | 5 | 0/5 | 0 | false alarms |
| adl:sit-walk | 1 | 0/1 | 0 | false alarms |

**Overall: 32/40 falls caught, 1/60 false alarms.** Hold was 'unknown' (clip ended before N s) on 24 of the 33 flagged clips.

Missed falls: gmdcsa_S3_Fall_18 (fall:bed), gmdcsa_S2_Fall_03 (fall:onto-bed), gmdcsa_S3_Fall_09 (fall:floor-sit), gmdcsa_S2_Fall_08 (fall:floor-sit), gmdcsa_S3_Fall_13 (fall:floor-sit), gmdcsa_S4_Fall_06 (fall:bed), urfall_fall-23 (fall:standing), gmdcsa_S3_Fall_17 (fall:onto-bed)

False alarms: gmdcsa_S4_ADL_05 (adl:exercise-floor)

## YOLO always-on filter: would it have called SAM?

YOLO pose + trigger rule (drop only, loose thresholds) fires on **38/40 falls** and 41/60 non-falls (each firing = one SAM call).

| type | trigger fired |
|---|---|
| fall:bed | 5/7 |
| fall:chair | 11/11 |
| fall:floor-sit | 3/3 |
| fall:onto-bed | 2/2 |
| fall:slow | 1/1 |
| fall:standing | 16/16 |
| adl:bend-pick | 11/15 |
| adl:exercise-floor | 4/4 |
| adl:exercise-stand | 3/4 |
| adl:get-up | 0/5 |
| adl:lie-bed | 13/20 |
| adl:lie-floor | 5/6 |
| adl:sit | 4/5 |
| adl:sit-walk | 1/1 |

Falls the YOLO trigger missed (never reach SAM): gmdcsa_S4_Fall_06 (fall:bed, coverage 0.89), gmdcsa_S2_Fall_11 (fall:bed, coverage 0.56)


For reference, YOLO tracks + the full v2 score: 30/40 falls, 5/60 false alarms.

## Muse Spark triage of flagged clips (urgency + family summary)

33 flagged clips triaged. Urgency: {'medium': 8, 'high': 25}. is_lying_still: {False: 10, True: 23}.

| type | high | medium | low |
|---|---|---|---|
| fall:bed | 4 | 1 | 0 |
| fall:chair | 10 | 1 | 0 |
| fall:slow | 0 | 1 | 0 |
| fall:standing | 11 | 4 | 0 |
| adl:exercise-floor | 0 | 1 | 0 |

Sample summaries:

- urfall_fall-01 (fall:standing): **medium** — "Went down quickly in the middle of the office near the desks at 3.9s into the clip and is still on the floor but appears to be moving."
- urfall_fall-17 (fall:standing): **high** — "Fell forward quickly in the middle of the room about 3 seconds into the clip and was still down on the floor when the clip ended."
- gmdcsa_S4_Fall_02 (fall:bed): **high** — "Fell off the bed onto the floor at 2.2s into the clip and is still down on the floor."
- gmdcsa_S4_Fall_15 (fall:bed): **high** — "Fell forward off the bed onto the floor at 2.6 seconds in and has been lying still on the floor for about 2 seconds."
- urfall_fall-16 (fall:chair): **high** — "Fell from the chair to the floor near the desk about 1.6s into the clip and was still down when the clip ended."
- urfall_fall-20 (fall:chair): **medium** — "Slipped off the chair onto the floor about 1.5 seconds into the clip and is still down but moving around."

## Per-clip results

| clip | type | lighting | true | SAM v2 (D/S/hold) | YOLO trigger | Spark urgency | lying still |
|---|---|---|---|---|---|---|---|
| urfall_fall-01 | fall:standing | day | fall | YES DS/unknown | YES | medium | no |
| urfall_fall-02 | fall:chair | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-04 | fall:chair | day | fall | YES DS/unknown | YES | high | no |
| urfall_fall-05 | fall:standing | day | fall | YES DS/unknown | YES | medium | no |
| urfall_fall-07 | fall:standing | day | fall | YES DS/unknown | YES | medium | no |
| urfall_fall-08 | fall:chair | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-10 | fall:chair | day | fall | YES DS/unknown | YES | high | no |
| urfall_fall-11 | fall:standing | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-13 | fall:standing | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-14 | fall:chair | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-16 | fall:chair | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-17 | fall:standing | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-19 | fall:standing | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-20 | fall:chair | day | fall | YES DS/unknown | YES | medium | no |
| urfall_fall-22 | fall:chair | day | fall | YES DS/unknown | YES | high | YES |
| urfall_fall-23 | fall:standing | day | fall | no D./unknown | YES | - | - |
| urfall_fall-25 | fall:standing | day | fall | YES DS/unknown | YES | medium | no |
| urfall_fall-28 | fall:chair | day | fall | YES DS/unknown | YES | high | no |
| urfall_adl-01 | adl:sit-walk | day | adl | no ../- | YES | - | - |
| urfall_adl-07 | adl:bend-pick | day | adl | no ../- | no | - | - |
| urfall_adl-08 | adl:bend-pick | day | adl | no ../- | no | - | - |
| urfall_adl-10 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| urfall_adl-11 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| urfall_adl-18 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| urfall_adl-19 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| urfall_adl-20 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| urfall_adl-21 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| urfall_adl-22 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| urfall_adl-23 | adl:lie-bed | day | adl | no D./failed | YES | - | - |
| urfall_adl-25 | adl:bend-pick | day | adl | no ../- | no | - | - |
| urfall_adl-27 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| urfall_adl-30 | adl:lie-floor | day | adl | no ../- | YES | - | - |
| urfall_adl-31 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| urfall_adl-32 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| urfall_adl-33 | adl:lie-bed | day | adl | no DS/failed | YES | - | - |
| urfall_adl-34 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| urfall_adl-35 | adl:lie-floor | day | adl | no ../- | YES | - | - |
| urfall_adl-36 | adl:lie-bed | day | adl | no ../- | no | - | - |
| urfall_adl-37 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| urfall_adl-38 | adl:lie-floor | day | adl | no D./failed | YES | - | - |
| urfall_adl-39 | adl:lie-floor | day | adl | no D./failed | YES | - | - |
| urfall_adl-40 | adl:lie-floor | day | adl | no ../- | YES | - | - |
| gmdcsa_S1_Fall_05 | fall:standing | day | fall | YES DS/ok | YES | high | YES |
| gmdcsa_S1_Fall_09 | fall:standing | day | fall | YES DS/unknown | YES | high | YES |
| gmdcsa_S1_Fall_12 | fall:standing | day | fall | YES DS/ok | YES | high | YES |
| gmdcsa_S1_Fall_15 | fall:chair | day | fall | YES DS/ok | YES | high | YES |
| gmdcsa_S2_Fall_02 | fall:standing | day | fall | YES DS/ok | YES | high | YES |
| gmdcsa_S2_Fall_04 | fall:standing | day | fall | YES DS/ok | YES | high | YES |
| gmdcsa_S2_Fall_08 | fall:floor-sit | day | fall | no ../- | YES | - | - |
| gmdcsa_S2_Fall_11 | fall:bed | night | fall | YES DS/ok | no | high | YES |
| gmdcsa_S3_Fall_05 | fall:slow | day | fall | YES DS/ok | YES | medium | no |
| gmdcsa_S4_Fall_10 | fall:chair | day | fall | YES DS/unknown | YES | high | YES |
| gmdcsa_S4_Fall_02 | fall:bed | day | fall | YES DS/unknown | YES | high | YES |
| gmdcsa_S4_Fall_06 | fall:bed | lights-off | fall | no ../- | no | - | - |
| gmdcsa_S4_Fall_12 | fall:bed | day | fall | YES DS/unknown | YES | high | YES |
| gmdcsa_S4_Fall_15 | fall:bed | day | fall | YES DS/unknown | YES | high | YES |
| gmdcsa_S3_Fall_12 | fall:bed | day | fall | YES DS/unknown | YES | medium | YES |
| gmdcsa_S3_Fall_18 | fall:bed | night | fall | no D./ok | YES | - | - |
| gmdcsa_S3_Fall_09 | fall:floor-sit | day | fall | no ../- | YES | - | - |
| gmdcsa_S3_Fall_13 | fall:floor-sit | day | fall | no ../- | YES | - | - |
| gmdcsa_S3_Fall_17 | fall:onto-bed | night | fall | no ../- | YES | - | - |
| gmdcsa_S3_Fall_16 | fall:standing | night | fall | YES DS/ok | YES | high | YES |
| gmdcsa_S2_Fall_03 | fall:onto-bed | day | fall | no ../- | YES | - | - |
| gmdcsa_S2_Fall_10 | fall:standing | night | fall | YES DS/ok | YES | high | YES |
| gmdcsa_S1_ADL_01 | adl:lie-bed | day | adl | no ../- | no | - | - |
| gmdcsa_S2_ADL_03 | adl:lie-bed | day | adl | no ../- | no | - | - |
| gmdcsa_S2_ADL_08 | adl:exercise-stand | day | adl | no ../- | YES | - | - |
| gmdcsa_S2_ADL_12 | adl:lie-bed | day | adl | no ../- | YES | - | - |
| gmdcsa_S2_ADL_22 | adl:exercise-floor | night | adl | no ../- | YES | - | - |
| gmdcsa_S3_ADL_08 | adl:lie-floor | day | adl | no ../- | no | - | - |
| gmdcsa_S3_ADL_09 | adl:get-up | day | adl | no ../- | no | - | - |
| gmdcsa_S4_ADL_05 | adl:exercise-floor | lights-off | adl | YES DS/unknown | YES | medium | no |
| gmdcsa_S4_ADL_10 | adl:lie-bed | lights-off | adl | no ../- | YES | - | - |
| gmdcsa_S4_ADL_15 | adl:lie-bed | day | adl | no ../- | no | - | - |
| gmdcsa_S1_ADL_02 | adl:lie-bed | day | adl | no ../- | no | - | - |
| gmdcsa_S1_ADL_03 | adl:lie-bed | day | adl | no ../- | no | - | - |
| gmdcsa_S1_ADL_04 | adl:get-up | day | adl | no ../- | no | - | - |
| gmdcsa_S1_ADL_15 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| gmdcsa_S1_ADL_16 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| gmdcsa_S2_ADL_02 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| gmdcsa_S2_ADL_04 | adl:exercise-stand | day | adl | no ../- | no | - | - |
| gmdcsa_S2_ADL_05 | adl:get-up | day | adl | no ../- | no | - | - |
| gmdcsa_S2_ADL_06 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| gmdcsa_S2_ADL_09 | adl:sit | day | adl | no ../- | YES | - | - |
| gmdcsa_S2_ADL_13 | adl:sit | day | adl | no ../- | YES | - | - |
| gmdcsa_S2_ADL_16 | adl:lie-bed | night | adl | no ../- | YES | - | - |
| gmdcsa_S2_ADL_20 | adl:sit | night | adl | no ../- | YES | - | - |
| gmdcsa_S3_ADL_03 | adl:get-up | day | adl | no ../- | no | - | - |
| gmdcsa_S3_ADL_04 | adl:bend-pick | day | adl | no ../- | YES | - | - |
| gmdcsa_S3_ADL_05 | adl:exercise-stand | day | adl | no ../- | YES | - | - |
| gmdcsa_S3_ADL_06 | adl:sit | day | adl | no D./ok | YES | - | - |
| gmdcsa_S3_ADL_07 | adl:exercise-stand | day | adl | no D./failed | YES | - | - |
| gmdcsa_S3_ADL_19 | adl:bend-pick | night | adl | no ../- | YES | - | - |
| gmdcsa_S4_ADL_06 | adl:exercise-floor | lights-off | adl | no ../- | YES | - | - |
| gmdcsa_S4_ADL_07 | adl:exercise-floor | lights-off | adl | no ../- | YES | - | - |
| gmdcsa_S4_ADL_08 | adl:get-up | lights-off | adl | no ../- | no | - | - |
| gmdcsa_S4_ADL_11 | adl:sit | lights-off | adl | no ../- | no | - | - |
| gmdcsa_S4_ADL_14 | adl:bend-pick | day | adl | no ../- | no | - | - |
| gmdcsa_S4_ADL_16 | adl:lie-bed | day | adl | no ../- | no | - | - |
| gmdcsa_S4_ADL_17 | adl:bend-pick | day | adl | no ../- | YES | - | - |

## SAM 3.1 latency and cost

| metric | median | mean | max |
|---|---|---|---|
| upload (s) | 5.2 | 6.1 | 19.0 |
| time to first frame (s) | 2.3 | 3.1 | 9.3 |
| total per clip (s) | 25.6 | 25.2 | 55.2 |
| frames per clip | 67 | 71 | 137 |

SAM: 100 clips, 7133 frames sent, 7549 billed, **$1.510**.

## Muse Spark latency and cost

33 calls (Counter({'video': 33})), latency median 5.5s / mean 6.0s / max 9.4s. Tokens: 49349 in, 32104 out. Cost **$0.1981** ($0.0060 per call).

**Round-2 API cost: $1.708. Round 1 was $0.519. Running total $2.23.**

YOLO pose: mean 36 ms/frame on this Mac's CPU.
