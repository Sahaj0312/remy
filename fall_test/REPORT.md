# Fall-detection feasibility test

40 clips: 20 falls, 20 ADL (20 UR Fall, 20 GMDCSA24). Clips downsampled to 5 fps, max width 640.

## Per-clip results

| clip | true | SAM score (D/S/H) | Spark verdict | final | YOLO score | note |
|---|---|---|---|---|---|---|
| urfall_fall-01 | fall | YES DSH | YES (0.90) | YES | YES DSH | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-04 | fall | YES DSH | YES (0.88) | YES | YES DSH | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-07 | fall | YES DSH | YES (0.88) | YES | YES DSH | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-10 | fall | YES DSH | YES (0.78) | YES | YES DSH | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-13 | fall | YES DSH | YES (0.85) | YES | YES DSH | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-16 | fall | YES DSH | YES (0.85) | YES | YES DSH | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-19 | fall | YES DSH | YES (0.85) | YES | no D.. | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-22 | fall | YES DSH | YES (0.88) | YES | YES DSH | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-25 | fall | YES DSH | YES (0.82) | YES | no DS. | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_fall-28 | fall | YES DSH | YES (0.90) | YES | no --- | fall (UR Fall Detection Dataset, cam0 RGB) |
| urfall_adl-01 | adl | no --- | not sent | no | no --- | ADL, no lying frames in dataset labels (walking/sitting type |
| urfall_adl-07 | adl | no --- | not sent | no | no --- | ADL, no lying frames; low min height/width ratio 0.62 (crouc |
| urfall_adl-08 | adl | no --- | not sent | no | no --- | ADL, no lying frames; low min height/width ratio 0.75 (crouc |
| urfall_adl-10 | adl | no D.H | no (0.95) | no | no D.H | ADL with lying frames (lying down on bed/sofa) |
| urfall_adl-20 | adl | no --- | not sent | no | no --- | ADL, no lying frames; low min height/width ratio 0.65 (crouc |
| urfall_adl-21 | adl | no --- | not sent | no | no --- | ADL with lying frames, very wide pose (lying down) |
| urfall_adl-22 | adl | no --- | not sent | no | no --- | ADL with lying frames, very wide pose (lying down) |
| urfall_adl-30 | adl | no --- | not sent | no | no D.. | ADL with many lying frames (lying down) |
| urfall_adl-35 | adl | no --- | not sent | no | no D.. | ADL with lying frames (lying down) |
| urfall_adl-38 | adl | no D.H | not sent | no | no D.H | ADL with many lying frames (lying down) |
| gmdcsa_S1_Fall_05 | fall | YES DSH | YES (0.82) | YES | YES DSH | Walking to falling. Right side fall |
| gmdcsa_S1_Fall_09 | fall | YES DSH | YES (0.80) | YES | YES DSH | Standing to falling. Forward fall, head not fully visible |
| gmdcsa_S1_Fall_12 | fall | YES DSH | YES (0.82) | YES | YES DSH | Standing to falling. Backward fall |
| gmdcsa_S1_Fall_15 | fall | YES DSH | YES (0.88) | YES | YES DSH | Sitting on chair (side view) to right side fall |
| gmdcsa_S2_Fall_02 | fall | YES DSH | no (0.82) | no | YES DSH | Right side fall on the ground after walking |
| gmdcsa_S2_Fall_04 | fall | YES DSH | no (0.85) | no | YES DSH | Looking out window then backward fall |
| gmdcsa_S2_Fall_08 | fall | no --- | not sent | no | no --- | Reading while sitting on the ground, then falling backward |
| gmdcsa_S2_Fall_11 | fall | no D.H | not sent | no | no --- | Falling from bed to ground (night) |
| gmdcsa_S3_Fall_05 | fall | YES DSH | no (0.80) | no | YES DSH | Falling slowly from standing position to the floor |
| gmdcsa_S4_Fall_10 | fall | YES DSH | YES (0.90) | YES | no --- | Reading while sitting on chair then forward fall to ground |
| gmdcsa_S1_ADL_01 | adl | no --- | not sent | no | no --- | Sitting on the bed to sleeping on right side on the bed |
| gmdcsa_S2_ADL_03 | adl | no --- | not sent | no | no --- | Walking; dusting the bed with a pillow; sleeping on the bed |
| gmdcsa_S2_ADL_08 | adl | no --- | not sent | no | YES DSH | Exercising: stretching thigh, touching the ground (pyramid p |
| gmdcsa_S2_ADL_12 | adl | no --- | not sent | no | no --- | Going to bed and sleeping belly down |
| gmdcsa_S2_ADL_22 | adl | no --- | not sent | no | YES DSH | Exercising (push-ups) and sitting (night) |
| gmdcsa_S3_ADL_08 | adl | no --- | not sent | no | no --- | Sitting to sleeping on the floor |
| gmdcsa_S3_ADL_09 | adl | no --- | not sent | no | no --- | Sleeping on the floor to sitting then standing |
| gmdcsa_S4_ADL_05 | adl | no D.H | not sent | no | no D.H | Push-ups, view 1 (lights off) |
| gmdcsa_S4_ADL_10 | adl | no --- | not sent | no | no --- | Walking then sleeping on the bed (lights off) |
| gmdcsa_S4_ADL_15 | adl | no --- | not sent | no | no --- | Sitting to sleeping on the bed |

D/S/H = drop / tall-to-wide shape / stays-low checks passed on the best candidate event.

## Overall: SAM 3.1 tracking + rule score, before and after Spark

| stage | falls caught | ADL false alarms |
|---|---|---|
| rule score only | 18/20 | 0/20 |
| + Spark verification | 15/20 | 0/20 |

## Overall: YOLO pose baseline + same rule score

| stage | falls caught | ADL false alarms |
|---|---|---|
| rule score only | 14/20 | 2/20 |

## SAM 3.1 latency and cost

| metric | median | mean | max |
|---|---|---|---|
| upload (s) | 3.5 | 4.1 | 13.2 |
| time to first frame after request (s) | 1.9 | 2.2 | 5.8 |
| total per clip incl. upload (s) | 9.9 | 10.8 | 24.9 |
| frames per clip | 32 | 36 | 68 |

SAM cost: 40 clips, 1429 frames sent, 2082 billed (50-frame minimum), **$0.416** at $0.20 per 1,000 video frames. Mean seconds of SAM processing per second of video: 1.06.

## Muse Spark latency and cost

19 calls, latency median 2.8s / mean 2.8s / max 5.1s. Tokens: 48978 in, 9660 out. Cost **$0.1023** ($0.0054 per call).

**Total API cost: $0.519**

## YOLO pose baseline speed

yolo11n-pose+iou-link on this Mac (CPU): mean 38 ms/frame (median 40, max 79), i.e. ~26 fps. No API cost.
