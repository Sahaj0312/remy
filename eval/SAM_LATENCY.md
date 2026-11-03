# SAM 3.1 latency versus clip length

Question: is the 8 to 10 s from "clip cut" to "fall confirmed" a function of how much video we send, or a fixed cost of the call? If it scales, a shorter or sparser clip buys time; if not, only self-hosting does.

## 1. The 100 eval runs (batch, three uploads in parallel)

Frames per clip 18 to 137, median 67. Linear fit of each timing against frame count:

| timing | median | p10 | p90 | fit | correlation |
|---|---|---|---|---|---|
| upload | 5.2 s | 2.6 s | 10.4 s | 4.4 s + 2.4 s per 100 frames | 0.21 |
| time to first frame | 2.3 s | 1.7 s | 5.6 s | 2.7 s + 0.5 s per 100 frames | 0.09 |
| stream after upload | 18.9 s | 8.4 s | 28.8 s | 5.6 s + 18.9 s per 100 frames | 0.68 |
| total | 25.6 s | 13.7 s | 35.4 s | 10.0 s + 21.4 s per 100 frames | 0.70 |

| frames | clips | upload | first frame | stream | total |
|---|---|---|---|---|---|
| 0 to 40 | 18 | 3.4 s | 2.1 s | 8.4 s | 14.2 s |
| 40 to 60 | 19 | 4.6 s | 2.4 s | 15.2 s | 21.0 s |
| 60 to 90 | 33 | 5.8 s | 2.2 s | 18.0 s | 26.4 s |
| 90 to 200 | 30 | 6.2 s | 2.6 s | 25.0 s | 31.1 s |

The stream time grows with frames; the first frame does not. The batch ran three clips at once, so the absolute numbers are higher than a single live call.

## 2. One fall clip, five variants, 3 sequential calls each

The live simulation's clip (`examples/event_gmdcsa_S1_Fall_12/clip.mp4`, 35 frames at 10 fps, 640 px), re-encoded
at 5 fps, shortened by half a second, and scaled to 480 px. Calls one at a time, variants interleaved.

| variant | frames | total, median (min to max) | upload | first frame | stream | fall confirmed |
|---|---|---|---|---|---|---|
| 35 frames, 10 fps, 640 px (the live clip) | 35 | 7.1 s (6.8 to 12.3) | 1.8 s | 1.5 s | 5.3 s | 3/3 |
| 18 frames, 5 fps, 640 px | 18 | 6.0 s (5.8 to 9.8) | 2.5 s | 1.8 s | 3.8 s | 3/3 |
| 30 frames, 10 fps, 640 px | 30 | 8.4 s (7.3 to 11.2) | 2.1 s | 2.7 s | 6.3 s | 3/3 |
| 15 frames, 5 fps, 640 px | 15 | 6.8 s (5.8 to 8.8) | 2.3 s | 2.2 s | 4.5 s | 3/3 |
| 35 frames, 10 fps, 480 px | 35 | 6.3 s (6.1 to 7.0) | 2.1 s | 1.7 s | 4.2 s | 3/3 |

The fourteen live events so far (35 to 64 frames) took 5.4 to 9.5 s each, with no visible dependence on frame
count inside that range.

## Reading

A live call is about 4 s of fixed cost (2 s upload, 2 s to the first frame) plus roughly 0.1 s per frame of streaming. Halving the frames (5 fps) or the width (480 px) each save about a second on a 7 s median, and run-to-run variance is 6 to 12 s for the same clip. Every variant still confirmed the fall, with the same drop measurement to two decimals at 10 fps and a slightly smaller one at 5 fps. Verdict: the live clip stays at 10 fps and 640 px, because the thresholds, the sensitivity sweep and the held-out check were all done at that rate and the gain is inside the noise. The frame count is not the lever. The only large cut left is to run SAM yourself, with no upload and no queue.
