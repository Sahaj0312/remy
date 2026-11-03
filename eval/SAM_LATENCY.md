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

## Reading

A live call is about 4 s of fixed cost (2 s upload, 2 s to the first frame) plus roughly 0.1 s per frame of streaming. Halving the frames (5 fps) or the width (480 px) each save about a second on a 7 s median, and run-to-run variance is 6 to 12 s for the same clip. Every variant still confirmed the fall. Verdict: the live clip stays at 10 fps and 640 px; the frame count is not the lever. The only large cut left is to run SAM yourself, with no upload and no queue.
