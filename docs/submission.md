# Submission form text

Six fields on the Cerebral Valley form. Deadline Wed Nov 4, 9:00 PM ET (6:00 PM PT). Submit a draft early and edit
until the freeze. No pushes to `main` after the deadline.

## 1. Public GitHub URL

https://github.com/Sahaj0312/remy

## 2. Demo video URL

(YouTube unlisted, 2:50 to 2:55, English voice-over with burned-in captions)

## 3. Track

Indie Developer

## 4. Problem statement

Build for Your World

## 5. Use of Meta products

> remy is a fall detector for my grandmother, built from scratch during the hackathon (first commit Oct 29). It runs
> on a Tapo camera and a laptop and uses three Meta models at runtime, each load-bearing:
>
> 1. **SAM 3.1 (Meta Model API)**: video tracking of "person" in the 5-second clip around a candidate drop. The fall
>    decision is made on the SAM track (drop in body lengths, ends horizontal, no get-up). SAM 3.1 image segmentation
>    also builds the room zones (floor, rug, sofa, bed) from one frame at setup.
> 2. **Muse Spark 1.3 (Meta Model API)**: sees an inline video burst around the drop plus the tracker numbers and
>    returns urgency and the one-line note the family reads; and classifies the spoken reply to "Are you okay?"
>    as okay / needs help / unclear / no answer, with the question as context, so any wording or language counts.
> 3. **Muse Voice Transcribe (Meta Model API)**: speech to text of the 15 seconds after the prompt, from the camera
>    microphone.
>
> Muse Spark Open (open weights, via Fireworks) was evaluated as a drop-in alternative for the triage stage, clip for
> clip against Spark 1.3 (`eval/TRIAGE_COMPARE.md`): within one step of urgency on 29 of 33 clips, 24 of 24 on reply
> transcripts. It is kept as an optional backend, not the default.
>
> Deterministic stages sit between the models: a local YOLO pose trigger decides when to call SAM, a rule on the SAM
> track decides whether it was a fall, help words short-circuit the classifier, and a floor-zone rule raises an event
> with no visible drop. Evaluation: 100 public clips (32/40 falls, 1/60 false alarms in-sample; 80% out-of-sample),
> a threshold sensitivity sweep, a 67-test suite, and four staged scenarios on the real camera on the last day
> (sofa quiet; a clear okay at 27 s; a silent fall urgent at 57 s; a shout urgent at 20 s with the audio).
>
> **Bonus categories:** Best Agentic Use-Case (the system detects, confirms, asks, listens, decides, escalates and
> re-checks on its own, end to end, with no person in the loop); Most Viral Project on X; Community Vote.
>
> **Keyless verification for judges:** `remy replay examples/event_gmdcsa_S1_Fall_12` replays a recorded event and
> re-scores its stored SAM track; `python eval/score_tracks.py --tracker sam --scorer v2` reproduces the eval from
> the committed tracks.

## 6. X post URL

Post before the deadline, paste the URL. Draft:

> My grandmother lives alone most of the day. So during the @Meta AI hackathon I built remy: a $30 camera notices a
> fall, SAM 3.1 confirms it, the camera asks "Are you okay?", Muse Spark listens, and the family gets a
> WhatsApp with what she said. 30 s from fall to message. Open source.
>
> [video] github.com/Sahaj0312/remy

Second post in the thread, for the technical crowd:

> Under the hood: YOLO pose at 5 fps on a laptop → fast drop cuts a 5 s clip → SAM 3.1 tracks the person → a rule
> in body lengths confirms the fall → Muse Spark writes the family note → Muse Voice Transcribe hears the reply →
> Muse Spark judges it. 32/40 falls, 1/60 false alarms on 100 public clips, 80% out of sample,
> ~$0.02 per event.

## Before the freeze

- [ ] Video uploaded, link in README and form, plays without login.
- [ ] `docs/demo.gif` committed (the README placeholder).
- [ ] X post up, URL in the form.
- [ ] $150 Meta credits claimed (each participant, by Nov 4 9 PM ET).
- [ ] Fresh clone: `pip install -e . && pytest && remy replay examples/event_gmdcsa_S1_Fall_12` passes.
- [ ] No secrets in the tree: `git grep -E "LLM_|EAA|fw_" HEAD` is empty.
- [ ] Last push before 6:00 PM PT Wednesday.
