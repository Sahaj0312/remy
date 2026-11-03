# Demo video: shot list and script

Three minutes for the judges. Rule to keep in mind: the video must show only what was built during the
hackathon, and judges must be able to tell what that is. Everything in remy qualifies; say so once on screen.

## Before you film (10 minutes)

1. Message the test number from your phone (any text) so the 24 h free-form window is open. Then check:
   `remy test-alert --text "demo starting"`. If that fails with a window error, message the number again.
2. Tapo app: speaker volume to maximum, privacy mode off, lights on in the room, daytime.
3. The live page instead of the terminal: `remy run` serves http://127.0.0.1:8790 with the camera view (boxes,
   zones, state banner), the state card, the last message sent, the model chips lighting up, and the log. Open
   it in a browser window sized 16:9 (1600x900 is good), full screen it, and screen record that window
   (cmd+shift+5). Keep the terminal open too for the full log.
4. Phone: open the WhatsApp chat with the test number, start iOS screen recording (Control Center). Keep it
   running the whole session; it is cheaper to cut later than to miss a message arriving. Tap each video when it
   lands so the recording shows it playing with sound. Before the first scenario, send yourself one test video so
   the path is proven on the day: `remy test-alert --video examples/event_gmdcsa_S1_Fall_12/clip.mp4 --text "video test"`.
5. Record the camera's own view at full resolution for the whole session (the event clips are 640 px, 10 fps):

   ```sh
   mkdir -p tapo && ffmpeg -rtsp_transport tcp -i "$(.venv/bin/python -c 'from remy.common import tapo_rtsp; print(tapo_rtsp(1))')" \
     -t 2400 -c:v copy -c:a aac tapo/demo_$(date +%H%M).mp4
   ```

   Keep the terminal time visible somewhere; it is how you line this up with the log. This recording is the
   camera's third and last RTSP session (the monitor holds two), so do not open VLC or another viewer while
   it runs.
6. Start the monitor in a second terminal: `remy run --source tapo --label demo`. Wait for the `zones:` line.
7. A second phone or a friend filming you from the doorway gives the "room" shots (you walking in, the fall,
   the speaker asking). If you are alone, prop the phone on the desk pointing at the rug.

## Scenarios to run, in this order

Wait for the `done`/`ALERT` line and about 40 s of cooldown between scenarios. Stand up and walk out of the
frame between them.

| # | what you do | what should happen | caption for the video |
|---|---|---|---|
| A | walk in, sit on the sofa, lie down on it for 60 s, get up, walk around | nothing | "Sofa and bed are zones. Lying there is fine." |
| B | walk to the rug, fall, stay down, answer "Yeah, I'm okay" clearly | the recording ends 2 s after you stop talking; calm WhatsApp ≈20 s after the fall, a video of the fall and of you saying it | "A clear okay, twenty seconds. The family sees the fall and hears her." |
| C | fall, stay down, say nothing | two prompts, then urgent WhatsApp ≈50 s with the fall video and "did not respond when asked if she was okay" | "No answer twice. Urgent, with the fall." |
| D | fall, stay down, say "Help! Help me!" loudly toward the camera | urgent WhatsApp, reason "She asked for help.", video of the fall and of you saying it | "A help word escalates at once, and the family hears it." |
| E | sit down on the rug slowly (no drop), lie back, stay still 30 s | floor rule event after 20 s, then the check-in; answer "I'm fine" | "No visible drop. On the floor and still for 20 s: it still asks." |
| F | optional, terminal only: while it runs, `pgrep -f "ffmpeg.*stream2" \| xargs kill` | "stream lost, reconnecting... stream back after 8s" | "A dropped camera reconnects on its own." |

If D is misheard again, do it once more; the mic is the weak link and a second take is cheap.
After the session, each event is in `events/<time>_demo_*/` with `clip.mp4`, `snapshot.jpg`, `log.txt`.

## Other shots and stills

- The camera on its shelf, and the room from the camera's point of view (use the full-res recording).
- The phone buzzing and the WhatsApp message appearing (screen recording, or film the phone in hand).
- Stills: `zones_overlay.jpg`; the architecture diagram from the README (screenshot the rendered Mermaid);
  the two results tables from the README; `eval/SENSITIVITY.md`; a terminal with `pytest` passing.
- One eval clip with the box overlay (`python eval/make_gifs.py gmdcsa_S1_Fall_12 --out eval/out/gifs/demo.gif --label fall`).
  Caption the dataset: "GMDCSA24 (MIT)". If you use a UR Fall clip, caption "UR Fall (CC BY-NC-SA 4.0)".
- The GitHub commit list, scrolled, as the "built during the hackathon" shot.

## Script (voice-over, ~3:00)

Times are targets. Speak plainly; the footage carries it.

**0:00 to 0:10, hook and provenance.** The rules want one clear provenance sentence, spoken and on screen, inside
the first ten seconds. Camera view: you walk in and fall onto the rug. Caption from the first frame:
"remy · built from scratch Oct 29 to Nov 4, 2026 · github.com/Sahaj0312/remy".

> This is remy, a fall detector for my grandmother. I built all of it from scratch during this hackathon, starting
> October 29.

**0:10 to 0:30, the hook pays off.** Terminal: `candidate drop`, `SAM 3.1 confirms`, `EVENT raised ... drop->event
9.3s`. The camera speaker asks "Are you okay? Can you hear me?". You: "Yeah, I'm okay." The phone shows the calm message.
Trim the waits; keep the timestamps visible so nobody thinks it is faked.

> It notices a fall, asks if she is okay, and tells the family what she said. Nani lives in a house where nobody is
> in the room most of the day. Pendants get left on the table. The camera was already there.

The terminal now names every model as it runs: `SAM 3.1 confirms`, `Muse Spark 1.3 triage`, `check-in: Muse Voice
Transcribe`, `classified okay (Muse Spark 1.3)`. Keep those lines readable; the rubric wants the live Meta calls
visible on screen.

**0:30 to 1:25, what it does.** Scenarios C, D, A in that order, each trimmed to 10 to 15 s with the page
and the phone side by side. (Filmed Nov 3: B calm at 27 s, C urgent at 57 s, D urgent at 20 s, A quiet.)

> It stops listening two seconds after she stops talking, so a clear okay reaches the family under thirty
> seconds after the fall. If she does not answer, it asks again, then sends an urgent message with the video of
> the fall.
>
> This time I shouted "help". The camera microphone heard a voice but the transcriber made no words of it. That
> still escalates, at once: the family gets an urgent message twenty seconds after the fall, with the sound of
> the shout in it.
>
> And lying on the sofa is not a fall. The sofa and the bed are zones it learned from one frame, and the pose
> model's hips decide where she is, so feet up on the coffee table do not count.

**1:25 to 2:05, how it works.** The diagram, then the log with timings, then `zones_overlay.jpg`.

> A small pose model watches the room on the laptop at five frames a second. When it sees a fast drop, it cuts
> the five seconds around it and sends that clip to SAM 3.1, which tracks the person frame by frame. That takes
> about nine seconds. A rule on the track checks that the drop was fast in body lengths, that she ended up
> horizontal, and that she has not got up. Muse Spark looks at a short burst and writes the one-line note for
> the family. The camera's own speaker asks the question. Muse Voice Transcribe turns the answer into text, and
> Spark decides whether that text is a clear "I'm okay". Only a clear okay is reassuring. Anything else escalates.

**2:05 to 2:35, how well it works.** The eval table, the latency table, the `pytest` terminal, the reconnect
lines, `SENSITIVITY.md`.

> On a hundred public clips it catches thirty-two of forty falls with one false alarm, push-ups, and eighty
> percent out of sample. The misses are slow falls from sitting or onto a bed; those look like lying down, and the
> floor rule is the answer for now. I also ran Meta's open-weight Spark on the same thirty-three clips: it agrees
> with the hosted one within a step on twenty-nine, so the judge could run on hardware I control later. On the
> real camera, every staged fall got a message that matched what happened. The camera microphone is the weak
> link: a shouted "help" came through as no words, so the policy escalates on anything unclear, with the audio.
> One event costs about two cents in API calls. The whole project so far is about four dollars.

**2:35 to 2:55, privacy and what is next.**

> Nothing streams anywhere. The only things that leave the house are the five-second clip around a drop, a
> two-second burst, and the few seconds of audio after the question, until she has answered. Next: a second
> camera for the bedroom, and Glimmer as a backend so the triage can run on hardware I control.

**2:55 to 3:00, close.** The repo on screen, the camera on the shelf.

> remy. Built for one person, in one week. The code is public.

## Editing notes

- Footage from Nov 3: `tapo/cuts/{A_sofa,B_fall_okay,C_fall_silent,D_fall_help}.mp4` (camera, 1080p, with room
  sound; the fall is 8 to 10 s into each cut), the page screen recordings, the phone screen recording, and each
  event's own `clip.mp4`, `answer_r1.mp4` and `alert.mp4` under `events/`.
- Side-by-side layout for the scenarios: camera view left, the page top right, phone bottom right. The
  timestamps in the page banner and the phone's clock are the proof it is live.
- Never speed up the audio of the check-in; the question and the answer are the demo.
- Subtitles for every spoken line from the room; the camera speaker is quiet on a phone recording.
- Keep the honest lines (the misheard "help", the slow-fall misses). Judges trust a demo that admits a limit.
