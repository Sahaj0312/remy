# Demo video: shot list and script

Three minutes for the judges. Rule to keep in mind: the video must show only what was built during the
hackathon, and judges must be able to tell what that is. Everything in remy qualifies; say so once on screen.

## Before you film (10 minutes)

1. Message the test number from your phone (any text) so the 24 h free-form window is open. Then check:
   `remy test-alert --text "demo starting"`. If that fails with a window error, message the number again.
2. Tapo app: speaker volume to maximum, privacy mode off, lights on in the room, daytime.
3. Terminal: big font, dark theme, window wide enough that log lines do not wrap. Start screen recording
   (cmd+shift+5, record the terminal window only).
4. Phone: open the WhatsApp chat with the test number, start iOS screen recording (Control Center). Keep it
   running the whole session; it is cheaper to cut later than to miss a message arriving.
5. Record the camera's own view at full resolution for the whole session (the event clips are 640 px, 10 fps):

   ```sh
   mkdir -p tapo && ffmpeg -rtsp_transport tcp -i "$(.venv/bin/python -c 'from remy.common import tapo_rtsp; print(tapo_rtsp(1))')" \
     -t 2400 -c:v copy -c:a aac tapo/demo_$(date +%H%M).mp4
   ```

   Keep the terminal time visible somewhere; it is how you line this up with the log.
6. Start the monitor in a second terminal: `remy run --source tapo --label demo`. Wait for the `zones:` line.
7. A second phone or a friend filming you from the doorway gives the "room" shots (you walking in, the fall,
   the speaker asking). If you are alone, prop the phone on the desk pointing at the rug.

## Scenarios to run, in this order

Wait for the `done`/`ALERT` line and about 40 s of cooldown between scenarios. Stand up and walk out of the
frame between them.

| # | what you do | what should happen | caption for the video |
|---|---|---|---|
| A | walk in, sit on the sofa, lie down on it for 60 s, get up, walk around | nothing | "Sofa and bed are zones. Lying there is fine." |
| B | walk to the rug, fall, stay down, answer "Yeah, I'm okay" clearly | calm WhatsApp ≈30 s after the fall | "A clear okay. Calm message, no photo." |
| C | fall, stay down, say nothing | two prompts, then urgent WhatsApp with the snapshot ≈50 s | "No answer twice. Urgent, with a snapshot." |
| D | fall, stay down, say "Help! Help me!" loudly toward the camera | urgent WhatsApp, reason "She asked for help." | "A help word escalates at once." |
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

**0:00 to 0:15, the hook.** Camera view: you walk in and fall onto the rug. Cut to the terminal: `candidate
drop`, `SAM confirms`, `EVENT raised ... drop->event 9.3s`. The camera speaker says "Nani, are you okay?". You:
"Yeah, I'm okay." The phone shows the calm message. Trim the waits; show the timestamps so nobody thinks it is
faked.

> This is remy. A camera in my grandmother's living room that notices a fall, asks if she is okay, and tells the
> family what she said.

**0:15 to 0:30, why.** The camera on the shelf. The empty room.

> My Nani lives in a house where nobody is in the room most of the day. Pendants get left on the table. The camera
> was already there. I built this during the Meta hackathon, between October 29 and November 4.

Caption: "Everything shown was built Oct 29 to Nov 4, 2026. github.com/Sahaj0312/remy"

**0:30 to 1:25, what it does.** Scenarios C, D, E, A in that order, each trimmed to 10 to 15 s with the
terminal and the phone side by side.

> If she does not answer, it asks again, then sends an urgent message with a photo. That is fifty seconds after
> the fall.
>
> A help word skips the second question. The family hears she asked for help.
>
> This one has no visible drop. She sat down on the floor and stayed still, so after twenty seconds it asks anyway.
>
> And lying on the sofa is not a fall. The sofa and the bed are zones it learned from one frame.

**1:25 to 2:05, how it works.** The diagram, then the log with timings, then `zones_overlay.jpg`.

> A small pose model watches the room on the laptop at five frames a second. When it sees a fast drop, it cuts
> the five seconds around it and sends that clip to SAM 3.1, which tracks the person frame by frame. That takes
> about nine seconds. A rule on the track checks that the drop was fast in body lengths, that she ended up
> horizontal, and that she has not got up. Muse Spark looks at a short burst and writes the one-line note for
> the family. The camera's own speaker asks the question. Muse Voice Transcribe turns the answer into text, and
> Spark decides whether that text is a clear "I'm okay". Only a clear okay is reassuring. Anything else escalates.

**2:05 to 2:35, how well it works.** The eval table, the latency table, the `pytest` terminal, the reconnect
lines, `SENSITIVITY.md`.

> On a hundred public clips it catches thirty-two of forty falls with one false alarm, push-ups. The misses are
> slow falls from sitting or onto a bed; those look like lying down, and the floor rule is the answer for now.
> On the real camera, every staged fall got the right message. The camera microphone is the weak link: it
> misheard "help" once, so the policy escalates on anything unclear. One event costs about two cents in API calls.
> The whole project so far is under three dollars.

**2:35 to 2:55, privacy and what is next.**

> Nothing streams anywhere. The only things that leave the house are the five-second clip around a drop, a
> two-second burst, and the fifteen seconds of audio after the question. Next: stop recording when she stops
> talking, so the message lands in twenty seconds instead of thirty; a second camera for the bedroom; and
> Glimmer as a backend so the triage can run on hardware I control.

**2:55 to 3:00, close.** The repo on screen, the camera on the shelf.

> remy. Built for one person, in one week. The code is public.

## Editing notes

- Side-by-side layout for the scenarios: camera view left, terminal top right, phone bottom right. The
  timestamps in the terminal and the phone's clock are the proof it is live.
- Never speed up the audio of the check-in; the question and the answer are the demo.
- Subtitles for every spoken line from the room; the camera speaker is quiet on a phone recording.
- Keep the honest lines (the misheard "help", the slow-fall misses). Judges trust a demo that admits a limit.
