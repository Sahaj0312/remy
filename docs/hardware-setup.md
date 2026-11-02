# Hardware setup

What the live monitor expects from the camera and the machine it runs on. Tested with one TP-Link Tapo C200
on home Wi-Fi and a MacBook on the same network.

## Tapo C200 settings (Tapo app)

1. **Camera account.** Camera → settings (gear) → Advanced settings → Camera account → create a username and
   password. These go into `.env` as `TAPO_USER` / `TAPO_PASS`. They are only for RTSP; they are not your
   Tapo login.
2. **Third-Party Compatibility: on.** Same Advanced settings page. Without it the camera refuses go2rtc's
   `tapo://` session with 401 and the speaker stays silent. RTSP works either way.
3. **Motion tracking: off** (pan/tilt models). The zones are polygons in the frame, so the view must not move.
   Also turn off any "privacy mode" schedule.
4. **Fixed placement.** Mount high (door-frame height or above), looking down across the room so the floor is
   large in the frame and the sofa and bed are visible. A low shelf view hides the floor behind furniture. If
   you move the camera, run `remy setup-zones` again. The old zones stay valid for nothing.
5. **Time.** The camera clock can drift (ours was one hour behind the Mac); it only affects the timestamp burned
   into the video, not the alerts, which use the Mac clock.
6. **Night vision.** Leave on Auto. The infrared image is black and white; YOLO still finds people in it, but the
   eval set has only 16 night clips, so expect less margin.
7. **Static IP.** Give the camera a DHCP reservation in the router so `TAPO_IP` does not change.

Streams: `rtsp://user:pass@IP:554/stream1` is 1920x1080 H.264 at about 15 fps with 8 kHz A-law audio;
`stream2` is 1280x720. The monitor uses stream2 for video (scaled to 640 px wide) and stream1 for the
microphone.

## Speaker: go2rtc

Two-way audio to the C200 needs the Tapo cloud protocol, which [go2rtc](https://github.com/AlexxIT/go2rtc) speaks.

1. Download the go2rtc binary for your platform into `tools/go2rtc/go2rtc` and `chmod +x` it (version 1.9.x works).
2. Put your Tapo **app login** password in `.env` as `TAPO_CLOUD_PASS`. go2rtc uses the cloud password, not the
   camera account.
3. `remy` writes `tools/go2rtc/go2rtc.yaml` from `infra/go2rtc.template.yaml` and starts go2rtc on
   127.0.0.1:1984 when it needs to speak. Nothing else is exposed.
4. Check it: `python -m remy.camera.go2rtc` plays `audio/are_you_okay.wav` through the camera. `--try-all` cycles
   the codecs if you hear nothing; `#audio=pcma` is what the C200 accepts.

The camera microphone cannot hear its own speaker, so a recording made while the prompt plays does not contain it.

## Check-in voice: Kokoro

Prompts are generated locally with [Kokoro](https://github.com/thewh1teagle/kokoro-onnx). Download
`kokoro-v1.0.onnx` and `voices-v1.0.bin` into `tools/kokoro/`, install `espeak-ng` (`brew install espeak-ng`),
then run `remy test-checkin --make-prompts`. Edit the text, name and voice under `checkin` in `config.json`.

## Zones

`remy setup-zones --stream 2` grabs one frame, asks SAM 3.1 for `floor`, `rug`, `bed` and `sofa` masks and writes
`zones.json` plus `zones_overlay.jpg`. Look at the overlay: the floor polygon must cover where a person would lie,
and the sofa and bed polygons must cover where a person lying down is fine. Add prompts with
`--prompts floor,rug,bed,sofa,mattress` if a piece of furniture is missing. The rules that use the zones:

- a fall is confirmed only if the body ends up outside sofa/bed polygons;
- the floor rule raises an event with no visible drop when someone lies still on floor/rug for 20 s.

Each prompt is one SAM image request (about $0.0025).

## macOS Local Network and ffmpeg

On recent macOS, a Python binary started from the terminal is not allowed to open sockets to LAN hosts
(`EHOSTUNREACH`) unless the terminal app has Local Network permission, and Homebrew Python often has no entry
to toggle. `ffmpeg` and `ffprobe` are allowed. So every RTSP read in `remy` goes through an ffmpeg subprocess
(`remy/camera/rtsp.py`, `remy/camera/audio_in.py`), and the webcam (`--source 0`) needs Camera permission for the
terminal app (System Settings → Privacy & Security → Camera).

Install the tools with `brew install ffmpeg espeak-ng`. `cloudflared` is only needed for the WhatsApp webhook
receiver and for Twilio image URLs.

## Machine

A laptop CPU is enough: YOLO11n-pose at 5 fps costs about 60 ms per frame on an Apple M-series CPU, and the heavy
work (SAM, Spark, transcription) is in the Meta Model API. The monitor needs a stable connection to the camera
and the internet; the check-in needs the camera speaker and microphone.
