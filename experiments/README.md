# experiments

One-off probes and superseded versions, kept because the README quotes their numbers. They import
`remy`, so run them after `pip install -e .` from the repo root.

| script | what it was for |
|---|---|
| `live_v1.py` | the first live pipeline (fixed 3 s wait, SAM, Spark; no zones, check-in or alerts); drop-to-event was 19 to 26 s, which motivated the v2 flow in `remy/monitor.py` |
| `tapo_probe.py` | measured the C200 streams: resolution, fps, time to first frame, jitter, glass-to-glass latency with a screen flash; writes snapshots to `tapo/` |
| `tapo_latency.py` | measured the audio-path lag of the RTSP stream with a beep from the Mac speaker: 0.2 to 0.45 s |
| `tapo_audio.py` | recorded the camera mic and tried Muse Voice Transcribe modes and audio levels before the check-in was built |
| `tapo_speaker_check.py` | proved the speaker path: go2rtc plays the phrase, the camera mic records, the transcript is checked (the camera mic cannot hear its own speaker, so this needs a second device or a quiet room) |
| `make_tts_piper.py` | the first "Are you okay?" prompt with Piper TTS, replaced by Kokoro (`remy test-checkin --make-prompts`) |

`python -m remy.camera.go2rtc --try-all` is the codec probe that found `#audio=pcma` for the speaker.
