# Silence stop on the real check-in recordings

10 recordings from the camera mic (not in git). Parameters: {"speech_db": -40, "loud_db": -30, "silence_db": -47, "speech_min_s": 0.2, "silence_s": 2.0, "min_s": 4.0, "max_s": 15}. `cut` is where the recording would have ended; `said` is what Muse Voice Transcribe returned for the full recording, `after cut` for the shortened one.

| event | recording | full | speech at | cut | saved | said | after cut |
|---|---|---|---|---|---|---|---|
| 20261030_164421_sim_dr | checkin_r1_164438.wav | 14.6 s | - | no cut | - | '' | '-' |
| 20261030_164421_sim_dr | checkin_r2_164458.wav | 14.6 s | - | no cut | - | '' | '-' |
| 20261031_174554_test5_ | checkin_r1_174606.wav | 14.5 s | 3.7 | 5.85 s | 8.7 s | 'Yeah.' | 'Yeah.' |
| 20261031_174558_test5_ | checkin_r1_174610.wav | 15.1 s | 0.25 | 4.0 s | 11.1 s | 'Yeah.' | 'Yeah.' |
| 20261031_174809_test5_ | checkin_r1_174820.wav | 15.1 s | - | no cut | - | '' | '-' |
| 20261031_174809_test5_ | checkin_r2_174841.wav | 15.1 s | - | no cut | - | '' | '-' |
| 20261031_174954_test5_ | checkin_r1_175005.wav | 15.1 s | 0.65 | 5.4 s | 9.7 s | '' | '' |
| 20261031_174954_test5_ | checkin_r2_175026.wav | 14.7 s | 0.15 | 5.7 s | 9.0 s | 'Hello?' | 'Hello?' |
| 20261101_190055_restru | checkin_r1_190111.wav | 14.5 s | - | no cut | - | '' | '-' |
| 20261101_190055_restru | checkin_r2_190132.wav | 15.0 s | - | no cut | - | '' | '-' |

Recordings with words: 3, all cut early; recordings with silence: 6, none cut; speech-level sound but no words: 1 (these now count as unclear and escalate at once).
