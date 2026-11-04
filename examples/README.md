# Files for running remy without the hardware

| folder | what | source |
|---|---|---|
| `clips/gmdcsa_S1_Fall_12.mp4` | a public fall clip, the source for `remy run` | GMDCSA24 (Alam et al. 2024), MIT |
| `event_gmdcsa_S1_Fall_12/` | one recorded event from that clip, for `remy replay` | a run of remy on the clip above |
| `voice/` | the six spoken check-in lines; `audio/` overrides them | generated with Kokoro, voice `af_heart` |
| `replies/` | sample answers for `remy run --reply`: `okay`, `help`, `silent` | generated with Kokoro, voice `af_bella`; no real person |

The replies are clean synthetic speech and they skip the camera mic. They show that the decision logic works, not
how well a real voice carries from the floor to an 8 kHz camera microphone.
