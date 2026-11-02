# Triage backends compared

The same flagged clips, the same system prompt and the same tracker numbers. Hosted Spark 1.3 sees the burst as a video file (the live path) or as JPEG frames; Spark Open on Fireworks sees frames, because the partner endpoint takes images only. Urgency is a judgment call, so agreement is the interesting number, not accuracy.

| backend | clips | high / medium / low | lying still | median latency | mean cost | mean tokens in / out |
|---|---|---|---|---|---|---|
| Spark 1.3, video burst (Meta API) | 33 | 25 / 8 / 0 | 23/33 | 5.5 s | $0.0060 | 1495 / 973 |
| Spark 1.3, frames (Meta API) | 33 | 26 / 5 / 2 | 22/33 | 4.2 s | $0.0126 | 7548 / 756 |
| Spark Open, frames (Fireworks) | 33 | 15 / 12 / 6 | 19/33 | 8.0 s | $0.0119 | 7542 / 584 |

## Agreement

| pair | same urgency | within one step | same lying-still |
|---|---|---|---|
| Spark 1.3, video burst (Meta API) vs Spark 1.3, frames (Meta API) | 26/33 | 33/33 | 24/33 |
| Spark 1.3, video burst (Meta API) vs Spark Open, frames (Fireworks) | 19/33 | 29/33 | 23/33 |
| Spark 1.3, frames (Meta API) vs Spark Open, frames (Fireworks) | 20/33 | 29/33 | 26/33 |

## Per clip

| clip | type | Spark 1.3, video burst (Meta API) | Spark 1.3, frames (Meta API) | Spark Open, frames (Fireworks) |
|---|---|---|---|---|
| gmdcsa_S1_Fall_05 | fall:standing | high, still | high, still | medium, still |
| gmdcsa_S1_Fall_09 | fall:standing | high, still | high, still | medium, still |
| gmdcsa_S1_Fall_12 | fall:standing | high, still | high, still | medium, still |
| gmdcsa_S1_Fall_15 | fall:chair | high, still | high, still | high, still |
| gmdcsa_S2_Fall_02 | fall:standing | high, still | high, still | low, moving |
| gmdcsa_S2_Fall_04 | fall:standing | high, still | high, still | low, moving |
| gmdcsa_S2_Fall_10 | fall:standing | high, still | medium, moving | medium, moving |
| gmdcsa_S2_Fall_11 | fall:bed | high, still | high, still | high, still |
| gmdcsa_S3_Fall_05 | fall:slow | medium, moving | medium, moving | medium, moving |
| gmdcsa_S3_Fall_12 | fall:bed | medium, still | low, moving | low, moving |
| gmdcsa_S3_Fall_16 | fall:standing | high, still | high, still | medium, still |
| gmdcsa_S4_ADL_05 | adl:exercise-floor | medium, moving | low, moving | low, moving |
| gmdcsa_S4_Fall_02 | fall:bed | high, still | high, still | low, moving |
| gmdcsa_S4_Fall_10 | fall:chair | high, still | high, still | high, still |
| gmdcsa_S4_Fall_12 | fall:bed | high, still | high, still | high, still |
| gmdcsa_S4_Fall_15 | fall:bed | high, still | high, still | high, still |
| urfall_fall-01 | fall:standing | medium, moving | high, still | medium, still |
| urfall_fall-02 | fall:chair | high, still | high, moving | high, moving |
| urfall_fall-04 | fall:chair | high, moving | high, still | medium, moving |
| urfall_fall-05 | fall:standing | medium, moving | medium, moving | medium, moving |
| urfall_fall-07 | fall:standing | medium, moving | medium, moving | medium, moving |
| urfall_fall-08 | fall:chair | high, still | high, moving | low, still |
| urfall_fall-10 | fall:chair | high, moving | high, moving | high, moving |
| urfall_fall-11 | fall:standing | high, still | high, still | high, still |
| urfall_fall-13 | fall:standing | high, still | medium, moving | high, still |
| urfall_fall-14 | fall:chair | high, still | high, still | high, moving |
| urfall_fall-16 | fall:chair | high, still | high, still | high, still |
| urfall_fall-17 | fall:standing | high, still | high, still | medium, still |
| urfall_fall-19 | fall:standing | high, still | high, still | high, still |
| urfall_fall-20 | fall:chair | medium, moving | high, moving | medium, moving |
| urfall_fall-22 | fall:chair | high, still | high, still | high, still |
| urfall_fall-25 | fall:standing | medium, moving | high, still | high, still |
| urfall_fall-28 | fall:chair | high, moving | high, still | high, still |

## Family notes, side by side (first five)

- **gmdcsa_S1_Fall_05**
  - Spark 1.3, video burst (Meta API): "Went down quickly near the chair 2.4 seconds into the clip and has been lying still on the floor for 3 seconds."
  - Spark 1.3, frames (Meta API): "Went down to the floor in the middle of the room by the blue door at 2.4s and has been lying still on his side for 3 seconds."
  - Spark Open, frames (Fireworks): "Went down slowly to the floor near the blue door at 2.4 seconds into the clip and has been lying still on his side for about 3 seconds."
- **gmdcsa_S1_Fall_09**
  - Spark 1.3, video burst (Meta API): "Went down quickly in front of the blue doors about 1.9s into the clip and has stayed still on the floor for 2.5 seconds."
  - Spark 1.3, frames (Meta API): "Went down face-first on the floor in front of the blue door 1.9 seconds into the clip and has been lying still for 2.5 seconds."
  - Spark Open, frames (Fireworks): "Went down to the floor in the center of the room near the blue doors at 1.9s into the clip and has been lying there for 2.5 seconds without getting up."
- **gmdcsa_S1_Fall_12**
  - Spark 1.3, video burst (Meta API): "Went down to the floor in the middle of the blue room at 2.7s into the clip and has been lying still for 3 seconds."
  - Spark 1.3, frames (Meta API): "Went down to the floor in the middle of the room by the blue door at 2.7s into the clip and has been lying still for 3.0 seconds."
  - Spark Open, frames (Fireworks): "Went down to the floor near the blue doors at 2.7 seconds into the clip and has been lying still on their back for about 3 seconds."
- **gmdcsa_S1_Fall_15**
  - Spark 1.3, video burst (Meta API): "Slipped off the chair onto the floor near the blue doors about 2.7 seconds into the clip and has been lying still for 3 seconds."
  - Spark 1.3, frames (Meta API): "Went down off the chair near the blue doors at 2.7s and has been lying still on the floor for 3.0 seconds."
  - Spark Open, frames (Fireworks): "Fell from the chair to the floor at 2.7 seconds into the clip near the blue doors and has been lying still on the floor for about 3 seconds without getting up."
- **gmdcsa_S2_Fall_02**
  - Spark 1.3, video burst (Meta API): "Went down onto the floor mat in front of the bed at 3.5 seconds and has been lying still for 3 seconds."
  - Spark 1.3, frames (Meta API): "Went down onto the mat in front of the bed at 3.5 seconds and has been lying still for about 3 seconds."
  - Spark Open, frames (Fireworks): "Went down slowly onto the mat in the center of the bedroom around 3.5s into the clip and is lying on his side, looking around and moving slightly."

## The reply classifier on 24 fixed transcripts

What the model decides from the words alone (the help-word rule is bypassed here). Expected labels are a careful listener's.

| backend | agree with expected | median latency | mean cost |
|---|---|---|---|
| Spark 1.3 (Meta API) | 24/24 | 1.5 s | $0.0019 |
| Spark Open (Fireworks) | 24/24 | 2.5 s | $0.0013 |

| transcript | expected | spark | spark_open |
|---|---|---|---|
| Yeah. | okay | okay | okay |
| I'm fine | okay | okay | okay |
| yes yes I'm alright | okay | okay | okay |
| no no I'm fine, just slipped | okay | okay | okay |
| all good | okay | okay | okay |
| haan main theek hoon | okay | okay | okay |
| theek hai | okay | okay | okay |
| I'm okay, don't worry | okay | okay | okay |
| help | needs_help | needs_help | needs_help |
| help me please | needs_help | needs_help | needs_help |
| bachao | needs_help | needs_help | needs_help |
| I can't get up | needs_help | needs_help | needs_help |
| my hip hurts a lot | needs_help | needs_help | needs_help |
| call someone | needs_help | needs_help | needs_help |
| it hurts | needs_help | needs_help | needs_help |
| Hello? | unclear | unclear | unclear |
| what time is it | unclear | unclear | unclear |
| and now the weather for tomorrow | unclear | unclear | unclear |
| hmm | unclear | unclear | unclear |
| who is that | unclear | unclear | unclear |
| the news at six | unclear | unclear | unclear |
| I'm not okay | needs_help | needs_help | needs_help |
| not really | needs_help | needs_help | needs_help |
| ow ow ow | needs_help | needs_help | needs_help |
