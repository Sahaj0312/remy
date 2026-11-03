"""Small ffmpeg helpers for the event clips: frames (+ optional audio) to MP4, a still frame under an audio
track, and joining clips into the one video the family gets. Every clip gets an AAC track (silent if there is
no audio) so the pieces join cleanly."""
from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def resample(frames, fps):
    """[(t, frame)] with uneven t -> one frame per 1/fps, nearest in time."""
    if len(frames) < 2:
        return frames
    ts = np.array([t for t, _ in frames]); out = []; g = frames[0][0]
    while g <= frames[-1][0] + 1e-6:
        out.append((g, frames[int(np.argmin(np.abs(ts - g)))][1])); g += 1 / fps
    return out


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples.astype(np.float32), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())


def _mux(video: Path, dst: Path, audio_wav: Path | None, rate: int) -> None:
    """video (no audio) + wav -> dst with AAC; no wav -> a silent AAC track of the same length."""
    if audio_wav is not None:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio_wav), "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "48k", "-ar", str(rate), "-ac", "1", "-shortest", "-movflags", "+faststart", str(dst)]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono", "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "48k", "-shortest", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def write_clip(frames, fps: float, dst: Path, audio: np.ndarray | None = None, rate: int = 16000) -> Path:
    """frames: [(t, HxWx3 BGR)] -> H.264 MP4 at fps, with `audio` (mono float samples at `rate`) or a silent track."""
    frames = resample(frames, fps); h, w = frames[0][1].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "v.mp4"
        proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:g}",
                                 "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for _, fr in frames:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close(); proc.wait()
        wav = None
        if audio is not None and len(audio) > rate // 10:
            wav = Path(tmp) / "a.wav"; _write_wav(wav, audio, rate)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _mux(video, dst, wav, rate)
    return dst


def freeze_clip(frame: np.ndarray, audio: np.ndarray, dst: Path, rate: int = 16000, fps: float = 10.0) -> Path:
    """One still frame held under an audio track (when the frames for the moment are gone from the buffer)."""
    seconds = max(1.0, len(audio) / rate)
    frames = [(k / fps, frame) for k in range(int(round(seconds * fps)) + 1)]
    return write_clip(frames, fps, dst, audio, rate)


def concat_clips(paths: list[Path], dst: Path, width: int | None = None) -> Path:
    """Join clips in order, re-encoding so size and timing differences do not matter."""
    paths = [p for p in paths if p and Path(p).exists()]
    if not paths:
        raise ValueError("no clips to join")
    if len(paths) == 1:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(paths[0]), "-c", "copy", "-movflags", "+faststart", str(dst)], check=True, capture_output=True)
        return dst
    w = width or probe(paths[0])["width"]
    inputs = []; filt = []
    for i, p in enumerate(paths):
        inputs += ["-i", str(p)]
        filt.append(f"[{i}:v:0]scale={w}:-2,setsar=1,fps=10[v{i}];[{i}:a:0]aresample=16000,aformat=channel_layouts=mono[a{i}]")
    chain = "".join(f"[v{i}][a{i}]" for i in range(len(paths)))
    fc = ";".join(filt) + f";{chain}concat=n={len(paths)}:v=1:a=1[v][a]"
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "48k",
                    "-movflags", "+faststart", str(dst)], check=True, capture_output=True)
    return dst


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {"duration_s": float(j.get("format", {}).get("duration", 0) or 0), "width": v.get("width"), "height": v.get("height"),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}
