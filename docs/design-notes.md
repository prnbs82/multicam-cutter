# Design notes — why the tool does what it does

These are the non-obvious decisions and measurements behind the code. Read them before changing
`joins.py`, `envelope.py`, `render.py` or their JavaScript twins in `web/index.html`.

## Timeline model
- Everything is on the **master clock** of the Zoom audio. Each camera file has one sync offset
  (audio cross-correlation, `sync.py`), a phone recorded in segments gets one offset per segment.
- Word edits (`tighten.json`) remove intervals of the master clock. The Cameras tab draws its timeline in
  **tightened time** (kept intervals + time added by joins); `D()`/`Mm()` in the UI and `kept_intervals` /
  `expand_pieces` in `render.py` map between the two.

## Whisper word times are biased
Measured against the acoustic envelope on the lecture recording: word **starts are ~60 ms late**, word
**ends are ~90 ms early**, and consecutive words are contiguous (Whisper gives no pause information).
Cutting at raw word times clips consonants. All edges therefore go through the 10 ms half-dB envelope
(`_multicam/envelope.u8`, `envelope.py`).

## The natural-cut rule (edgeSnap 3)
When words are removed between a kept word P and a kept word N:
1. measure the real silence after P and before N (speech threshold = 95th percentile of the louder half
   of all hops − 15 dB);
2. keep up to **0.35 s** of silence on each side and cut inside it — never inside a removed word;
   contiguous speech (no silence at all) → cut at the midpoint of the two biases;
3. if the kept pauses together are **< 0.30 s**, insert the missing time to reach ~0.45 s: a still hold
   with room tone, or a dissolve from the removed footage when its edges are silent;
4. joins are pose-matched (MediaPipe, `posematch.py`) in the silent windows; motion 0.045–0.10 → optical-flow
   morph (`morph.py`, DIS flow validated photometrically, falls back to dissolve); motion ≥ 0.10 → the
   camera is **switched** at the join (a normal cut hides the jump).
Camera changes never add time. All of this runs identically in Python (export) and JavaScript (preview).

## Frame-exact rendering
- Pieces are encoded with `-ss` + `-frames:v` + `tpad`, then joined with the concat demuxer using
  **explicit `duration` lines** — pieces cut from the Zoom mp4 carry bogus container durations and the
  demuxer would otherwise truncate silently. `verify_output` compares frame counts and lengths.
- A listed piece file that does not exist also makes the concat demuxer stop silently: pose-fix times are
  quantised to the 30 fps grid, zero-frame items are dropped, and every piece is asserted to exist.
- One render uses **one** video encoder (hardware or libx264) — mixing encoders in a concat is unsafe.
  If a hardware encoder fails mid-way the export restarts on the CPU.

## Hardware policy (`hw.py`)
- Encoders are *proven* by encoding test frames, not just listed. `auto` prefers NVIDIA NVENC, Intel Quick
  Sync and Apple VideoToolbox; VA-API is used automatically only on weak CPUs (< 4 physical cores) because
  x264 CRF 18 looks better on a desktop CPU. `MULTICAM_ENCODER=<kind>` or the Advanced selector override it.
- Whisper runs on CUDA (float16) when CTranslate2 sees a GPU, else CPU int8; the model size follows RAM/VRAM
  (large-v3 ≥ 12 GB RAM or ≥ 6 GB VRAM, medium, small). AMD and Apple GPUs run Whisper on the CPU.
- MediaPipe, OpenCV and MiniLM embeddings run on the CPU everywhere (MiniLM uses CUDA/MPS when present).
- Memory: the pipeline streams; clip length is limited by disk and time, not RAM (`capacity.py`).
  Gaze analysis once buffered all frames of an 11-minute clip (28 GB) and was OOM-killed — keep decoders streaming.

## AI usage
Only two cosmetic features call a chat model: topic *names* and image-moment *suggestions*. Both have
heuristic fallbacks (TF-IDF keywords; concrete-object keywords and proper nouns). `llm.py` talks to the
Claude CLI, the Anthropic API or any OpenAI-compatible endpoint (xAI Grok, OpenAI, Ollama). Keys live in
`~/.config/multicam/ai.json`, never in a lecture folder or the repo.

## B-roll sources
Wikipedia page images → Wikimedia Commons search → Openverse (CC), optional Unsplash with the user's key.
Wikimedia rate-limits non-standard thumbnail widths (use 1280); a 429 blocks that host for 10 minutes.
Credits are stored per image and written into the export's `.txt`.
