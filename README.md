# Multicam Cutter

Turn a lecture recorded with several cameras (plus a Zoom recording) into tight, watchable
clips — in the browser, on your own computer.

- **Sync by sound.** Drop the camera files and the Zoom recording in one folder; the tool aligns
  them by audio. No clapperboard, no manual offsets.
- **Edit by words.** A word-level transcript of the whole lecture (Whisper, runs locally). Select
  words, press Delete. One click removes dead air and fillers. Every cut is placed in the real
  silence around the words and pose-matched, so the result looks like an ordinary edit.
- **Pick cameras per clip.** Camera angles are proposed automatically from where the speaker is
  looking; change them by pressing 1/2/3 while it plays. Reframe/zoom any shot.
- **Add pictures.** The tool suggests moments where an image helps and fetches a freely licensed
  one (Wikipedia / Wikimedia Commons / Openverse, optionally Unsplash) with its credit.
- **Export** frame-exact 1080p clips with the best audio, transitions, images, fade-out and a
  transcript + credits text file.

Everything that matters runs locally and works offline after the first setup. An AI chat model is
**optional** (it only names topics and suggests image moments) — Claude, Grok/xAI, OpenAI, Ollama
or none at all.

## Install (Linux, macOS; Windows via WSL2)

Requirements: **Python 3.10–3.12** and **ffmpeg**. 8 GB RAM works (with a smaller Whisper model);
16 GB+ is comfortable. Any GPU is a bonus, none is required.

```bash
git clone https://github.com/prnbs82/multicam-cutter.git
cd multicam-cutter
./install.sh            # creates .venv, installs packages, downloads the models, installs the `multicam` command
```

- **Ubuntu/Debian**: `sudo apt install ffmpeg python3-venv` first.
- **macOS**: `brew install ffmpeg python@3.12` first (Intel and Apple Silicon both work).
- **Windows**: install WSL2 with Ubuntu (`wsl --install` in PowerShell), open Ubuntu, then follow
  the Linux steps. Keep the lecture folder on the Linux side (`~/lectures/…`, not `/mnt/c/…`) for fast
  ffmpeg I/O, and raise WSL's RAM cap in `C:\Users\<you>\.wslconfig` (`[wsl2]`, `memory=12GB`, `swap=8GB`).
  An NVIDIA GPU works for Whisper (CUDA) in WSL2, but **not** for video encoding — exports use the CPU there.
- **NVIDIA GPU**: `./install.sh --cuda` puts Whisper on the GPU (much faster transcripts).
- `./install.sh --no-models` skips the model downloads (they happen on first use instead).

Full dependency specification (system packages, Python packages and why, models downloaded at
runtime, hardware acceleration, pitfalls) — written so a person *or an AI agent* can set up a new
machine from it: [`SETUP.md`](SETUP.md).

The installer ends with a hardware report (`multicam doctor`): which video encoders work on this
machine, which Whisper model/device will be used, and anything missing.

## Run

```bash
multicam "/path/to/lecture folder"
```

The folder holds the camera files (`.mp4`/`.MOV`…), the Zoom recording (video + its audio file,
optionally its `.vtt` transcript). The first start scans, syncs and builds preview proxies (a few
minutes), then opens `http://127.0.0.1:8765/` in your browser. Later starts are instant;
`multicam` alone reopens the last folder, `multicam --stop` stops the server.

Files of one camera recorded in segments should share a name prefix (`record1_BZ.MOV`,
`record2_BZ.MOV`, …) — they are grouped into one angle. Add a camera later by dropping its file in
the folder and starting again.

## The three tabs

The interface is simple by default; the **Advanced** switch in the header reveals every setting
(thresholds, join tuning, export edges, encoder choice, AI assistant, Unsplash key, checkpoints…).

### 1 · Edit words & clips
Transcribe the lecture (once; runs in the background). Press **✨ Tighten automatically** — long
pauses are shortened and fillers removed. Click a word to seek, select words and press **Del** to
remove them, click a struck word to restore it, double-click to correct a misheard word. Select
the words of a passage and press **▣** to make a **clip** (edges land exactly on the words).
**Topics** (collapsed) segments the lecture into subjects and can make a clip per subject.

*How a cut is made:* the app keeps up to 0.35 s of the real silence after the last kept word and
before the next one, cuts inside it, inserts a short pause where none exists, pose-matches the two
sides and — if the speaker moved a lot — switches camera there. Nothing to configure.

### 2 · Cameras
Pick a clip. Angles are pre-filled from the speaker's gaze (or press **1/2/3** while it plays to
cut; a press near a join lands on it). Timeline ticks show joins (`|` cut, `⟋` dissolve, `▮` hold,
`≈` morph). **⤢ Frame shot** zooms/reframes the current camera block. **Export** renders the clip.

### 3 · Images
Pick a clip → **Suggest images**. Each card shows the sentence, the picture and its credit:
**✓ Use**, **Ignore**, **↻ try another**, or upload your own. **🔍 Find an image for …** asks for a
picture at the current moment. Preview scrubs around the image; **⟲** loops the transition.
Full-frame (optional slow zoom) or picture-in-picture (corner and size). **Export** from here too.

Exports go to `<lecture>/clips/<clip name>.mp4` (+ `.txt` with the transcript and image credits).

## Hardware: what runs where

| Task | Runs on | Notes |
|---|---|---|
| Video export | GPU encoder when proven (NVIDIA NVENC, Intel Quick Sync, Apple VideoToolbox) else CPU x264 | `auto` prefers the CPU over VA-API on desktop CPUs (better picture); pick VA-API under Advanced on weak machines. One export = one encoder; if a GPU encoder fails the export restarts on the CPU. |
| Transcript (Whisper) | NVIDIA GPU (CUDA) if present, else CPU | Model follows memory: large-v3 (≥ 12 GB RAM / ≥ 6 GB VRAM), medium, small. CPU large-v3 ≈ 1.5× realtime on 6 cores. |
| Gaze proposals, pose matching, morphs | CPU (MediaPipe, OpenCV) | streaming, RAM-bounded |
| Topic segmentation | CPU (or CUDA/MPS) | MiniLM embeddings, 90 MB download |
| Topic names, image moments | optional AI assistant | see below; keyword fallback otherwise |

`multicam doctor` prints the decisions for your machine. Override the encoder with the header
selector (Advanced) or `MULTICAM_ENCODER=libx264`. Clip length is limited by disk, not RAM —
everything streams; the 💻 badge shows what fits.

## AI assistant (optional)

Only two features use a chat model: naming topics and suggesting where an image would help. Without
one, topics get keyword names and image moments come from concrete nouns and proper names in the
transcript. To add one, open **Advanced → Images tab → 🤖 AI assistant**:

- **Claude CLI** — detected automatically if `claude` is installed.
- **Anthropic API key**.
- **OpenAI-compatible** — xAI **Grok** (`https://api.x.ai/v1`), OpenAI, or a local **Ollama**
  (`http://localhost:11434/v1`); presets fill in the URL and model.

Keys are stored in `~/.config/multicam/ai.json` on your computer only. Environment variables
`ANTHROPIC_API_KEY`, `XAI_API_KEY`, `OPENAI_API_KEY` are also picked up.

## Images and copyright

Suggested pictures come from Wikipedia/Wikimedia Commons and Openverse (public domain / Creative
Commons; the credit is burned in and listed in the `.txt`) and, with your own access key,
Unsplash (Unsplash License). Check the licence shown on each card before publishing; your own
uploads are your responsibility.

## Command line

```bash
multicam doctor [lecture]                     # hardware + setup report
python multicam.py setup "<lecture>"          # init + sync + proxies (what the launcher does on first start)
python multicam.py transcribe "<lecture>"     # whole lecture (or --range A B), --model auto|large-v3|medium|small
python multicam.py render "<lecture>" --clip "<name>" --tighten
python multicam.py topics|gaze|posematch|broll|capacity ...   # see --help
python selftest.py --make /tmp/synthetic      # end-to-end test on a generated fake lecture
```

Use `.venv/bin/python` (or `source .venv/bin/activate`) for the `python` commands.

## Saving, resuming, checkpoints

Everything autosaves within a second (`<lecture>/_multicam/cuts.json`, `tighten.json`); reopening
resumes on the same tab at the same position. **⛃ Checkpoint…** (Advanced) stores named snapshots
you can restore.

## Files

```
multicam-cutter/          the tool: multicam (launcher), multicam.py (CLI), server.py, render.py,
                          joins.py + envelope.py (cut rules), hw.py (hardware), llm.py (AI), web/index.html (UI)
<lecture>/_multicam/      project.json, sync.json, layout.json, proxies/, multiview.mp4, words.json,
                          tighten.json, cuts.json, topics.json, joinfix.json, gaze/, broll/, checkpoints/, render/, logs/
~/.config/multicam/       hw.json (encoder probe + choice), ai.json (AI assistant), keys.json (Unsplash), last.json
```

Design notes (why the cuts are placed the way they are, measured Whisper biases, ffmpeg pitfalls):
[`docs/design-notes.md`](docs/design-notes.md). Licence: MIT.
