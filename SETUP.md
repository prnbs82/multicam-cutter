# SETUP.md — dependencies and environment for Multicam Cutter

This file is written for a person *or an LLM agent* that has to install and run the tool on a
machine it has never seen. It lists every dependency, what it is for, what is optional, where
things are stored, and how to verify the installation. `install.sh` automates the Python part;
this document is the specification behind it.

## 1. Supported platforms

| Platform | Status | Notes |
|---|---|---|
| Linux x86_64 (Ubuntu 22.04+, Debian 12+, Fedora 39+) | supported | reference platform |
| macOS 13+ (Intel and Apple Silicon) | supported | Homebrew for ffmpeg + Python |
| Windows 10/11 | via **WSL2** (Ubuntu) only | native Windows is not supported: `server.py` uses POSIX process groups (`os.killpg`, `start_new_session`) |
| Raspberry Pi / ARM Linux | untested | mediapipe wheels exist for aarch64; expect slow transcripts |

Minimum hardware: 4 CPU cores, 8 GB RAM, 20 GB free disk. Comfortable: 6+ cores, 16 GB RAM.
GPUs are optional (see §6).

## 2. System dependencies (install with the OS package manager)

| Dependency | Required? | Version | Purpose | Install |
|---|---|---|---|---|
| `ffmpeg` + `ffprobe` | **required** | ≥ 4.4 (5.x/6.x/7.x fine), must include `libx264`, `aac`, filters `xfade`, `tpad`, `zoompan`, `drawtext` (needs libfreetype), `lavfi` | all video/audio processing | `sudo apt install ffmpeg` · `brew install ffmpeg` · `sudo dnf install ffmpeg` (Fedora: enable RPM Fusion) |
| Python | **required** | **3.10, 3.11 or 3.12** (not 3.13+: no mediapipe wheels) | everything | `sudo apt install python3 python3-venv python3-pip` · `brew install python@3.12` |
| `curl` | required by launcher | any | readiness check | usually preinstalled |
| `lsof` or `fuser` (psmisc) | recommended | any | launcher stops a stale server on the port | `sudo apt install lsof` (macOS has lsof) |
| `xdg-open` (Linux) / `open` (macOS) | optional | – | open the browser automatically | part of `xdg-utils` |
| `git` | optional | any | cloning / updating | `sudo apt install git` |
| A web browser | **required** | Firefox or Chromium/Chrome/Edge/Safari, recent | the UI runs at `http://127.0.0.1:8765/` | – |
| NVIDIA driver | optional | ≥ 525 (CUDA 12) | Whisper on the GPU, NVENC encoder | vendor installer; in WSL2 the Windows driver is enough |
| Intel media driver (`intel-media-va-driver`) or Mesa (AMD) | optional | – | VA-API / Quick Sync hardware encoders on Linux | `sudo apt install intel-media-va-driver-non-free` / `mesa-va-drivers` |

## 3. Python packages (`requirements.txt`)

Install into a virtual environment (`python3 -m venv .venv && . .venv/bin/activate`).

| Package | Pin | Purpose | Notes |
|---|---|---|---|
| `numpy` | `>=1.24,<2` | arrays everywhere | **must stay < 2** — mediapipe 0.10.15 is built against NumPy 1.x |
| `scipy` | `>=1.10` | audio cross-correlation for camera sync (`sync.py`), WAV io | |
| `psutil` | `>=5.9` | RAM/swap/cores report (`capacity.py`, `hw.py`) — portable replacement for `/proc/meminfo` | |
| `faster-whisper` | `>=1.1,<1.3` | word-level transcription (`transcribe.py`) | pulls `ctranslate2`, `tokenizers`, `huggingface_hub`, `onnxruntime` (VAD) |
| `mediapipe` | `==0.10.15` | face detection + face mesh (gaze proposals, `gaze.py`), pose landmarks (pose matching, `posematch.py`) | pulls `opencv-contrib-python`, `protobuf`, `absl-py`, `flatbuffers`, `jax`/`jaxlib` on some platforms. Uses the legacy `mp.solutions.*` API — do not upgrade to 0.10.20+ without checking it still exists |
| `torch` | `>=2.1` | MiniLM embeddings for topic segmentation (`topics.py`) | CPU build is enough; `install.sh` uses the CPU wheel index on Linux (`--index-url https://download.pytorch.org/whl/cpu`) to avoid a 2+ GB CUDA download. `--cuda` keeps the default (CUDA) wheel |
| `transformers` | `>=4.40,<5` | loads `sentence-transformers/all-MiniLM-L6-v2` | |
| `scikit-learn` | `>=1.3` | TF-IDF keyword labels + agglomerative clustering of topics | |
| `pillow` | `>=9` | placeholder / own-image handling in B-roll (`broll.py`) | |

Transitive but important: `opencv` (comes with mediapipe; used for DIS optical-flow morphs in
`morph.py`, frame decoding), `ctranslate2` (Whisper inference engine; needs CUDA 12 + cuDNN 9 for
GPU mode — `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` provides them without a system CUDA).

Versions verified working on the reference machine (Ubuntu 22.04, Python 3.10.12, ffmpeg 4.4.2):
mediapipe 0.10.15 · faster-whisper 1.2.1 · ctranslate2 4.8.1 · transformers 4.46.1 · torch 2.8.0
· scipy 1.14.1 · numpy 1.26.2 · psutil 5.9.5 · scikit-learn 1.6.1 · pillow 9.5.0.

No JavaScript build step: the UI is a single static file, `web/index.html`, served by
`server.py` (Python standard library `http.server`). No Node.js needed. No database.

## 4. Models downloaded at runtime (network needed once)

| Model | Size on disk | Downloaded by | Cache location | When |
|---|---|---|---|---|
| Whisper `large-v3` (CTranslate2) | ~3.1 GB | `faster_whisper` from Hugging Face (`Systran/faster-whisper-large-v3`) | `~/.cache/huggingface/hub/` | first transcript, or `install.sh` |
| Whisper `medium` / `small` | ~1.5 GB / ~0.5 GB | same | same | chosen automatically on machines with < 12 GB RAM (`hw.whisper_choice`) |
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | `transformers` | `~/.cache/huggingface/hub/` | first Topics analysis, or `install.sh` |
| MediaPipe face detection / face mesh / pose lite | bundled | inside the `mediapipe` wheel | – | – |
| Silero VAD (faster-whisper) | ~2 MB | bundled in the wheel | – | – |

Offline machines: run `install.sh` (without `--no-models`) once on a connected network, or copy
`~/.cache/huggingface/hub/models--Systran--faster-whisper-*` and `models--sentence-transformers--all-MiniLM-L6-v2`
across. Set `HF_HOME` to relocate the cache.

## 5. Optional external services

None are required. The tool never sends the video or audio anywhere.

| Service | Used for | Configuration |
|---|---|---|
| Wikipedia / Wikimedia Commons / Openverse APIs | free-licence image candidates for B-roll | no key; needs internet during "Suggest images" |
| Unsplash API | additional photos | access key in Advanced → Images tab, stored in `~/.config/multicam/keys.json` (`unsplash_access_key`) or env `UNSPLASH_ACCESS_KEY` |
| AI assistant (chat model) | topic names, image-moment suggestions only | `~/.config/multicam/ai.json`: `{"provider": "none|claude-cli|anthropic|openai", "base_url": ..., "model": ..., "api_key": ...}`; env `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `OPENAI_API_KEY` are auto-detected; `claude` CLI on PATH is auto-detected. Without any: keyword labels and noun-phrase moments |

OpenAI-compatible endpoints known to work with `provider: openai`: xAI Grok (`https://api.x.ai/v1`),
OpenAI (`https://api.openai.com/v1`), Ollama (`http://localhost:11434/v1`, no key), LM Studio.

## 6. Hardware acceleration (all optional, all auto-detected by `hw.py`)

| Hardware | What it accelerates | Requirements | Fallback |
|---|---|---|---|
| NVIDIA GPU | Whisper (CUDA float16, ~10× faster than CPU); export encoder `h264_nvenc` | driver ≥ 525; `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` (done by `install.sh --cuda`); ffmpeg built with nvenc (distro builds are) | CPU int8 Whisper; libx264 |
| Intel iGPU/Arc | export encoder `h264_qsv` / `h264_vaapi` | Intel media driver, `/dev/dri/renderD*` readable by the user (group `render` or `video`) | libx264 |
| AMD GPU (Linux) | export encoder `h264_vaapi` | Mesa VA drivers | libx264 — **and `auto` prefers libx264 on CPUs with ≥ 4 cores**, because AMD VCE quality is below x264 CRF 18 |
| Apple Silicon / Intel Mac | export encoder `h264_videotoolbox`; MiniLM on MPS | macOS ffmpeg (Homebrew) | libx264 |
| No GPU | – | – | everything runs on the CPU; only speed changes |

Encoders are *proven* by encoding test frames (`hw.prove_encoder`); the result is cached in
`~/.config/multicam/hw.json`. Force one with `MULTICAM_ENCODER=libx264|h264_nvenc|h264_qsv|h264_vaapi|h264_videotoolbox`.
A single export always uses one encoder; if a hardware encoder fails mid-way the export restarts on libx264.

## 7. Files, ports, environment variables

- Tool directory: the repository (no installation outside it except the `multicam` symlink in
  `~/.local/bin` and an optional Desktop entry in `~/.local/share/applications`).
- Per-lecture work directory: `<lecture folder>/_multicam/` (proxies, sync, transcript, edits, renders, logs).
  Safe to delete; regenerated on next start (proxies and transcript take time).
- User config: `~/.config/multicam/` (`hw.json`, `ai.json`, `keys.json`, `last.json`). Override the location with `MULTICAM_CONFIG=<dir>`.
- Port: `8765` on `127.0.0.1` only (override `MULTICAM_PORT`). Not reachable from other machines by design.
- Outputs: `<lecture folder>/clips/<clip name>.mp4` + `.txt`, or `<lecture folder>/multicam_output.mp4`.
- Environment variables: `MULTICAM_ENCODER`, `MULTICAM_CONFIG`, `MULTICAM_PORT`, `HF_HOME`,
  `ANTHROPIC_API_KEY` / `XAI_API_KEY` / `OPENAI_API_KEY`, `UNSPLASH_ACCESS_KEY`.
  When running from inside a Claude Code session, `llm.py` strips `CLAUDE*` variables before calling the `claude` CLI.

## 8. Install and verify (copy-paste)

```bash
# Ubuntu / Debian
sudo apt install -y ffmpeg python3 python3-venv python3-pip lsof git
# macOS
# brew install ffmpeg python@3.12 git

git clone https://github.com/<owner>/multicam-cutter.git
cd multicam-cutter
./install.sh                 # add --cuda on an NVIDIA machine, --no-models to defer downloads
```

Verification, in order:

```bash
.venv/bin/python multicam.py doctor            # 0 problems expected; lists proven encoders, Whisper device/model, AI provider
.venv/bin/python selftest.py --make /tmp/synthetic-lecture 8799   # generates a fake lecture, exercises the whole API + render; prints SELFTEST PASSED (~5 min)
multicam "/path/to/a real lecture folder"      # first start: init + sync + proxies, then opens the browser
```

Expected input folder: the camera recordings (`.mp4`/`.mov`/`.mkv`…, any resolution/fps; a camera
recorded in segments = files sharing a name prefix), the Zoom recording (`.mp4`) and its audio
(`.m4a`; used as the master audio), optionally the Zoom `.vtt` transcript. Files must overlap in time
(same event) — sync is by audio cross-correlation.

## 9. Known pitfalls

- **NumPy 2.x breaks mediapipe 0.10.15** — keep `numpy<2` (pinned).
- **Python 3.13** has no mediapipe wheel — use 3.10–3.12 (`install.sh` picks one automatically).
- **ffmpeg without libfreetype** cannot draw image credits (`drawtext`) — distro/Homebrew builds include it; static minimal builds may not.
- **`/dev/dri` permissions** — hardware encoders on Linux need the user in the `render`/`video` group; otherwise `doctor` shows them as unusable and libx264 is used (fine).
- **Whisper on CUDA fails with "libcudnn… not found"** — install `nvidia-cudnn-cu12` in the venv (`install.sh --cuda` does); the tool falls back to CPU automatically and says so in the transcript status.
- **Low RAM (< 6 GB free)** — analyses (gaze, pose, images) refuse to start and say why; close other programs. Exports stream and need disk, not RAM.
- **macOS Gatekeeper** — nothing to approve; everything is Python + Homebrew ffmpeg.
- **WSL2** — put the lecture folder on the Linux side (`~/lectures/...`), not `/mnt/c/...`, for 5–10× faster ffmpeg I/O. The browser on Windows opens `http://127.0.0.1:8765/` normally.
