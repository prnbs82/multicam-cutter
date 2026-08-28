#!/usr/bin/env bash
# Multicam Cutter — one-shot installer for Linux and macOS (Windows: install WSL2/Ubuntu first, then run this inside it).
#   ./install.sh              CPU-only PyTorch (works everywhere), downloads the models it will need
#   ./install.sh --cuda       PyTorch + CTranslate2 with CUDA for NVIDIA GPUs (Whisper on the GPU)
#   ./install.sh --no-models  skip the model downloads (they happen on first use instead)
set -euo pipefail
cd "$(dirname "$0")"
CUDA=0; MODELS=1
for a in "$@"; do case "$a" in --cuda) CUDA=1;; --no-models) MODELS=0;; -h|--help) sed -n 2,5p "$0"; exit 0;; esac; done
OS="$(uname -s)"
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/5  Checking ffmpeg and Python"
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "ffmpeg is missing. Install it and re-run:"
  if [ "$OS" = Darwin ]; then echo "    brew install ffmpeg      (get Homebrew from https://brew.sh)"; else echo "    sudo apt install ffmpeg   # Debian/Ubuntu (Fedora: sudo dnf install ffmpeg)"; fi
  exit 1
fi
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Need Python 3.10, 3.11 or 3.12 (mediapipe has no wheels for other versions)."
  if [ "$OS" = Darwin ]; then echo "    brew install python@3.12"; else echo "    sudo apt install python3.12 python3.12-venv   (or python3.11 / python3.10)"; fi
  exit 1
fi
echo "using $($PY --version) at $(command -v $PY); ffmpeg $(ffmpeg -version | head -1 | awk '{print $3}')"

say "2/5  Creating the Python environment (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

say "3/5  Installing Python packages (this takes a few minutes the first time)"
if [ "$OS" = Linux ] && [ "$CUDA" = 0 ]; then
  python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install --quiet -r requirements.txt
if [ "$CUDA" = 1 ]; then
  # CUDA runtime + cuDNN wheels so faster-whisper (CTranslate2) can use the GPU without a system CUDA install
  python -m pip install --quiet nvidia-cublas-cu12 nvidia-cudnn-cu12
fi

if [ "$MODELS" = 1 ]; then
  say "4/5  Downloading models (Whisper for the transcript, MiniLM for topics) — sizes depend on your RAM"
  python - <<'PY'
import hw
dev, ctype, model = hw.whisper_choice()
print(f'  Whisper {model} ({dev}, {ctype}) ...', flush=True)
from faster_whisper.utils import download_model
download_model(model)
print('  MiniLM ...', flush=True)
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('  models ready')
PY
else
  say "4/5  Skipping model downloads (they happen on first use)"
fi

say "5/5  Installing the 'multicam' command"
mkdir -p "$HOME/.local/bin"
ln -sf "$PWD/multicam" "$HOME/.local/bin/multicam"
chmod +x multicam
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "NOTE: add ~/.local/bin to your PATH (e.g. echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc or ~/.zshrc)";; esac
if [ "$OS" = Linux ] && [ -d "$HOME/.local/share/applications" ]; then
  cat > "$HOME/.local/share/applications/multicam-cutter.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Multicam Cutter
Comment=Cut a multi-camera lecture recording
Exec=$PWD/multicam
Terminal=true
Categories=AudioVideo;Video;
DESK
fi

say "Hardware check"
python multicam.py doctor || true
cat <<TXT

Done. Start the tool with:
    multicam "/path/to/your lecture folder"
(the folder holds the camera files, the Zoom recording and its audio; the first start analyses and syncs them)
TXT
