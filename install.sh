#!/usr/bin/env bash
# Multicam Cutter installer for Linux and macOS (Windows: inside WSL2/Ubuntu).
#
# Shows exactly what it is going to install (system packages, Python packages, models, shortcuts),
# asks once, then installs everything quietly. Details go to install.log.
#
#   ./install.sh              interactive: show the plan, ask, install
#   ./install.sh --yes        no question (unattended / CI)
#   ./install.sh --dry-run    only show the plan
#   ./install.sh --no-models  do not download the models now (they download on first use)
#   ./install.sh --cuda       force the NVIDIA (CUDA) build of PyTorch/CTranslate2 (auto-detected normally)
#   ./install.sh --cpu        force the CPU build even if an NVIDIA GPU is present
set -uo pipefail
cd "$(dirname "$0")"
TOOL_DIR="$PWD"
LOG="$TOOL_DIR/install.log"; : > "$LOG"
YES=0; DRY=0; MODELS=1; GPU=auto
for a in "$@"; do case "$a" in
  --yes|-y) YES=1;; --dry-run) DRY=1;; --no-models) MODELS=0;; --cuda) GPU=cuda;; --cpu) GPU=cpu;;
  -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
  *) echo "unknown option: $a (see --help)"; exit 2;; esac; done

OS="$(uname -s)"; ARCH="$(uname -m)"
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; echo; echo "Details: $LOG"; tail -n 15 "$LOG"; exit 1; }
run()  { # run "description" cmd... — quiet, logged, one line of status
  local what="$1"; shift
  printf '  … %s' "$what"
  if "$@" >>"$LOG" 2>&1; then printf '\r'; ok "$what"; else printf '\r'; fail "$what"; fi
}

# ------------------------------------------------------------------ what is here, what is missing
PKG=""; SUDO=""
if [ "$OS" = Darwin ]; then PKG=brew
elif command -v apt-get >/dev/null; then PKG=apt
elif command -v dnf >/dev/null; then PKG=dnf
elif command -v pacman >/dev/null; then PKG=pacman
fi
[ "$OS" = Linux ] && [ "$(id -u)" != 0 ] && SUDO=sudo

SYS_PKGS=()      # system packages to install
NOTES=()
have() { command -v "$1" >/dev/null 2>&1; }

if ! have ffmpeg || ! have ffprobe; then
  case "$PKG" in apt|dnf|pacman) SYS_PKGS+=(ffmpeg);; brew) SYS_PKGS+=(ffmpeg);; esac
fi
have curl || { case "$PKG" in apt|dnf|pacman) SYS_PKGS+=(curl);; esac; }
have lsof || have fuser || { case "$PKG" in apt) SYS_PKGS+=(lsof);; dnf) SYS_PKGS+=(lsof);; pacman) SYS_PKGS+=(lsof);; esac; }

py_ok() { "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; }
PY=""
for c in python3.12 python3.11 python3.10 python3; do have "$c" && py_ok "$c" && { PY="$c"; break; }; done
NEED_PY=0
if [ -z "$PY" ]; then
  NEED_PY=1
  case "$PKG" in
    apt) SYS_PKGS+=(python3.12 python3.12-venv);;
    dnf) SYS_PKGS+=(python3.12);;
    pacman) SYS_PKGS+=(python);;                      # Arch ships the newest Python; may be 3.13 (see NOTES)
    brew) SYS_PKGS+=(python@3.12);;
  esac
  [ "$PKG" = pacman ] && NOTES+=("Arch's python may be newer than 3.12; if the doctor complains about mediapipe, install python312 from the AUR.")
elif [ "$PKG" = apt ] && ! "$PY" -c 'import venv, ensurepip' >/dev/null 2>&1; then
  SYS_PKGS+=("$(basename "$PY")-venv")
fi
if [ "$PKG" = brew ] && ! have brew; then NEED_BREW=1; else NEED_BREW=0; fi
if [ "$OS" = Linux ] && [ -z "$PKG" ] && [ ${#SYS_PKGS[@]} -gt 0 ]; then
  echo "This Linux has no apt/dnf/pacman; please install by hand: ${SYS_PKGS[*]}"; exit 1
fi

# GPU: NVIDIA -> CUDA build of torch + CUDA libs for Whisper
NVIDIA=0; have nvidia-smi && nvidia-smi -L >/dev/null 2>&1 && NVIDIA=1
[ "$GPU" = auto ] && { [ "$NVIDIA" = 1 ] && GPU=cuda || GPU=cpu; }
[ "$OS" = Darwin ] && GPU=cpu

# RAM -> Whisper model size (same rule as hw.whisper_choice)
if [ "$OS" = Darwin ]; then RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 )); else RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576 )); fi
if [ "$GPU" = cuda ]; then
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1); VRAM_MB=${VRAM_MB:-4000}
  if [ "$VRAM_MB" -ge 6000 ]; then WHISPER=large-v3; WSIZE="3.1 GB"; elif [ "$VRAM_MB" -ge 3500 ]; then WHISPER=medium; WSIZE="1.5 GB"; else WHISPER=small; WSIZE="0.5 GB"; fi
else
  if [ "$RAM_GB" -ge 12 ]; then WHISPER=large-v3; WSIZE="3.1 GB"; elif [ "$RAM_GB" -ge 6 ]; then WHISPER=medium; WSIZE="1.5 GB"; else WHISPER=small; WSIZE="0.5 GB"; fi
fi
[ "$GPU" = cuda ] && PYSIZE="~3.5 GB (PyTorch with CUDA, CUDA runtime + cuDNN, mediapipe, Whisper, transformers…)" || PYSIZE="~1.2 GB (PyTorch CPU build, mediapipe, Whisper, transformers…)"

# ------------------------------------------------------------------ the plan
echo
bold "Multicam Cutter — installation plan for this $([ "$OS" = Darwin ] && echo Mac || echo "Linux") ($ARCH, ${RAM_GB} GB RAM$([ "$NVIDIA" = 1 ] && echo ', NVIDIA GPU'))"
echo
n=1
if [ "${NEED_BREW:-0}" = 1 ]; then
  echo "  $n. Homebrew (the macOS package manager; needed to install ffmpeg) — its installer asks for your password"; n=$((n+1))
fi
if [ ${#SYS_PKGS[@]} -gt 0 ]; then
  case "$PKG" in
    apt) echo "  $n. System packages via apt (asks for your sudo password): ${SYS_PKGS[*]}";;
    dnf) echo "  $n. System packages via dnf (asks for your sudo password): ${SYS_PKGS[*]}";;
    pacman) echo "  $n. System packages via pacman (asks for your sudo password): ${SYS_PKGS[*]}";;
    brew) echo "  $n. System packages via Homebrew: ${SYS_PKGS[*]}";;
  esac
  n=$((n+1))
else
  echo "  ·  System packages: ffmpeg and Python $([ -n "$PY" ] && "$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])') are already present — nothing to install"
fi
echo "  $n. A private Python environment in $TOOL_DIR/.venv (nothing touches the system Python)"; n=$((n+1))
echo "  $n. Python packages into that environment, $PYSIZE download:"
echo "        numpy, scipy, psutil, faster-whisper (transcripts), mediapipe (gaze & pose), torch + transformers (topics),"
echo "        scikit-learn, pillow$([ "$GPU" = cuda ] && echo ', nvidia-cublas-cu12, nvidia-cudnn-cu12')"; n=$((n+1))
if [ "$MODELS" = 1 ]; then
  echo "  $n. Models into ~/.cache/huggingface: Whisper $WHISPER ($WSIZE) for transcripts, MiniLM (90 MB) for topics"; n=$((n+1))
else
  echo "  ·  Models: skipped (--no-models); they download on first use"
fi
echo "  $n. The 'multicam' command → ~/.local/bin/multicam$([ "$OS" = Linux ] && echo ', and a Multicam Cutter entry in your applications menu')"; n=$((n+1))
echo "  $n. A hardware check (which video encoders work here, GPU use, missing pieces)"
for x in ${NOTES[@]+"${NOTES[@]}"}; do echo "  note: $x"; done
echo
echo "  Nothing is sent anywhere; the only network traffic is downloading the packages and models above."
echo "  Full log: $LOG"
echo
[ "$DRY" = 1 ] && exit 0
if [ "$YES" != 1 ]; then
  read -r -p "Proceed with the installation? [Y/n] " ans
  case "${ans:-Y}" in y|Y|yes|YES) ;; *) echo "Nothing installed."; exit 0;; esac
fi
echo

# ------------------------------------------------------------------ do it
if [ "${NEED_BREW:-0}" = 1 ]; then
  bold "Homebrew"
  if [ "$YES" = 1 ]; then export NONINTERACTIVE=1; fi
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" >>"$LOG" 2>&1 || fail "Homebrew install"
  for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$p" ] && eval "$("$p" shellenv)"; done
  ok "Homebrew"
fi
if [ ${#SYS_PKGS[@]} -gt 0 ]; then
  bold "System packages"
  case "$PKG" in
    apt) run "apt update" $SUDO apt-get update -qq
         run "apt install ${SYS_PKGS[*]}" env DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${SYS_PKGS[@]}";;
    dnf) run "dnf install ${SYS_PKGS[*]}" $SUDO dnf install -y -q "${SYS_PKGS[@]}";;
    pacman) run "pacman -S ${SYS_PKGS[*]}" $SUDO pacman -S --noconfirm --needed "${SYS_PKGS[@]}";;
    brew) run "brew install ${SYS_PKGS[*]}" brew install -q "${SYS_PKGS[@]}";;
  esac
  hash -r
  if [ "$NEED_PY" = 1 ]; then
    PY=""; for c in python3.12 python3.11 python3.10 python3 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do have "$c" && py_ok "$c" && { PY="$c"; break; }; done
    [ -n "$PY" ] || fail "no Python 3.10–3.12 after installing packages (on Ubuntu 22.04: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv)"
  fi
  have ffmpeg || fail "ffmpeg still not found after installing packages"
fi

bold "Python environment"
[ -d .venv ] || run "create .venv with $("$PY" --version)" "$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
run "upgrade pip" python -m pip install --quiet --upgrade pip wheel

bold "Python packages"
if [ "$OS" = Linux ] && [ "$GPU" = cpu ]; then
  run "torch (CPU build)" python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
run "requirements.txt" python -m pip install --quiet -r requirements.txt
if [ "$GPU" = cuda ]; then
  run "CUDA runtime + cuDNN for Whisper" python -m pip install --quiet nvidia-cublas-cu12 nvidia-cudnn-cu12
fi

if [ "$MODELS" = 1 ]; then
  bold "Models"
  run "Whisper $WHISPER ($WSIZE)" python -c "from faster_whisper.utils import download_model; download_model('$WHISPER')"
  run "MiniLM (90 MB)" python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
fi

bold "Shortcuts"
mkdir -p "$HOME/.local/bin"; chmod +x "$TOOL_DIR/multicam"
run "~/.local/bin/multicam" ln -sf "$TOOL_DIR/multicam" "$HOME/.local/bin/multicam"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *)
  RC="$HOME/.bashrc"; [ "${SHELL##*/}" = zsh ] && RC="$HOME/.zshrc"
  grep -qs 'HOME/.local/bin' "$RC" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC"
  NOTES+=("~/.local/bin was added to your PATH in $RC — open a new terminal before typing 'multicam'.");;
esac
if [ "$OS" = Linux ]; then
  mkdir -p "$HOME/.local/share/applications"
  printf '[Desktop Entry]\nType=Application\nName=Multicam Cutter\nComment=Cut a multi-camera lecture recording\nExec=%s\nTerminal=true\nCategories=AudioVideo;Video;\n' "$TOOL_DIR/multicam" > "$HOME/.local/share/applications/multicam-cutter.desktop"
  ok "applications menu entry"
fi

bold "Hardware check"
python multicam.py doctor || true
echo
for x in ${NOTES[@]+"${NOTES[@]}"}; do echo "note: $x"; done
bold "Done. Start the tool with:   multicam \"/path/to/your lecture folder\""
echo "(the folder holds the camera files, the Zoom recording and its audio; the first start analyses and syncs them)"
