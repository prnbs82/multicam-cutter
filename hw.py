"""Hardware detection with clean fallbacks: which video encoder ffmpeg can really use here, whether Whisper can run on a
GPU, which torch device to use, and how big a Whisper model this machine can hold.

Results are cached in ~/.config/multicam/hw.json (re-probed when the ffmpeg build, Python or GPU signature changes, or
with `multicam.py doctor --refresh`). A user choice in that file ("encoder": "libx264") or the environment variable
MULTICAM_ENCODER always wins over the probe.
"""
import json, os, platform, shutil, subprocess, sys, tempfile

CONFIG_DIR = os.environ.get('MULTICAM_CONFIG') or os.path.join(os.path.expanduser('~'), '.config', 'multicam')

# (ffmpeg encoder, human name, platforms it can exist on)
ENCODERS = [
    ('h264_nvenc', 'NVIDIA NVENC', ('Linux', 'Windows')),
    ('h264_qsv', 'Intel Quick Sync', ('Linux', 'Windows')),
    ('h264_vaapi', 'VA-API (Intel/AMD)', ('Linux',)),
    ('h264_videotoolbox', 'Apple VideoToolbox', ('Darwin',)),
    ('libx264', 'CPU (libx264)', ('Linux', 'Darwin', 'Windows')),
]
NAMES = {k: n for k, n, _ in ENCODERS}


# ------------------------------------------------------------------ config files
def config_path(name):
    """Absolute path of `name` inside CONFIG_DIR (~/.config/multicam or $MULTICAM_CONFIG)."""
    return os.path.join(CONFIG_DIR, name)


def load_config(name, default=None):
    """Read a JSON file from CONFIG_DIR; returns `default` when it is missing or not valid JSON."""
    try:
        with open(config_path(name), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_config(name, data):
    """Write `data` as indented JSON to CONFIG_DIR/name atomically (pid-named temp file + os.replace); creates CONFIG_DIR."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = config_path(name) + f'.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, config_path(name))


# ------------------------------------------------------------------ encoder argument sets
def _vaapi_device():
    """First /dev/dri/renderD* node (for VA-API), or None when there is no render node."""
    for d in sorted(os.listdir('/dev/dri')) if os.path.isdir('/dev/dri') else []:
        if d.startswith('renderD'):
            return '/dev/dri/' + d
    return None


def encoder_args(kind, quality='final', fps=30):
    """ffmpeg output options for `kind`. quality: 'final' (visually lossless-ish) or 'proxy' (fast, small).
    Returns (args, vf_tail): vf_tail must be appended to the END of the video filter chain (VA-API needs an upload)."""
    gop = ['-g', str(fps * 2)]
    if kind == 'libx264':
        if quality == 'proxy':
            return ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', *gop, '-pix_fmt', 'yuv420p'], ''
        return ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-x264-params', f'keyint={fps * 2}:min-keyint={fps}', '-video_track_timescale', '90000'], ''
    if kind == 'h264_nvenc':
        q = '28' if quality == 'proxy' else '19'
        return ['-c:v', 'h264_nvenc', '-preset', 'p4' if quality == 'proxy' else 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', q,
                '-b:v', '0', '-bf', '2', *gop, '-pix_fmt', 'yuv420p', '-video_track_timescale', '90000'], ''
    if kind == 'h264_qsv':
        q = '28' if quality == 'proxy' else '19'
        return ['-c:v', 'h264_qsv', '-preset', 'veryfast' if quality == 'proxy' else 'medium', '-global_quality', q,
                '-look_ahead', '0', *gop, '-pix_fmt', 'nv12', '-video_track_timescale', '90000'], ''
    if kind == 'h264_vaapi':
        dev = _vaapi_device() or '/dev/dri/renderD128'
        q = '28' if quality == 'proxy' else '20'
        return ['-vaapi_device', dev, '-c:v', 'h264_vaapi', '-rc_mode', 'CQP', '-qp', q, *gop,
                '-video_track_timescale', '90000'], ',format=nv12,hwupload'
    if kind == 'h264_videotoolbox':
        q = '50' if quality == 'proxy' else '65'
        return ['-c:v', 'h264_videotoolbox', '-q:v', q, '-allow_sw', '1', '-realtime', '0', *gop, '-pix_fmt', 'yuv420p',
                '-video_track_timescale', '90000'], ''
    raise ValueError(f'unknown encoder {kind!r}')


def prove_encoder(kind, seconds=0.5, size='1920x1080', fps=30):
    """Encode a few synthetic frames with `kind` and check they come out. Returns (ok, detail)."""
    args, tail = encoder_args(kind, 'final', fps)
    n = int(seconds * fps)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'probe.mp4')
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', f'testsrc2=size={size}:rate={fps}', '-frames:v', str(n),
               '-vf', 'format=yuv420p' + tail, *args, out]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, str(e)
        if r.returncode != 0:
            last = (r.stderr.strip().splitlines() or ['failed'])[-1]
            if 'Error while opening encoder' in last or 'Cannot load' in last or 'No such device' in last or 'not found' in last:
                last = 'no usable device for this encoder'
            return False, last[:120]
        try:
            pr = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                                 'stream=nb_read_frames,codec_name', '-of', 'json', out], capture_output=True, text=True, timeout=60)
            st = json.loads(pr.stdout)['streams'][0]
            frames = int(st.get('nb_read_frames') or 0)
        except Exception as e:
            return False, f'probe read failed: {e}'
        if frames != n:
            return False, f'produced {frames} of {n} frames'
        return True, f'{frames} frames OK'


# ------------------------------------------------------------------ system facts
def ffmpeg_version():
    """Version token of the installed ffmpeg (third word of the first `ffmpeg -version` line), or None when it cannot be run."""
    try:
        return subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10).stdout.splitlines()[0].split()[2]
    except Exception:
        return None


def ffmpeg_encoders():
    """Set of video encoder names listed by `ffmpeg -encoders` (lines flagged ' V'); empty set when ffmpeg fails."""
    try:
        out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return set()
    return {line.split()[1] for line in out.splitlines() if line.startswith(' V') and len(line.split()) > 1}


_FILTERS = None


def ffmpeg_filters():
    """Names of the filters this ffmpeg build has (cached)."""
    global _FILTERS
    if _FILTERS is None:
        try:
            out = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True, timeout=10).stdout
            _FILTERS = {line.split()[1] for line in out.splitlines() if line.startswith(' ') and len(line.split()) > 2}
        except Exception:
            _FILTERS = set()
    return _FILTERS


REQUIRED_FILTERS = {'xfade': 'dissolves and holds', 'tpad': 'frame-exact pieces', 'zoompan': 'image zoom', 'overlay': 'picture-in-picture'}
OPTIONAL_FILTERS = {'drawtext': 'image credits burned into the picture (needs an ffmpeg built with libfreetype; credits are always in the .txt)'}


def nvidia_gpus():
    """[{name, vram_gb}] per GPU reported by nvidia-smi; [] when nvidia-smi is absent or fails."""
    if not shutil.which('nvidia-smi'):
        return []
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and parts[1].isdigit():
            gpus.append({'name': parts[0], 'vram_gb': round(int(parts[1]) / 1024, 1)})
    return gpus


def gpu_names():
    """Best-effort list of graphics devices (name only) for the report."""
    names = [g['name'] for g in nvidia_gpus()]
    if names:
        return names
    sysname = platform.system()
    if sysname == 'Linux' and shutil.which('lspci'):
        try:
            out = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10).stdout
            names = [l.split(': ', 1)[1] for l in out.splitlines() if any(k in l for k in ('VGA', '3D controller', 'Display controller')) and ': ' in l]
        except Exception:
            pass
    elif sysname == 'Darwin':
        names = ['Apple GPU (Metal)' if platform.machine() == 'arm64' else 'Mac GPU']
    return names[:3]


def cuda_devices():
    """CUDA devices usable by CTranslate2 (faster-whisper) — 0 on AMD/Apple/CPU-only boxes."""
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def torch_device():
    """'cuda', 'mps' or 'cpu' for torch (MiniLM embeddings); 'cpu' when torch is not installed."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def physical_cores():
    """Number of physical CPU cores via psutil, falling back to os.cpu_count() and finally 4."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except Exception:
        return os.cpu_count() or 4


def ram_total_gb():
    """Total RAM in GiB via psutil; assumes 8.0 when psutil is unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        return 8.0


def whisper_choice(model=None):
    """(device, compute_type, model) for faster-whisper. GPU when CUDA exists; the model size follows RAM/VRAM unless given."""
    gpus = nvidia_gpus()
    if cuda_devices() > 0:
        vram = max([g['vram_gb'] for g in gpus] or [4.0])
        auto = 'large-v3' if vram >= 6 else ('medium' if vram >= 3.5 else 'small')
        return 'cuda', 'float16', model or auto
    ram = ram_total_gb()
    auto = 'large-v3' if ram >= 12 else ('medium' if ram >= 6 else 'small')
    return 'cpu', 'int8', model or auto


# ------------------------------------------------------------------ the probe (cached)
def _signature():
    """Machine fingerprint that invalidates the hw.json probe cache: policy version, ffmpeg/Python versions, OS, GPUs and the
    encoders this ffmpeg lists."""
    return {'policy': 2, 'ffmpeg': ffmpeg_version(), 'python': sys.version.split()[0], 'system': platform.system(), 'machine': platform.machine(),
            'gpus': gpu_names(), 'encoders_listed': sorted(e for e in ffmpeg_encoders() if e in NAMES)}


def probe(refresh=False):
    """Prove which encoders work here (in preference order) and cache the outcome."""
    sig = _signature()
    cache = load_config('hw.json', {}) or {}
    if not refresh and cache.get('signature') == sig and cache.get('proofs'):
        return cache
    listed = set(sig['encoders_listed'])
    proofs = {}
    for kind, _name, platforms in ENCODERS:
        if kind not in listed or sig['system'] not in platforms:
            proofs[kind] = {'ok': False, 'detail': 'not in this ffmpeg build' if kind not in listed else 'not on this OS'}
            continue
        if kind == 'h264_vaapi' and not _vaapi_device():
            proofs[kind] = {'ok': False, 'detail': 'no /dev/dri render node'}
            continue
        ok, detail = prove_encoder(kind)
        proofs[kind] = {'ok': ok, 'detail': detail}
    # auto policy: NVENC / Quick Sync / VideoToolbox when proven (mature, near-x264 quality at these settings);
    # VA-API only when the CPU is weak (< 4 physical cores) — on a desktop CPU x264 CRF 18 looks better; libx264 otherwise
    best = 'libx264'
    for k in ('h264_nvenc', 'h264_qsv', 'h264_videotoolbox'):
        if proofs.get(k, {}).get('ok'):
            best = k; break
    else:
        if proofs.get('h264_vaapi', {}).get('ok') and physical_cores() < 4:
            best = 'h264_vaapi'
    cache.update({'signature': sig, 'proofs': proofs, 'best': best})
    cache.setdefault('encoder', 'auto')
    try:
        save_config('hw.json', cache)
    except OSError:
        pass
    return cache


def encoder(quality='final', fps=30):
    """The encoder to use now: MULTICAM_ENCODER env > hw.json "encoder" > best proven. Returns (kind, args, vf_tail)."""
    forced = os.environ.get('MULTICAM_ENCODER')
    if forced and forced in NAMES:
        kind = forced
    else:
        cache = probe()
        choice = cache.get('encoder', 'auto')
        kind = choice if choice in NAMES and choice != 'auto' and cache['proofs'].get(choice, {}).get('ok', choice == 'libx264') else cache['best']
    args, tail = encoder_args(kind, quality, fps)
    return kind, args, tail


def set_encoder(choice):
    """Store the user's encoder choice in hw.json ('auto' unless `choice` is a known kind); returns the updated cache."""
    cache = probe()
    cache['encoder'] = choice if choice in NAMES else 'auto'
    save_config('hw.json', cache)
    return cache


# ------------------------------------------------------------------ report / doctor
def report(refresh=False):
    """Dict of system, CPU/RAM, GPU, encoder (used/choice/best/proofs) and Whisper facts for `doctor` and the Advanced panel.
    Runs the (cached) encoder probe; refresh=True re-proves the encoders."""
    cache = probe(refresh)
    dev, ctype, model = whisper_choice()
    kind = encoder()[0]
    return {
        'system': f"{platform.system()} {platform.release()} ({platform.machine()})",
        'python': sys.version.split()[0], 'ffmpeg': cache['signature']['ffmpeg'],
        'cpu_cores_physical': physical_cores(), 'cpu_cores_logical': os.cpu_count() or 0, 'ram_total_gb': round(ram_total_gb(), 1),
        'gpus': gpu_names(), 'cuda_devices': cuda_devices(), 'torch_device': torch_device(),
        'encoder': kind, 'encoder_name': NAMES[kind], 'encoder_choice': cache.get('encoder', 'auto'), 'encoder_best': cache['best'],
        'encoder_proofs': cache['proofs'],
        'whisper': {'device': dev, 'compute_type': ctype, 'model': model},
    }


def missing_python_modules():
    """[{module, package, needed_for}] for every optional Python dependency that fails to import."""
    out = []
    for mod, pkg, what in (('numpy', 'numpy', 'everything'), ('scipy', 'scipy', 'audio sync'), ('psutil', 'psutil', 'capacity report'),
                           ('faster_whisper', 'faster-whisper', 'transcripts'), ('mediapipe', 'mediapipe', 'gaze proposals + pose matching'),
                           ('cv2', 'opencv (comes with mediapipe)', 'pose matching + morph'), ('torch', 'torch', 'topics'),
                           ('transformers', 'transformers', 'topics')):
        try:
            __import__(mod)
        except Exception:
            out.append({'module': mod, 'package': pkg, 'needed_for': what})
    return out


def doctor(lecture_dir=None, refresh=False):
    """Print a human-readable report; returns the number of hard problems found."""
    problems = 0
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        hint = 'brew install ffmpeg' if platform.system() == 'Darwin' else 'sudo apt install ffmpeg   (or your distro equivalent)'
        print(f'✗ ffmpeg/ffprobe not found — install it:  {hint}')
        return 1
    r = report(refresh)
    print(f"System   : {r['system']}, Python {r['python']}, ffmpeg {r['ffmpeg']}")
    print(f"CPU/RAM  : {r['cpu_cores_physical']} cores ({r['cpu_cores_logical']} threads), {r['ram_total_gb']} GB RAM")
    print(f"GPU      : {', '.join(r['gpus']) or 'none detected'}" + (f"  (CUDA devices: {r['cuda_devices']})" if r['cuda_devices'] else ''))
    print('Encoders :')
    for kind, name, _p in ENCODERS:
        pr = r['encoder_proofs'].get(kind, {})
        mark = '✓' if pr.get('ok') else '·'
        use = '  ← used for exports' if kind == r['encoder'] else ('  (usable; pick it under Advanced if the CPU export is too slow)' if pr.get('ok') else '')
        print(f"  {mark} {name:22s} {pr.get('detail', '')}{use}")
    if r['encoder_choice'] not in ('auto', r['encoder']):
        print(f"  (your choice {r['encoder_choice']!r} is not usable here — falling back to {r['encoder']})")
    print(f"Whisper  : {r['whisper']['model']} on {r['whisper']['device']} ({r['whisper']['compute_type']})")
    print(f"Topics   : MiniLM on {r['torch_device']}")
    filters = ffmpeg_filters()
    for f, why in REQUIRED_FILTERS.items():
        if filters and f not in filters:
            print(f"✗ ffmpeg lacks the '{f}' filter — needed for {why}; install a full ffmpeg build")
            problems += 1
    for f, why in OPTIONAL_FILTERS.items():
        if filters and f not in filters:
            print(f"· ffmpeg lacks the '{f}' filter — {why}")
    miss = missing_python_modules()
    for m in miss:
        print(f"✗ python module {m['module']} missing (pip install {m['package']}) — needed for {m['needed_for']}")
    problems += len(miss)
    try:
        from llm import provider_status
        ps = provider_status()
        print(f"AI       : {ps['detail']}")
    except Exception as e:
        print(f'AI       : (could not check: {e})')
    if lecture_dir:
        from capacity import report as cap
        c = cap(lecture_dir)
        print(f"Lecture  : {c['ram_available_gb']} GB RAM free, {c['disk_free_gb']} GB disk free, {c['recommended_workers']} parallel encoders, "
              f"~{c['est_render_speed_x']}× realtime export, clips up to ~{c['max_clip_minutes_by_disk']} min fit on disk")
        for n in c['notes']:
            print('  note:', n)
    print('OK' if not problems else f'{problems} problem(s) to fix')
    return problems


if __name__ == '__main__':
    sys.exit(1 if doctor(sys.argv[1] if len(sys.argv) > 1 else None, refresh='--refresh' in sys.argv) else 0)
