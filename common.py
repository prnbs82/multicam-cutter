"""Shared helpers: paths, ffprobe, json io."""
import json, os, subprocess

VIDEO_EXT = {'.mp4', '.mov', '.mkv', '.m4v', '.avi', '.webm', '.mts', '.m2ts'}
AUDIO_EXT = {'.m4a', '.wav', '.mp3', '.aac', '.flac', '.ogg'}
COLORS = ['#3fb950', '#f0883e', '#58a6ff', '#d2a8ff', '#f778ba', '#e3b341', '#79c0ff', '#ff7b72']
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))


def work_dir(lecture_dir):
    return os.path.join(lecture_dir, '_multicam')


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    """Atomic write; the temp name is unique per process/thread so concurrent writers never collide."""
    import threading
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def ffprobe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def probe_summary(path):
    j = ffprobe(path)
    fmt = j['format']
    v = next((s for s in j['streams'] if s['codec_type'] == 'video' and s.get('codec_name') not in ('mjpeg', 'png')), None)
    a = next((s for s in j['streams'] if s['codec_type'] == 'audio' and s.get('codec_name') not in (None, 'unknown')), None)
    info = {
        'name': os.path.basename(path),
        'duration': float(fmt.get('duration', 0)),
        'size': int(fmt.get('size', 0)),
        'mtime': os.path.getmtime(path),
        'creation_time': (fmt.get('tags') or {}).get('creation_time'),
        'has_video': v is not None,
        'has_audio': a is not None,
    }
    if v:
        num, den = v.get('r_frame_rate', '0/1').split('/')
        info.update(width=v['width'], height=v['height'], fps=(float(num) / float(den)) if float(den) else 0,
                    vcodec=v['codec_name'])
    return info


def run(cmd, **kw):
    """Run a command, raise with stderr on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr[-4000:]}")
    return r
