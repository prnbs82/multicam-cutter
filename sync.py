"""Audio cross-correlation sync of every video file against the master audio."""
import os, numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, butter, sosfiltfilt
from common import run, load_json, save_json, work_dir

SR = 8000
CHUNK = 60.0        # seconds of audio per probe chunk
POSITIONS = (0.1, 0.5, 0.9)


def extract_wav(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src) and os.path.getsize(dst) > 1000:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    run(['ffmpeg', '-y', '-v', 'error', '-i', src, '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SR),
         '-c:a', 'pcm_s16le', dst])


_SOS = butter(4, [300, 3400], btype='band', fs=SR, output='sos')


def load(path):
    sr, x = wavfile.read(path)
    assert sr == SR
    x = x.astype(np.float32) / 32768.0
    x = sosfiltfilt(_SOS, x).astype(np.float32)
    return x


def norm(x):
    x = x - x.mean()
    s = x.std()
    return x / s if s > 0 else x


def best_lag(master, chunk):
    """Return (lag_samples, peak/second-peak ratio). lag = position of chunk start in master."""
    c = correlate(master, norm(chunk), mode='full', method='fft')
    k = int(np.argmax(c))
    peak = c[k]
    lo, hi = max(0, k - SR), min(len(c), k + SR)
    c[lo:hi] = 0
    second = float(np.max(c))
    return k - (len(chunk) - 1), float(peak / (abs(second) + 1e-9))


def pick_chunks(x):
    n = len(x)
    L = int(CHUNK * SR)
    if n <= L:
        return [(0, x)]
    rms_all = np.sqrt(np.mean(x ** 2)) + 1e-9
    out = []
    for p in POSITIONS:
        s = int(p * n) - L // 2
        s = min(max(0, s), n - L)
        for _ in range(6):  # avoid near-silent chunks
            seg = x[s:s + L]
            if np.sqrt(np.mean(seg ** 2)) > 0.2 * rms_all or s + L + 30 * SR > n:
                break
            s = min(s + 30 * SR, n - L)
        out.append((s, x[s:s + L]))
    return out


def sync_project(lecture_dir, force=False):
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    proj = load_json(os.path.join(wd, 'project.json'))
    spath = os.path.join(wd, 'sync.json')
    sync = load_json(spath, {'files': {}}) if not force else {'files': {}}
    wavd = os.path.join(wd, 'wav')

    master_name = proj['master_audio']['name']
    print(f'extracting master wav: {master_name}')
    extract_wav(os.path.join(ld, master_name), os.path.join(wavd, master_name + '.wav'))
    master = norm(load(os.path.join(wavd, master_name + '.wav')))
    sync['master'] = {'name': master_name, 'duration': len(master) / SR}

    for a in proj['angles']:
        for f in a['files']:
            name = f['name']
            prev = sync['files'].get(name)
            if prev and prev.get('mtime') == f['mtime'] and not force:
                print(f'  {name}: cached offset {prev["offset"]:.3f}s')
                continue
            if name == master_name:
                sync['files'][name] = {'offset': 0.0, 'drift_ppm': 0.0, 'confidence': 99.0, 'chunks': [], 'mtime': f['mtime']}
                continue
            print(f'  {name}: extracting wav ...', flush=True)
            wpath = os.path.join(wavd, name + '.wav')
            extract_wav(os.path.join(ld, name), wpath)
            x = load(wpath)
            res = []
            for s, chunk in pick_chunks(x):
                lag, conf = best_lag(master, chunk)
                off = lag / SR - s / SR
                res.append({'chunk_start': s / SR, 'offset': off, 'confidence': conf})
                print(f'     chunk@{s / SR:7.1f}s -> offset {off:9.3f}s  conf {conf:5.2f}', flush=True)
            good = [r for r in res if r['confidence'] > 1.3] or res
            offs = np.array([r['offset'] for r in good])
            starts = np.array([r['chunk_start'] for r in good])
            offset = float(np.median(offs))
            drift = 0.0
            if len(good) >= 2 and starts.max() > starts.min():
                slope = np.polyfit(starts, offs, 1)[0]
                drift = float(slope * 1e6)
            spread = float(offs.max() - offs.min())
            conf = float(min(r['confidence'] for r in good))
            sync['files'][name] = {'offset': offset, 'drift_ppm': drift, 'spread': spread,
                                   'confidence': conf, 'chunks': res, 'mtime': f['mtime']}
            flag = '' if conf > 2 and spread < 0.1 else '   <-- CHECK (low confidence or inconsistent)'
            print(f'  {name}: offset {offset:.3f}s drift {drift:+.0f}ppm spread {spread * 1000:.0f}ms conf {conf:.2f}{flag}')
            save_json(spath, sync)
    save_json(spath, sync)
    print(f'wrote {spath}')
    return sync
