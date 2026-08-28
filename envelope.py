"""Quantised 10 ms loudness envelope of the master audio, shared by the renderer (Python) and the browser (JS twin).

u8 value = clip(round(2 * dBFS + 240), 0, 255)  ->  0.5 dB steps from -120 dBFS. Both sides snap removal edges to the
quietest hop in a window around the transcript boundary, so cuts never land inside voiced audio.
"""
import os
import numpy as np
from common import load_json, work_dir

HOP = 0.01
SNAP_BACK, SNAP_FWD = 0.20, 0.10     # left edge searches [a-0.20, a+0.10]; right edge mirrors it [b-0.10, b+0.20]


def build(wd):
    path = os.path.join(wd, 'envelope.u8')
    proj = load_json(os.path.join(wd, 'project.json'))
    wav8 = os.path.join(wd, 'wav', proj['master_audio']['name'] + '.wav')
    if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(wav8):
        return path
    from scipy.io import wavfile
    sr, x = wavfile.read(wav8)
    x = x.astype(np.float32) / 32768
    hop = int(sr * HOP)
    n = len(x) // hop
    env = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
    db = 20 * np.log10(env + 1e-6)
    u8 = np.clip(np.round(2 * db + 240), 0, 255).astype(np.uint8)
    with open(path, 'wb') as f:
        f.write(u8.tobytes())
    return path


def load(wd):
    path = build(wd) if os.path.exists(os.path.join(wd, 'wav')) else None
    if not path or not os.path.exists(path):
        return None
    return np.frombuffer(open(path, 'rb').read(), dtype=np.uint8)


def snap(u8, t, side):
    """Quietest hop near t. side='a' (start of a removal): window [t-0.20, t+0.10]; side='b' (end): [t-0.10, t+0.20].
    Ties -> closest to t. Returns the snapped time (hop start)."""
    if u8 is None or not len(u8):
        return t
    back, fwd = (SNAP_BACK, SNAP_FWD) if side == 'a' else (SNAP_FWD, SNAP_BACK)
    i0, i1 = max(0, int((t - back) / HOP)), min(len(u8) - 1, int((t + fwd) / HOP))
    if i1 <= i0:
        return t
    seg = u8[i0:i1 + 1].astype(np.int32)
    ic = int(t / HOP) - i0
    # score = loudness + tiny distance penalty (so equal loudness picks the nearest hop); integer arithmetic for JS parity
    score = seg * 1000 + np.abs(np.arange(len(seg)) - ic)
    k = int(np.argmin(score))
    return round((i0 + k) * HOP, 3)


START_BIAS, END_BIAS = -0.06, +0.09   # measured on this recording: Whisper word starts are ~60 ms late, ends ~90 ms early
RADIUS, PENALTY = 0.12, 4              # search +-120 ms around the expected boundary; 4 half-dB per 10 ms of distance


def snap_boundary(u8, expected, radius=RADIUS, penalty=PENALTY):
    """Quietest hop near the EXPECTED boundary, with a distance penalty so a plosive closure 100 ms away inside a word
    cannot beat the real gap next to the boundary. Integer arithmetic (JS twin must match)."""
    if u8 is None or not len(u8):
        return round(expected, 3)
    i0, i1 = max(0, int((expected - radius) / HOP)), min(len(u8) - 1, int((expected + radius) / HOP))
    if i1 <= i0:
        return round(expected, 3)
    ic = int(expected / HOP)
    seg = u8[i0:i1 + 1].astype(np.int32)
    score = seg + penalty * np.abs(np.arange(i0, i1 + 1) - ic)
    k = int(np.argmin(score))
    return round((i0 + k) * HOP, 3)


def word_edges(words, a, b):
    """Expected acoustic boundaries of a removal that covers the words whose midpoints lie in [a,b]."""
    inside = [i for i, w in enumerate(words) if a <= (w['s'] + w['e']) / 2 < b]
    if not inside:
        return None
    i0, i1 = inside[0], inside[-1]
    w0, w1 = words[i0], words[i1]
    prev = words[i0 - 1] if i0 > 0 else None
    nxt = words[i1 + 1] if i1 + 1 < len(words) else None
    ea = (prev['e'] + END_BIAS + w0['s'] + START_BIAS) / 2 if prev and w0['s'] - prev['e'] < 0.3 else w0['s'] + START_BIAS - 0.03
    eb = (w1['e'] + END_BIAS + nxt['s'] + START_BIAS) / 2 if nxt and nxt['s'] - w1['e'] < 0.3 else w1['e'] + END_BIAS + 0.03
    return ea, eb, i0, i1


def snap_removals(removals, u8, words=None):
    """Snap the edges of word/filler removals to the acoustic boundary nearest the bias-corrected word times
    (pause removals already sit in silence and keep their exact targets)."""
    out = []
    for r in removals:
        if r.get('kind') in ('words', 'filler', 'pause-fixed') and words:
            we = word_edges(words, r['a'], r['b'])
            if we:
                ea, eb, i0, i1 = we
                a, b = snap_boundary(u8, ea), snap_boundary(u8, eb)
                b = max(b, round(words[i1]['e'] + 0.02, 3))     # never leave the tail of the last removed word behind
                a = min(a, round(words[i0]['s'] - 0.02, 3))     # never keep the onset of the first removed word
                if b - a >= 0.05:
                    r = {**r, 'a': a, 'b': b}
        elif r.get('kind') in ('words', 'filler', 'pause-fixed'):
            a, b = snap(u8, r['a'], 'a'), snap(u8, r['b'], 'b')
            if b - a >= 0.05:
                r = {**r, 'a': a, 'b': b}
        out.append(r)
    return out


# ------------------------------------------------------------------ v3: natural cut — keep the real pauses around a removal
KEEP_MAX = 0.35          # seconds of natural silence kept on each side of a cut
_THR = {}

def speech_threshold(u8):
    """u8 level (half-dB units) below which a hop counts as quiet: 12 dB under the median of voiced hops."""
    key = (len(u8), int(u8[:4096].sum()), int(u8[-4096:].sum()), int(u8[len(u8) // 2:len(u8) // 2 + 4096].sum()))   # content fingerprint (ids get reused)
    if key not in _THR:
        med = float(np.median(u8)) if len(u8) else 150.0
        loud = u8[u8 > med]                        # the louder half = speech; robust whether speech or silence dominates the file
        p95 = float(np.percentile(loud, 95)) if len(loud) else 200.0
        _THR[key] = p95 - 30                       # 30 half-dB = 15 dB below the typical speech peaks (~ -35 dBFS on this recording)
    return _THR[key]


def quiet_extent(u8, t, direction, max_win=0.6, tol=2):
    """Seconds of quiet audio from t going forward (+1) or backward (-1), tolerating tol-1 loud hops."""
    if u8 is None or not len(u8):
        return 0.0
    thr = speech_threshold(u8)
    i, k, bad, n = int(t / HOP), 0, 0, len(u8)
    while k < int(max_win / HOP):
        j = i + (k if direction > 0 else -k - 1)
        if j < 0 or j >= n:
            break
        bad = bad + 1 if u8[j] >= thr else 0
        if bad >= tol:
            k -= tol - 1
            break
        k += 1
    return max(0, k) * HOP


def word_edges_v3(words, a, b, u8):
    """Edges for a removal covering the words whose midpoints lie in [a,b]: keep up to KEEP_MAX of the natural silence after
    the previous kept word and before the next kept word, cut inside silence, never inside a removed word.
    Returns (a, b, kept_before, kept_after) or None when no words are inside."""
    inside = [i for i, w in enumerate(words) if a <= (w['s'] + w['e']) / 2 < b]
    if not inside:
        return None
    i0, i1 = inside[0], inside[-1]
    w0, w1 = words[i0], words[i1]
    prev = words[i0 - 1] if i0 > 0 else None
    nxt = words[i1 + 1] if i1 + 1 < len(words) else None
    w0_real, w1_real = w0['s'] + START_BIAS, w1['e'] + END_BIAS
    if prev:
        pe = prev['e'] + END_BIAS
        if w0_real <= pe + 0.02:                    # contiguous speech: the biases overlap -> cut at the midpoint, nothing to keep
            na, kb = (pe + w0_real) / 2, 0.0
        else:
            kb = min(KEEP_MAX, quiet_extent(u8, pe, +1, max_win=w0_real - pe))
            na = min(pe + kb, w0_real - 0.01)
            na = snap_boundary(u8, na, radius=0.05, penalty=6)
            na = max(pe - 0.02, min(na, w0_real - 0.01))
            kb = max(0.0, na - pe)
    else:
        na, kb = round(max(0.0, w0_real - 0.05), 3), 0.0
    if nxt:
        ns = nxt['s'] + START_BIAS
        if ns <= w1_real + 0.02:                    # contiguous speech
            nb, ka = (w1_real + ns) / 2, 0.0
        else:
            ka = min(KEEP_MAX, quiet_extent(u8, ns, -1, max_win=ns - w1_real))
            nb = max(ns - ka, w1_real + 0.01)
            nb = snap_boundary(u8, nb, radius=0.05, penalty=6)
            nb = min(ns + 0.02, max(nb, w1_real + 0.01))
            ka = max(0.0, ns - nb)
    else:
        nb, ka = round(w1_real + 0.05, 3), 0.0
    return round(na, 3), round(nb, 3), round(kb, 3), round(ka, 3)


def snap_removals_v3(removals, u8, words):
    out = []
    for r in removals:
        if r.get('kind') in ('words', 'filler', 'pause-fixed') and words and u8 is not None:
            we = word_edges_v3(words, r['a'], r['b'], u8)
            if we and we[1] - we[0] >= 0.05:
                r = {**r, 'a': we[0], 'b': we[1], 'kept': [we[2], we[3]]}
        out.append(r)
    return out
