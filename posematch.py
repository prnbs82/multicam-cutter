"""Pose-matched cut points (after Berthouzoz, Li & Agrawala, SIGGRAPH 2012).

For a same-camera join, candidate frames are taken ONLY from the removed footage on both sides (silent parts),
scored by upper-body pose similarity (MediaPipe pose landmarks; image-difference fallback), and the join is moved
to the best-matching pair. Results are cached in _multicam/joinfix.json keyed by the join's removed region.
"""
import os, time
import numpy as np
from common import load_json, save_json, work_dir

PFPS = 25            # proxy frame rate
MAX_WIN = 0.5        # seconds searched on each side of the join
UPPER = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]   # nose, eyes, ears, shoulders, elbows, wrists, hips
MIN_GAIN = 0.90      # accept a new pair only if its motion score is < 90 % of the original pair's

_pose = None
def _get_pose():
    """Lazily create and return the shared MediaPipe Pose detector (static images, lightest model, GLOG quiet)."""
    global _pose
    if _pose is None:
        os.environ.setdefault('GLOG_minloglevel', '2')
        import mediapipe as mp
        _pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0, min_detection_confidence=0.5)
    return _pose


KEY_VERSION = 'k3'   # bump when the search rules change so cached fixes are recomputed
SIL_FRAC = 0.25      # a 10 ms hop is 'silent' when its RMS is below this fraction of the median speech RMS

def fix_key(j):
    """Cache key of a join in joinfix.json: left end, right start (3 decimals), left camera and KEY_VERSION."""
    return f"{j['lb']:.3f}-{j['ra']:.3f}-{j['camL']}-{KEY_VERSION}"


class Envelope:
    """10 ms RMS envelope of the master audio (from the 8 kHz sync wav); tells how far the audio is quiet around a time."""
    def __init__(self, wd):
        """Load the 8 kHz sync wav of the master audio and compute the 10 ms RMS envelope and the silence threshold;
        self.ok is False (and nothing else is set) when the wav does not exist yet."""
        from scipy.io import wavfile
        proj = load_json(os.path.join(wd, 'project.json'))
        path = os.path.join(wd, 'wav', proj['master_audio']['name'] + '.wav')
        self.ok = os.path.exists(path)
        if not self.ok:
            return
        sr, x = wavfile.read(path)
        x = x.astype(np.float32) / 32768
        self.hop_s = 0.01
        hop = int(sr * self.hop_s)
        n = len(x) // hop
        self.env = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
        voiced = self.env[self.env > 0.001]
        self.thr = SIL_FRAC * float(np.median(voiced)) if len(voiced) else 0.005
    def quiet_extent(self, t, direction, max_win=MAX_WIN, tol=2):
        """Seconds of quiet audio from t going backwards (-1) or forwards (+1); tolerates `tol`-1 loud hops."""
        if not self.ok:
            return 0.0
        i, k, bad = int(t / self.hop_s), 0, 0
        n = len(self.env)
        while k < int(max_win / self.hop_s):
            j = i + (k if direction > 0 else -k - 1)
            if j < 0 or j >= n:
                break
            bad = bad + 1 if self.env[j] >= self.thr else 0
            if bad >= tol:
                k -= tol - 1
                break
            k += 1
        return max(0, k) * self.hop_s


class FrameSource:
    """Random access to proxy frames (_multicam/proxies/<cam>.mp4) through OpenCV, one VideoCapture per camera, opened lazily."""
    def __init__(self, wd):
        """Remember the work dir; captures are opened per camera on first use."""
        import cv2
        self.cv2, self.wd, self.caps = cv2, wd, {}
    def frame(self, cam, t):
        """BGR frame of camera `cam` at time t (seconds on the proxy = master clock), or None when the seek/read fails."""
        cv2 = self.cv2
        cap = self.caps.get(cam)
        if cap is None:
            cap = self.caps[cam] = cv2.VideoCapture(os.path.join(self.wd, 'proxies', f'{cam}.mp4'))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, f = cap.read()
        return f if ok else None
    def close(self):
        """Release every open VideoCapture."""
        for c in self.caps.values():
            c.release()


def features(frame):
    """(pose landmark array [n,3] or None, small grayscale image)"""
    import cv2
    small = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)), (5, 5), 0).astype(np.float32)
    try:
        res = _get_pose().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm = res.pose_landmarks
    except Exception:
        lm = None
    pts = None
    if lm is not None:
        pts = np.array([[lm.landmark[i].x, lm.landmark[i].y, lm.landmark[i].visibility] for i in UPPER], dtype=np.float32)
    return pts, small


def motion_score(fa, fb):
    """Lower = more similar. Pose distance (normalised coords) when both have enough landmarks, else image difference."""
    pa, ia = fa
    pb, ib = fb
    if pa is not None and pb is not None:
        vis = (pa[:, 2] > 0.5) & (pb[:, 2] > 0.5)
        if vis.sum() >= 5:
            d = np.linalg.norm(pa[vis, :2] - pb[vis, :2], axis=1)
            return float(d.mean()), 'pose'
    return float(np.abs(ia - ib).mean() / 255.0), 'image'


MARGIN = 0.04        # never come closer than this to a spoken word

def silent_windows(j, words, env=None):
    """Search ranges around the join, restricted to silence. Silence is decided ACOUSTICALLY (RMS envelope; Whisper's word
    boundaries are contiguous and say nothing about pauses), and additionally guarded by the words so removed speech is
    never shown. Returns (left_back, left_fwd, right_back, right_fwd) in seconds."""
    lb, ra = j['lb'], j['ra']
    half = max(0.0, (ra - lb) / 2 - 0.02)
    inside = [w for w in words if w['e'] > lb and w['s'] < ra]
    left_fwd = min(MAX_WIN, half, (min(w['s'] for w in inside) - lb - MARGIN) if inside else half)      # into removed footage
    right_back = min(MAX_WIN, half, (ra - max(w['e'] for w in inside) - MARGIN) if inside else half)
    left_back = right_fwd = MAX_WIN                                                                      # into kept footage (acoustic gate below)
    if env is not None and env.ok:
        left_back = min(left_back, env.quiet_extent(lb, -1))
        right_fwd = min(right_fwd, env.quiet_extent(ra, +1))
        left_fwd = min(left_fwd, env.quiet_extent(lb, +1))
        right_back = min(right_back, env.quiet_extent(ra, -1))
    else:                                                                                                # no audio envelope: fall back to word gaps
        prev_end = max([w['e'] for w in words if w['e'] <= lb + 1e-6] or [lb])
        next_start = min([w['s'] for w in words if w['s'] >= ra - 1e-6] or [ra])
        left_back = min(MAX_WIN, max(0.0, lb - prev_end - MARGIN))
        right_fwd = min(MAX_WIN, max(0.0, next_start - ra - MARGIN))
    return max(0.0, left_back), max(0.0, left_fwd), max(0.0, right_back), max(0.0, right_fwd)


def refine_join(src, j, words, env=None):
    """Return a fix dict {lb2, ra2, score, base, method, gain} or None (keep the original join)."""
    lback, lfwd, rback, rfwd = silent_windows(j, words, env)
    if lback + lfwd < 1 / PFPS and rback + rfwd < 1 / PFPS:
        return None
    lt = sorted(set([j['lb'] + k / PFPS for k in range(int(lfwd * PFPS) + 1)] + [j['lb'] - k / PFPS for k in range(int(lback * PFPS) + 1)]), key=lambda t: abs(t - j['lb']))
    rt = sorted(set([j['ra'] - k / PFPS for k in range(int(rback * PFPS) + 1)] + [j['ra'] + k / PFPS for k in range(int(rfwd * PFPS) + 1)]), key=lambda t: abs(t - j['ra']))
    lf = [(t, features(f)) for t in lt if (f := src.frame(j['camL'], max(0.0, t - 1 / PFPS / 2))) is not None]
    rf = [(t, features(f)) for t in rt if (f := src.frame(j['camR'], t)) is not None]
    if not lf or not rf:
        return None
    base, method = motion_score(lf[0][1], rf[0][1])
    best = (base, lf[0][0], rf[0][0], method)
    for tl, fl in lf:
        for tr, fr in rf:
            sc, m = motion_score(fl, fr)
            if sc < best[0]:
                best = (sc, tl, tr, m)
    sc, tl, tr, m = best
    if sc >= base * MIN_GAIN or (tl == lf[0][0] and tr == rf[0][0]):
        return {'lb2': j['lb'], 'ra2': j['ra'], 'score': base, 'base': base, 'method': method, 'gain': 0.0, 'kept': True}
    return {'lb2': round(tl, 3), 'ra2': round(tr, 3), 'score': sc, 'base': base, 'method': m, 'gain': round(1 - sc / max(base, 1e-9), 3), 'kept': False}


def refine_all(lecture_dir, plan, words, status_cb=None, force=False):
    """Compute/refresh fixes for every non-cut join in `plan`; returns the full fixes dict (cached in joinfix.json)."""
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    path = os.path.join(wd, 'joinfix.json')
    fixes = load_json(path, {}) if not force else {}
    todo = [j for j in plan if j['type'] != 'cut' and fix_key(j) not in fixes]
    if not todo:
        return fixes
    src = FrameSource(wd)
    env = Envelope(wd)
    t0 = time.time()
    try:
        for n, j in enumerate(todo):
            fx = refine_join(src, j, words, env)
            fixes[fix_key(j)] = fx or {'lb2': j['lb'], 'ra2': j['ra'], 'kept': True, 'score': None, 'base': None, 'method': 'none', 'gain': 0.0}
            if (n + 1) % 5 == 0 or n + 1 == len(todo):
                save_json(path, fixes)
            if status_cb:
                status_cb(n + 1, len(todo), time.time() - t0)
    finally:
        src.close()
        save_json(path, fixes)
    return fixes


def run_posematch(lecture_dir, force=False):
    """CLI/job entry: refine every same-camera join of the whole (tightened) lecture; writes posematch/status.json."""
    from joins import migrate_tighten, join_plan
    from render import kept_intervals
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    def status(**kw):
        """Write posematch/status.json with the given fields plus a timestamp."""
        save_json(os.path.join(wd, 'posematch', 'status.json'), {**kw, 'time': time.time()})
    try:
        words = load_json(os.path.join(wd, 'words.json'), {'words': []})['words']
        cuts = load_json(os.path.join(wd, 'cuts.json'), {})
        layout = load_json(os.path.join(wd, 'layout.json'))
        import envelope as envmod
        u8 = envmod.load(wd)
        tdoc, _ = migrate_tighten(load_json(os.path.join(wd, 'tighten.json'), {}) or {}, words, u8)
        skips = cuts.get('skips', []) + [{'a': r['a'], 'b': r['b']} for r in tdoc['removals']]
        intervals = kept_intervals(0.0, float(layout['duration']), skips, min_keep=0.15)
        plan = join_plan(intervals, sorted(cuts.get('cuts', []), key=lambda c: c['t']), words, tdoc, layout, cuts.get('fallback') or layout['angles'][0]['id'], u8=u8)
        todo = [j for j in plan if j['type'] != 'cut']
        status(state='matching', progress=0, message=f'{len(todo)} same-camera joins to match')
        fixes = refine_all(ld, plan, words, force=force,
                           status_cb=lambda n, tot, el: status(state='matching', progress=n / tot, message=f'{n}/{tot} joins matched · {el / max(n, 1):.1f}s per join · ~{(tot - n) * el / max(n, 1) / 60:.0f} min left'))
        moved = [fixes[fix_key(j)] for j in todo if not fixes[fix_key(j)].get('kept')]
        gains = [f['gain'] for f in moved]
        msg = f'{len(moved)} of {len(todo)} joins moved to a better frame pair' + (f' (median motion −{np.median(gains) * 100:.0f}%)' if gains else '')
        status(state='done', progress=1, message=msg, moved=len(moved), total=len(todo))
        print(msg)
        return fixes
    except Exception as e:
        status(state='error', progress=0, message=str(e)[-1500:])
        raise


def motions_from_fixes(plan, fixes):
    """{(lb, ra, camL): pose motion across the join} for joins that have a pose-based measurement."""
    out = {}
    for j in plan:
        fx = (fixes or {}).get(fix_key(j))
        if fx and fx.get('method') == 'pose' and fx.get('base') is not None:
            out[(round(j['lb'], 3), round(j['ra'], 3), j['camL'])] = float(fx['base'])
    return out
