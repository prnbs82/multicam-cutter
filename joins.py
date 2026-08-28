"""Seamless joins: settings/migration of tighten.json (v2), derived pause shortening, join plan, room tone.

The JavaScript in web/index.html implements the same rules (migrateTighten / computeJoins); keep them in sync.
"""
import os
from common import load_json, run, work_dir

FPS = 30
DEFAULTS = {'pauseThreshold': 0.7, 'pauseInSentence': 0.30, 'pauseSentenceEnd': 0.55, 'dissolveFrames': 6, 'holdMs': 120,
            'fadeIn': 0.3, 'tailOut': 0.8,   # export edges: fade in from black; hold the last frame + room tone and fade out
            'startLead': 0.3, 'endHold': 1.0}  # clips: real footage kept before the first / after the last word (while the audio is quiet)
FADE = 0.008   # audio fade at every hard boundary (s)
MOTION_MORPH = 0.045   # measured pose motion across a same-camera join at/above which a morph replaces dissolve/hold
MOTION_MAX = 0.10      # above this the speaker moved too far for any same-camera trick: switch camera at the join (automatic)
PAUSE_MIN, PAUSE_TARGET = 0.30, 0.45   # natural cut: if the kept silence around a word cut is shorter than PAUSE_MIN, insert up to PAUSE_TARGET


# ------------------------------------------------------------------ pauses
def settings_of(doc):
    """Join/pause settings: DEFAULTS overridden by every key of the tighten doc that is not None."""
    s = dict(DEFAULTS)
    for k in DEFAULTS:
        if doc.get(k) is not None:
            s[k] = doc[k]
    return s


def pause_target(words, i, s):
    """Target pause length (s) after word i: pauseSentenceEnd when the word ends with . ? or !, else pauseInSentence."""
    return s['pauseSentenceEnd'] if words[i]['t'].rstrip()[-1:] in '.?!' else s['pauseInSentence']


def pause_removal(words, i, s):
    """Removal that shortens pause i (gap after word i) to its target, accounting for the dissolve that is added back."""
    if i < 0 or i + 1 >= len(words):
        return None
    w0, w1 = words[i], words[i + 1]
    g = w1['s'] - w0['e']
    T = pause_target(words, i, s)
    d = s['dissolveFrames'] / FPS
    if g <= T + 0.1:
        return None
    keep = (T - d) / 2 if g >= T + d else T - g / 2          # audible gap = keep + min(d, removed/2) + keep = T
    keep = max(0.04, keep)
    a, b = w0['e'] + keep, w1['s'] - keep
    if b - a < 0.05:
        return None
    return {'kind': 'pause', 'i': i, 'a': round(a, 3), 'b': round(b, 3), 'label': f'pause {g:.1f}s → {T:.2f}s'}


def find_pause_index(words, a, b, thr):
    """Index i of the pause [w_i.e, w_{i+1}.s] that contains [a,b] and is longer than thr, else None."""
    lo, hi, r = 0, len(words) - 1, -1
    while lo <= hi:                       # last word with e <= a (+tolerance)
        m = (lo + hi) >> 1
        if words[m]['e'] <= a + 0.002:
            r = m; lo = m + 1
        else:
            hi = m - 1
    for i in (r, r - 1, r + 1):
        if 0 <= i < len(words) - 1 and words[i]['e'] <= a + 0.002 and words[i + 1]['s'] >= b - 0.002 and words[i + 1]['s'] - words[i]['e'] > thr:
            return i
    return None


def migrate_tighten(doc, words, u8=None):
    """v1/v2 tighten doc -> v2 view. Idempotent. Never loses an edit (unmatched pause removals become 'pause-fixed').
    If `u8` (loudness envelope) is given and the doc's word removals were not yet acoustically snapped, snap them."""
    doc = doc or {}
    s = settings_of(doc)
    out = {**s, 'version': 2, 'corrections': dict(doc.get('corrections') or {}), 'joinOverrides': dict(doc.get('joinOverrides') or {}),
           'removals': [], 'edgeSnap': (doc.get('edgeSnap') if doc.get('edgeSnap') in (2, 3) else (1 if doc.get('edgeSnap') else 0))}
    if doc.get('fillers'):
        out['fillers'] = list(doc['fillers'])
    stats = {'pause_matched': 0, 'pause_fixed': 0, 'pause_dropped': 0, 'other': 0}
    for r in doc.get('removals') or []:
        if r.get('kind') == 'pause':
            i = r.get('i')
            if i is None or not (0 <= i < len(words) - 1) or not (words[i]['e'] - 0.002 <= r['a'] and words[i + 1]['s'] + 0.002 >= r['b']):
                i = find_pause_index(words, r['a'], r['b'], s['pauseThreshold']) if words else None
            if i is None:
                out['removals'].append({**r, 'kind': 'pause-fixed'}); stats['pause_fixed'] += 1
                continue
            rr = pause_removal(words, i, s)
            if rr is None:                # pause no longer worth shortening under current targets: keep as fixed so nothing is lost
                out['removals'].append({**r, 'kind': 'pause-fixed', 'i': i}); stats['pause_fixed'] += 1
            else:
                out['removals'].append(rr); stats['pause_matched'] += 1
        else:
            out['removals'].append(dict(r)); stats['other'] += 1
    if u8 is not None and out['edgeSnap'] != 3:          # edgeSnap 3 = natural-cut rule: keep real pauses around the cut (2026-08-28)
        from envelope import snap_removals_v3
        out['removals'] = snap_removals_v3(out['removals'], u8, words)
        out['edgeSnap'] = 3
        stats['snapped'] = True
    out['removals'].sort(key=lambda r: r['a'])
    return out, stats


def removed_union(ranges):
    """Merge overlapping or touching (a, b) ranges into a sorted list of disjoint [a, b] lists; empty and inverted ranges are dropped."""
    rs = sorted((float(a), float(b)) for a, b in ranges if b > a)
    out = []
    for a, b in rs:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# ------------------------------------------------------------------ join plan
def cam_at(cuts, t, fallback):
    """Camera active at master time t: the camera of the last cut with cut time <= t (cuts sorted by time), else `fallback`."""
    cam = None
    for c in cuts:
        if c['t'] <= t:
            cam = c['cam']
        else:
            break
    return cam or fallback


def angle_covers(layout, aid, t0, t1):
    """True when angle `aid` has a single coverage segment spanning all of [t0, t1] (1 us tolerance); False for unknown angles."""
    a = next((x for x in layout['angles'] if x['id'] == aid), None)
    return bool(a) and any(cv['start'] <= t0 + 1e-6 and cv['end'] >= t1 - 1e-6 for cv in a['coverage'])


def source_for(layout, aid, t):
    """Coverage segment (file, offset, start, end) of angle `aid` that contains master time t, or None."""
    a = next((x for x in layout['angles'] if x['id'] == aid), None)
    if not a:
        return None
    for cv in a['coverage']:
        if cv['start'] <= t + 1e-6 and cv['end'] >= t - 1e-6:
            return cv
    return None


def removal_kind_at(doc, lb, ra):
    """'words' if any word/filler removal takes part in the removed region [lb, ra], 'pause' for pure pause shortening, None if unknown."""
    kinds = set()
    for r in doc.get('removals') or []:
        if r['b'] > lb - 0.02 and r['a'] < ra + 0.02:
            kinds.add('words' if r.get('kind') in ('words', 'filler', 'pause-fixed') else 'pause')
    if not kinds:
        return None
    return 'words' if 'words' in kinds else 'pause'


def kept_pause_at(u8, words, lb, ra):
    """Natural silence kept on both sides of a cut: quiet going backward from lb (bounded by the previous word) + forward from ra."""
    if u8 is None:
        return 0.0
    from envelope import quiet_extent, START_BIAS, END_BIAS, KEEP_MAX
    prev_e = max([w['e'] + END_BIAS for w in words if w['e'] + END_BIAS <= lb + 0.01] or [lb - KEEP_MAX])
    next_s = min([w['s'] + START_BIAS for w in words if w['s'] + START_BIAS >= ra - 0.01] or [ra + KEEP_MAX])
    kb = min(KEEP_MAX, quiet_extent(u8, lb, -1, max_win=max(0.0, lb - prev_e)))
    ka = min(KEEP_MAX, quiet_extent(u8, ra, +1, max_win=max(0.0, next_s - ra)))
    return round(kb + ka, 3)


def other_camera(layout, cam, t0, t1):
    """Another angle covering [t0, t1] to switch to from `cam`: the other speaker cameras first, Slides last; None when nothing covers it."""
    order = [a['id'] for a in layout['angles'] if a['id'] != cam and a['id'] != 'slides'] + [a['id'] for a in layout['angles'] if a['id'] == 'slides' and a['id'] != cam]
    return next((aid for aid in order if angle_covers(layout, aid, t0, t1)), None)


def join_plan(intervals, cuts, words, doc, layout, fallback, motions=None, u8=None):
    """One entry per boundary between consecutive kept intervals. `motions`: {(lb,ra,cam): pose motion} from posematch.
    Word-removal joins follow the natural-cut rule: kept silence measured; a pause inserted when there is none; camera switch
    proposed (`switch_to`) when the speaker moved too much."""
    s = settings_of(doc)
    motions = motions or {}
    ov = doc.get('joinOverrides') or {}
    cuts = sorted(cuts, key=lambda c: c['t'])
    starts = sorted((w['s'], w['e']) for w in words)
    import bisect
    def speech_in(t0, t1):
        """Does any word overlap [t0, t1]? (words sorted by start; check the few candidates around t0)"""
        i = bisect.bisect_left(starts, (t0 - 3.0, 0))
        while i < len(starts) and starts[i][0] < t1:
            if starts[i][1] > t0:
                return True
            i += 1
        return False
    d_full = s['dissolveFrames'] / FPS
    plan = []
    for k in range(len(intervals) - 1):
        lb, ra = intervals[k][1], intervals[k + 1][0]
        cam_l, cam_r = cam_at(cuts, lb - 1 / FPS, fallback), cam_at(cuts, ra, fallback)
        d_edge = min(d_full, (ra - lb) / 2)
        # the dissolve only uses the edges of the removed region: it is a 'pause' join when those edges are silent
        kind = 'words' if speech_in(lb, lb + d_edge) or speech_in(ra - d_edge, ra) else 'pause'
        o = ov.get(f'{ra:.3f}') or {}
        motion = motions.get((round(lb, 3), round(ra, 3), cam_l))
        rkind = removal_kind_at(doc, lb, ra)
        kept = kept_pause_at(u8, words, lb, ra) if rkind == 'words' else None
        switch_to, insert = None, 0.0
        typ = o.get('type') if o.get('type') in ('cut', 'dissolve', 'hold', 'morph') else None
        if typ is None:
            if cam_l != cam_r or (ra - lb) < 2 / FPS:      # camera change, or a removal too short to be visible -> plain cut
                typ = 'cut'
            elif rkind == 'words':                          # natural-cut rule for word removals
                if motion is not None and motion >= MOTION_MAX and not o.get('type'):
                    switch_to = other_camera(layout, cam_l, ra, min(intervals[k + 1][1], ra + 2.0))
                if kept < PAUSE_MIN:
                    insert = round(PAUSE_TARGET - kept, 3)
                    typ = 'dissolve' if kind == 'pause' else 'hold'
                elif switch_to:
                    typ = 'cut'
                elif motion is not None and MOTION_MORPH <= motion < MOTION_MAX and s.get('morph', True):
                    typ = 'morph'
                else:
                    typ = 'dissolve' if kind == 'pause' else 'hold'
                    insert = min(6 / FPS, kept / 2)
            elif motion is not None and MOTION_MORPH <= motion < MOTION_MAX and s.get('morph', True):
                typ = 'morph'                               # visible but moderate jump -> warp instead of blend
            elif motion is not None and motion >= MOTION_MAX:
                typ = 'hold' if s['holdMs'] > 0 else 'cut'
            else:
                typ = 'dissolve' if kind == 'pause' else ('hold' if s['holdMs'] > 0 else 'cut')
        frames = 0
        avail = 0.0
        if typ == 'dissolve':
            if o.get('frames'):                       # user-chosen pause length: footage is padded (cloned frames / room tone) if the removal is shorter
                frames = min(int(o['frames']), 3 * FPS)
            elif rkind == 'words':
                frames = max(0, int(round(insert * FPS)))
            else:
                frames = min(int(s['dissolveFrames']), int((ra - lb) / 2 * FPS))
            d = frames / FPS
            avail = min(d, (ra - lb) / 2)             # real footage available on each side of the join
            if frames < 1 or not (angle_covers(layout, cam_l, lb, lb + avail) and angle_covers(layout, cam_r, ra - avail, ra)):
                typ, frames = ('hold', 0) if s['holdMs'] > 0 else ('cut', 0)
        if typ == 'morph':
            base = s['dissolveFrames'] if kind == 'pause' else round(s['holdMs'] / 1000 * FPS)
            frames = min(int(o.get('frames') or max(base, 4)), 3 * FPS)
            avail = min(frames / FPS, (ra - lb) / 2)
            if frames < 1:
                typ = 'cut'
        if typ == 'hold':
            frames = min(int(o.get('frames') or (round(insert * FPS) if rkind == 'words' else round(s['holdMs'] / 1000 * FPS))), 3 * FPS)
            if frames < 1:
                typ = 'cut'
        if typ == 'cut':
            frames = 0
        plan.append({'k': k, 'lb': lb, 'ra': ra, 'camL': cam_l, 'camR': cam_r, 'kind': kind, 'type': typ, 'frames': frames, 'add': frames / FPS,
                     'avail': round(avail, 4), 'manual': bool(o.get('frames') or o.get('type')), 'motion': motion,
                     'rkind': rkind, 'kept': kept, 'switch_to': switch_to})
    return plan


# ------------------------------------------------------------------ room tone
def room_tone(lecture_dir, words, force=False):
    """Quietest 1 s inside a word gap > 1.2 s -> _multicam/roomtone.wav (48 kHz stereo). Falls back to near-silence."""
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    dst = os.path.join(wd, 'roomtone.wav')
    if os.path.exists(dst) and not force:
        return dst
    proj = load_json(os.path.join(wd, 'project.json'))
    master = os.path.join(ld, proj['master_audio']['name'])
    best = None
    wav8 = os.path.join(wd, 'wav', proj['master_audio']['name'] + '.wav')
    if words and os.path.exists(wav8):
        import numpy as np
        from scipy.io import wavfile
        sr, x = wavfile.read(wav8)
        x = x.astype('float32') / 32768
        for i in range(len(words) - 1):
            g0, g1 = words[i]['e'] + 0.15, words[i + 1]['s'] - 0.15
            if g1 - g0 < 1.0:
                continue
            t = g0
            while t + 1.0 <= g1:
                seg = x[int(t * sr):int((t + 1.0) * sr)]
                if len(seg):
                    rms = float(np.sqrt(np.mean(seg ** 2)))
                    if best is None or rms < best[0]:
                        best = (rms, t)
                t += 0.25
    if best:
        run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{best[1]:.3f}', '-t', '1.0', '-i', master, '-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le', dst])
        print(f'room tone: {best[1]:.1f}s (rms {best[0]:.5f})')
    else:
        run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-t', '1.0', '-af', 'volume=-60dB', '-c:a', 'pcm_s16le', dst])
        print('room tone: synthetic near-silence (no word gaps available)')
    return dst


# ------------------------------------------------------------------ pose-matched fixes -> adjusted intervals
def apply_fixes(intervals, plan, fixes, overrides):
    """Move joins to their pose-matched frame pairs: kept interval k ends at lb2, k+1 starts at ra2.
    Overrides keyed by the original join time are re-keyed to the new one. Returns (intervals2, overrides2, n_moved)."""
    from posematch import fix_key
    iv = [list(x) for x in intervals]
    ov2 = dict(overrides or {})
    moved = 0
    for j in plan:
        fx = fixes.get(fix_key(j)) if fixes else None
        if not fx or fx.get('kept') or j['type'] == 'cut':
            continue
        lb2, ra2 = round(float(fx['lb2']) * FPS) / FPS, round(float(fx['ra2']) * FPS) / FPS    # pose fixes come from the 25 fps proxies: snap to the output grid
        k = j['k']
        # the join may move into the kept silence on either side, but must stay strictly inside the neighbouring kept intervals
        if not (iv[k][0] + 0.1 < lb2 < ra2 < iv[k + 1][1] - 0.1):
            continue
        iv[k][1] = lb2
        iv[k + 1][0] = ra2
        key_old, key_new = f"{j['ra']:.3f}", f"{ra2:.3f}"
        if key_old in ov2 and key_new != key_old:
            ov2[key_new] = ov2.pop(key_old)
        moved += 1
    return [tuple(x) for x in iv], ov2, moved
