"""Final render from the ORIGINAL files: cuts.json + sync -> pieces -> concat -> mux master audio."""
import math, os, shutil, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from common import run, load_json, save_json, work_dir

CHUNK = 90.0   # seconds: long pieces are split into chunks so every CPU core stays busy and progress is smooth

OW, OH, OFPS = 1920, 1080, 30
# encoder settings come from hw.py (NVENC / Quick Sync / VA-API / VideoToolbox when this machine proves them, else libx264);
# VTAIL is appended to the end of every video filter chain (VA-API needs an upload step). libx264 is the safe default until
# use_encoder() runs at the start of render().
ENC_KIND = 'libx264'
X264 = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-x264-params', f'keyint={OFPS * 2}:min-keyint={OFPS}', '-video_track_timescale', '90000']
VTAIL = ''
_FORCE_ENCODER = None


def use_encoder(kind=None):
    """Select the export encoder for this process (hw.encoder() unless forced); returns its kind."""
    global ENC_KIND, X264, VTAIL
    from hw import encoder, encoder_args
    kind = kind or _FORCE_ENCODER
    if kind:
        args, tail = encoder_args(kind, 'final', OFPS)
    else:
        kind, args, tail = encoder('final', OFPS)
    if os.environ.get('MULTICAM_TEST_BREAK_HW') and kind != 'libx264':      # test hook: make the hardware encoder fail
        args = args + ['-no_such_option_for_testing', '1']
    ENC_KIND, X264, VTAIL = kind, args, tail
    return kind


def qf(t):
    """Quantize a time to the output frame grid."""
    return round(t * OFPS) / OFPS


def kept_intervals(a, b, skips, min_keep=1 / OFPS):
    """[a,b) minus the union of skip ranges, all quantized to the frame grid; drops kept slivers < min_keep."""
    a, b = qf(a), qf(b)
    sk = sorted(((qf(s['a']), qf(s['b'])) for s in skips if s['b'] > a and s['a'] < b), key=lambda r: r[0])
    out, cur = [], a
    for sa, sb in sk:
        if sa > cur:
            out.append((cur, min(sa, b)))
        cur = max(cur, sb)
    if cur < b:
        out.append((cur, b))
    return [(x, y) for x, y in out if y - x >= min_keep - 1e-6]


def crop_vf(crop):
    """ffmpeg filter for a normalised 16:9 crop box {x,y,w} on the padded OWxOH frame (h follows w); '' when no crop."""
    if not crop or not (0 < float(crop.get('w', 1)) < 0.999):
        return ''
    w = max(0.2, min(1.0, float(crop['w'])))
    x = max(0.0, min(1 - w, float(crop.get('x', 0))))
    y = max(0.0, min(1 - w, float(crop.get('y', 0))))
    pw, ph = int(round(w * OW / 2)) * 2, int(round(w * OH / 2)) * 2
    px, py = int(round(x * OW / 2)) * 2, int(round(y * OH / 2)) * 2
    px, py = min(px, OW - pw), min(py, OH - ph)
    return f',crop={pw}:{ph}:{px}:{py},scale={OW}:{OH}:flags=lanczos'


def crop_at(cuts, t):
    """Crop box of the camera segment active at master time t (last cut <= t)."""
    cr = None
    for c in sorted(cuts, key=lambda c: c['t']):
        if qf(c['t']) <= t:
            cr = c.get('crop')
        else:
            break
    return cr


def image_at(images, t):
    """The accepted B-roll image whose [a,b) window contains master time t, or None."""
    for im in images or []:
        if im['a'] <= t < im['b']:
            return im
    return None


def expand_pieces(cuts, layout, intervals, fallback, images=None):
    """Turn the cut list into atomic pieces [{a,b,angle,file,offset,crop,image}] over the kept intervals, coverage resolved.
    Piece boundaries also fall on the edges of accepted B-roll images, so a piece is either fully inside an image or outside."""
    angles = {a['id']: a for a in layout['angles']}
    cuts = sorted(cuts, key=lambda c: c['t'])
    images = images or []
    segs, warnings = [], []
    for in_pt, out_pt in intervals:
        extra = [qf(x) for im in images for x in (im['a'], im['b']) if in_pt < qf(x) < out_pt]
        bounds = sorted(set([in_pt] + [qf(c['t']) for c in cuts if in_pt < qf(c['t']) < out_pt] + extra + [out_pt]))
        for a, b in zip(bounds, bounds[1:]):
            if b - a <= 0:
                continue
            cam, crop = None, None
            for c in cuts:
                if qf(c['t']) <= a:
                    cam, crop = c['cam'], c.get('crop')
            segs.append((a, b, cam or fallback, crop))

    def covering(cam, t):
        """Coverage entry of `cam` whose [start,end) contains master time t, or None."""
        for cv in angles[cam]['coverage']:
            if cv['start'] <= t < cv['end']:
                return cv
        return None

    pieces = []
    for a, b, cam, crop in segs:
        t = a
        while b - t > 1e-6:
            cv = covering(cam, t)
            src_cam = cam
            if cv is None:
                # next point where cam becomes covered (or end)
                nxt = min([cv2['start'] for cv2 in angles[cam]['coverage'] if cv2['start'] > t] + [b])
                cv = covering(fallback, t)
                src_cam = fallback
                warnings.append(f'{cam} has no signal at {t:.1f}s -> using {fallback}')
                if cv is None:
                    pieces.append({'a': t, 'b': nxt, 'angle': cam, 'file': None, 'offset': 0, 'crop': crop})
                    t = nxt
                    continue
                end = min(nxt, cv['end'], b)
            else:
                end = min(cv['end'], b)
            end = qf(end)
            if end <= t:
                end = min(b, t + 1 / OFPS)
            pieces.append({'a': t, 'b': end, 'angle': src_cam, 'file': cv['file'], 'offset': cv['offset'], 'crop': crop})
            t = end
    for p in pieces:
        im = image_at(images, (p['a'] + p['b']) / 2)
        if im:
            p['image'] = im
    # merge consecutive pieces that continue the same source file (removes 1-frame slivers)
    merged = []
    for p in pieces:
        q = merged[-1] if merged else None
        if q and q['file'] == p['file'] and q['offset'] == p['offset'] and q.get('crop') == p.get('crop') and q.get('image') == p.get('image') and abs(q['b'] - p['a']) < 1e-6:
            q['b'] = p['b']
        else:
            merged.append(dict(p))
    return merged, warnings


def split_pieces(pieces):
    """Split pieces longer than CHUNK seconds into equal frame-grid-aligned chunks (parallel encoding, smooth progress); others pass through."""
    out = []
    for p in pieces:
        n = max(1, math.ceil((p['b'] - p['a']) / CHUNK))
        if n == 1:
            out.append(p)
            continue
        step = (p['b'] - p['a']) / n
        t = p['a']
        for i in range(n):
            e = p['b'] if i == n - 1 else qf(t + step)
            out.append({**p, 'a': t, 'b': e})
            t = e
    return out


def has_drawtext():
    """False on ffmpeg builds without libfreetype (credits then go to the .txt only)."""
    if os.environ.get('MULTICAM_TEST_NO_DRAWTEXT'):
        return False
    from hw import ffmpeg_filters
    return 'drawtext' in ffmpeg_filters()


def credit_filter(text, tmpdir):
    """drawtext for the image credit (bottom-right, small, boxed); '' when there is nothing to credit or no drawtext filter."""
    text = (text or '').strip()
    if not text or not has_drawtext():
        return ''
    path = os.path.join(tmpdir, 'credit.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return f",drawtext=textfile='{path}':fontsize=26:fontcolor=white@0.9:box=1:boxcolor=black@0.45:boxborderw=8:x=w-tw-28:y=h-th-22"


def pip_xy(pos):
    """overlay x:y expressions for a corner position ('tr' default), 40 px margins."""
    return {'tl': ('40', '40'), 'tr': ('W-w-40', '40'), 'bl': ('40', 'H-h-40'), 'br': ('W-w-40', 'H-h-40')}.get(pos or 'tr', ('W-w-40', '40'))


def image_vf(piece, n, tmpdir):
    """Full-frame B-roll: fill the frame, optional slow zoom continuous across the image window, credit line."""
    im = piece['image']
    z = float(im.get('zoom') if im.get('zoom') is not None else 1.08)
    if z <= 1.0005:                                         # zoom off: a plain still
        return (f'scale={OW}:{OH}:force_original_aspect_ratio=increase,crop={OW}:{OH},setsar=1,fps={OFPS},format=yuv420p' + credit_filter(im.get('credit'), tmpdir))
    span = max(1e-6, im['b'] - im['a'])
    f0, f1 = (piece['a'] - im['a']) / span, (piece['b'] - im['a']) / span
    z0, z1 = 1 + (z - 1) * f0, 1 + (z - 1) * f1
    return (f'scale={OW}:{OH}:force_original_aspect_ratio=increase,crop={OW}:{OH},setsar=1,'
            f"zoompan=z='{z0:.5f}+({z1:.5f}-{z0:.5f})*on/{max(1, n)}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={OW}x{OH}:fps={OFPS},"
            f'format=yuv420p' + credit_filter(im.get('credit'), tmpdir))


def encode_piece(ld, piece, dst, on_progress=None, vf_extra=''):
    """Encode one piece; on_progress(seconds_done) is called as ffmpeg reports progress. vf_extra: e.g. a fade-in."""
    n = round((piece['b'] - piece['a']) * OFPS)
    if n <= 0:
        return None
    im = piece.get('image')
    if im and im.get('mode', 'full') == 'full':
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            img = os.path.join(os.path.dirname(os.path.dirname(dst)), '..', 'broll', im['src']) if not os.path.isabs(im['src']) else im['src']
            img = os.path.normpath(img)
            cmd = ['ffmpeg', '-y', '-v', 'error', '-loop', '1', '-framerate', str(OFPS), '-t', f'{n / OFPS + 0.5:.3f}', '-i', img,
                   '-vf', image_vf(piece, n, td) + vf_extra + VTAIL, '-frames:v', str(n), *X264, '-nostats', '-progress', 'pipe:1', dst]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in proc.stdout:
                if line.startswith('out_time_us=') and on_progress:
                    try: on_progress(int(line.split('=')[1]) / 1e6)
                    except ValueError: pass
            err = proc.stderr.read()
            if proc.wait() != 0:
                raise RuntimeError(f"image piece {piece['a']:.2f}-{piece['b']:.2f} failed: {err[-1500:]}")
        if on_progress:
            on_progress(piece['b'] - piece['a'])
        return dst
    if piece['file'] is None:
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', f'color=c=black:s={OW}x{OH}:r={OFPS}',
               '-vf', 'format=yuv420p' + VTAIL, '-frames:v', str(n), *X264]
    else:
        ss = piece['a'] - piece['offset']
        src = os.path.join(ld, piece['file'])
        vf = (f'fps={OFPS},scale={OW}:{OH}:force_original_aspect_ratio=decrease,pad={OW}:{OH}:-1:-1,'
              f'format=yuv420p' + crop_vf(piece.get('crop')) + ',tpad=stop_mode=clone:stop=-1' + vf_extra)
        cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', f'{ss:.4f}', '-i', src, '-map', '0:v:0', '-an',
               '-vf', vf + VTAIL, '-frames:v', str(n), *X264]
        if im and im.get('mode') == 'pip':      # picture-in-picture: camera stays, image in the top-right corner
            import tempfile
            td = tempfile.mkdtemp()
            img = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(dst)), '..', 'broll', im['src']))
            pw = int(round(OW * max(0.15, min(0.6, float(im.get('size') or 0.35))) / 2)) * 2
            ox, oy = pip_xy(im.get('pos'))
            cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', f'{ss:.4f}', '-i', src, '-loop', '1', '-framerate', str(OFPS), '-t', f'{n / OFPS + 0.5:.3f}', '-i', img,
                   '-filter_complex', f'[0:v]{vf}[base];[1:v]scale={pw}:-2:force_original_aspect_ratio=decrease,pad=iw+6:ih+6:3:3:color=white,format=yuv420p[img];'
                                      f'[base][img]overlay={ox}:{oy}:shortest=0{credit_filter(im.get("credit"), td)}{VTAIL}[v]',
                   '-map', '[v]', '-an', '-frames:v', str(n), *X264]
    cmd += ['-nostats', '-progress', 'pipe:1', dst]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith('out_time_us=') and on_progress:
            try:
                on_progress(int(line.split('=')[1]) / 1e6)
            except ValueError:
                pass
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on piece {piece['a']:.2f}-{piece['b']:.2f}: {err[-2000:]}")
    if on_progress:
        on_progress(piece['b'] - piece['a'])
    return dst


def verify_output(path, expected_frames, expected_len):
    """ffprobe the muxed file and compare frame count and video/audio durations with the plan; returns a summary string.
    Raises RuntimeError on mismatch (frames exact, video within 0.1 s, audio within 0.2 s)."""
    import json
    pr = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', path],
                                   capture_output=True, text=True).stdout)
    v = next(s for s in pr['streams'] if s['codec_type'] == 'video')
    a = next(s for s in pr['streams'] if s['codec_type'] == 'audio')
    frames, vdur, adur = int(v.get('nb_frames', 0)), float(v.get('duration', 0)), float(a.get('duration', 0))
    info = f'{frames} frames (expected {expected_frames}), video {vdur:.2f}s, audio {adur:.2f}s (expected {expected_len:.2f}s)'
    if frames != expected_frames or abs(vdur - expected_len) > 0.1 or abs(adur - expected_len) > 0.2:
        raise RuntimeError('output verification FAILED: ' + info)
    return info


def cam_at(cuts, t, fallback):
    """Camera id active at master time t (last cut whose quantized time <= t), or fallback when there is none."""
    cam = None
    for c in sorted(cuts, key=lambda c: c['t']):
        if qf(c['t']) <= t:
            cam = c['cam']
    return cam or fallback


def auto_switch_cuts(cuts, layout, intervals, fallback):
    """Ephemeral cuts that change angle at every join where the same camera would continue (hides jump cuts)."""
    angles = layout['angles']
    def covers(aid, t0, t1):
        """True when angle aid has a single coverage entry spanning all of [t0, t1]."""
        return any(cv['start'] <= t0 and cv['end'] >= t1 for cv in next(a for a in angles if a['id'] == aid)['coverage'])
    extra = []
    work = list(cuts)
    for (la, lb), (ra, rb) in zip(intervals, intervals[1:]):
        cam_l = cam_at(work, lb - 1 / OFPS, fallback)
        cam_r = cam_at(work, ra, fallback)
        if cam_l != cam_r:
            continue
        need_until = min(rb, ra + 2.0)
        # prefer other camera angles, Slides last
        order = [a['id'] for a in angles if a['id'] != cam_l and a['id'] != 'slides'] + [a['id'] for a in angles if a['id'] == 'slides' and a['id'] != cam_l]
        alt = next((aid for aid in order if covers(aid, ra, need_until)), None)
        if alt:
            extra.append({'t': ra, 'cam': alt, 'auto': True})
            work.append(extra[-1])
    return extra


def kept_words(words, intervals, corrections):
    """Words whose midpoint lies in a kept interval, with the text replaced by the user's correction (keyed '%.3f' of the start)."""
    out = []
    for w in words:
        mid = (w['s'] + w['e']) / 2
        if any(a <= mid < b for a, b in intervals):
            out.append({**w, 't': corrections.get(f"{w['s']:.3f}", w['t'])})
    return out


def words_to_text(words):
    """Paragraphs: break at long original gaps or sentence end + pause."""
    paras, cur, prev = [], [], None
    for w in words:
        if prev is not None:
            gap = w['s'] - prev['e']
            if gap > 1.5 or (prev['t'][-1:] in '.?!' and gap > 0.6):
                paras.append(' '.join(cur))
                cur = []
        cur.append(w['t'])
        prev = w
    if cur:
        paras.append(' '.join(cur))
    return '\n\n'.join(paras) + '\n'


def safe_name(name):
    """File-system-safe form of a clip name (runs of odd characters -> '_'); 'clip' when nothing is left."""
    import re
    return re.sub(r'[^\w\- .]+', '_', name).strip() or 'clip'


def vf_chain():
    """Base video filter chain of every camera frame: 30 fps, letterboxed to 1920x1080, yuv420p."""
    return (f'fps={OFPS},scale={OW}:{OH}:force_original_aspect_ratio=decrease,pad={OW}:{OH}:-1:-1,format=yuv420p')


def encode_transition(ld, layout, j, dst, on_progress=None, cuts=None, images=None):
    """Dissolve / hold / morph transition piece of exactly j['frames'] frames (see joins.join_plan); uses the framing of the adjacent shots.
    If an accepted image is showing on either side, that side is the image frame (still)."""
    from joins import source_for
    n = j['frames']
    imL, imR = image_at(images, j['lb'] - 1 / OFPS), image_at(images, j['ra'])
    if (imL or imR) and n > 0:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            def still(im, cam, t, out):
                """Write one PNG at `out`: the full-frame image `im` when it is showing, else camera `cam` at master time t with its crop."""
                if im and im.get('mode', 'full') == 'full':
                    img = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(dst)), '..', 'broll', im['src']))
                    run(['ffmpeg', '-y', '-v', 'error', '-loop', '1', '-t', '0.1', '-i', img, '-vf', f'scale={OW}:{OH}:force_original_aspect_ratio=increase,crop={OW}:{OH},setsar=1,format=yuv420p' + credit_filter(im.get('credit'), td), '-frames:v', '1', out])
                else:
                    cv = source_for(layout, cam, t)
                    run(['ffmpeg', '-y', '-v', 'error', '-ss', f"{max(0.0, t - cv['offset']):.4f}", '-i', os.path.join(ld, cv['file']), '-vf', vf_chain() + crop_vf(crop_at(cuts or [], t)), '-frames:v', '1', out])
            pa, pb = os.path.join(td, 'a.png'), os.path.join(td, 'b.png')
            still(imL, j['camL'], j['lb'] - 1 / OFPS, pa); still(imR, j['camR'], j['ra'], pb)
            d = n / OFPS
            run(['ffmpeg', '-y', '-v', 'error', '-loop', '1', '-t', f'{d + 0.5:.3f}', '-i', pa, '-loop', '1', '-t', f'{d + 0.5:.3f}', '-i', pb,
                 '-filter_complex', f'[0:v]format=yuv420p,setsar=1[a];[1:v]format=yuv420p,setsar=1[b];[a][b]xfade=transition=fade:duration={d:.4f}:offset=0{VTAIL}[v]',
                 '-map', '[v]', '-frames:v', str(n), *X264, dst])
        if on_progress:
            on_progress(n / OFPS)
        return dst
    vfL = vf_chain() + crop_vf(crop_at(cuts or [], j['lb'] - 1 / OFPS))
    vfR = vf_chain() + crop_vf(crop_at(cuts or [], j['ra']))
    if n <= 0:
        return None
    d = n / OFPS
    cvL, cvR = source_for(layout, j['camL'], j['lb']), source_for(layout, j['camR'], j['ra'])
    if cvL is None or cvR is None:
        raise RuntimeError(f"no source footage for transition at {j['ra']:.2f}s")
    fL, fR = os.path.join(ld, cvL['file']), os.path.join(ld, cvR['file'])
    if j['type'] == 'morph':
        from morph import morph_frames, write_video
        import cv2, tempfile
        with tempfile.TemporaryDirectory() as td:
            pa, pb = os.path.join(td, 'a.png'), os.path.join(td, 'b.png')
            run(['ffmpeg', '-y', '-v', 'error', '-ss', f"{max(0.0, j['lb'] - 1 / OFPS - cvL['offset']):.4f}", '-i', fL, '-vf', vfL, '-frames:v', '1', pa])
            run(['ffmpeg', '-y', '-v', 'error', '-ss', f"{j['ra'] - cvR['offset']:.4f}", '-i', fR, '-vf', vfR, '-frames:v', '1', pb])
            A, B = cv2.imread(pa), cv2.imread(pb)
        frames, info = morph_frames(A, B, n) if A is not None and B is not None else (None, {'ok': False})
        if frames is not None:
            write_video(frames, dst, OFPS, X264, VTAIL)
            if on_progress:
                on_progress(d)
            return dst
        print(f"  morph at {j['ra']:.2f}s not reliable ({info}); falling back to {'dissolve' if j['kind'] == 'pause' else 'hold'}", flush=True)
        j = {**j, 'type': 'dissolve' if j['kind'] == 'pause' else 'hold'}
    if j['type'] == 'dissolve':
        av = j.get('avail', d) or d                    # footage available per side; the rest is padded with cloned frames
        padB = max(0.0, d - av)
        inputs = ['-ss', f"{j['lb'] - cvL['offset']:.4f}", '-t', f'{av + 0.2:.4f}', '-i', fL,
                  '-ss', f"{j['ra'] - av - cvR['offset']:.4f}", '-t', f'{av + 0.2:.4f}', '-i', fR]
        fc = (f'[0:v]{vfL},trim=duration={av:.4f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop=-1[a];'
              f'[1:v]{vfR},trim=duration={av:.4f},setpts=PTS-STARTPTS'
              + (f',tpad=start_mode=clone:start_duration={padB:.4f}' if padB > 0.001 else '') + ',tpad=stop_mode=clone:stop=-1[b];'
              f'[a][b]xfade=transition=fade:duration={d:.4f}:offset=0{VTAIL}[v]')
    else:  # hold: last frame of the left side dissolving into the first frame of the right side
        inputs = ['-ss', f"{max(0.0, j['lb'] - 1 / OFPS - cvL['offset']):.4f}", '-t', '0.2', '-i', fL,
                  '-ss', f"{j['ra'] - cvR['offset']:.4f}", '-t', '0.2', '-i', fR]
        fc = (f'[0:v]{vfL},trim=end_frame=1,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop=-1[a];'
              f'[1:v]{vfR},trim=end_frame=1,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop=-1[b];'
              f'[a][b]xfade=transition=fade:duration={d:.4f}:offset=0{VTAIL}[v]')
    cmd = ['ffmpeg', '-y', '-v', 'error', *inputs, '-filter_complex', fc, '-map', '[v]', '-frames:v', str(n), *X264, dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"transition at {j['ra']:.2f}s failed: {r.stderr[-1500:]}")
    if on_progress:
        on_progress(d)
    return dst


def encode_tail(ld, layout, cam, t_end, frames, dst, fade_s, cuts=None, images=None):
    """Ending: hold the last frame for `frames` frames while fading to black (with the last shot's framing, or the image if one is showing)."""
    from joins import source_for
    im = image_at(images, t_end - 1 / OFPS)
    if im and im.get('mode', 'full') == 'full':
        img = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(dst)), '..', 'broll', im['src']))
        d = frames / OFPS
        run(['ffmpeg', '-y', '-v', 'error', '-loop', '1', '-framerate', str(OFPS), '-t', f'{d + 0.5:.3f}', '-i', img,
             '-vf', f'scale={OW}:{OH}:force_original_aspect_ratio=increase,crop={OW}:{OH},setsar=1,format=yuv420p,fade=t=out:st=0:d={max(0.04, min(fade_s, d)):.4f}' + VTAIL, '-frames:v', str(frames), *X264, dst])
        return dst
    cv = source_for(layout, cam, t_end - 1 / OFPS)
    vfT = vf_chain() + crop_vf(crop_at(cuts or [], t_end - 1 / OFPS))
    if cv is None:
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', f'color=c=black:s={OW}x{OH}:r={OFPS}', '-vf', 'format=yuv420p' + VTAIL, '-frames:v', str(frames), *X264, dst]
    else:
        d = frames / OFPS
        cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', f"{max(0.0, t_end - 1 / OFPS - cv['offset']):.4f}", '-t', '0.2', '-i', os.path.join(ld, cv['file']),
               '-vf', f'{vfT},trim=end_frame=1,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop=-1,fade=t=out:st=0:d={max(0.04, min(fade_s, d)):.4f}' + VTAIL,
               '-frames:v', str(frames), *X264, dst]
    run(cmd)
    return dst


def audio_item(master, roomtone, item, dst):
    """One wav per timeline item, exact length, click-free edges."""
    from joins import FADE
    if item['type'] == 'audio':          # kept interval [a,b) from the master audio
        a, b = item['a'], item['b']
        ln = b - a
        fi = max(FADE, min(float(item.get('fadeIn', 0) or 0), ln / 2))
        fo = max(FADE, min(float(item.get('fadeOut', 0) or 0), ln / 2))     # last kept interval: decay the voice before the ending tail
        af = f'afade=t=in:d={fi:.4f},afade=t=out:st={max(0.0, ln - fo):.4f}:d={fo:.4f}'
        cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', f'{a:.4f}', '-t', f'{ln:.4f}', '-i', master, '-af', af]
    elif item['type'] == 'tail':         # ending: room tone fading out
        d = item['len']
        cmd = ['ffmpeg', '-y', '-v', 'error', '-stream_loop', '-1', '-i', roomtone, '-t', f'{d:.4f}',
               '-af', f'afade=t=in:d={FADE},afade=t=out:st=0:d={d:.4f}']
    elif item['type'] == 'dissolve' or (item['type'] == 'morph' and item['join']['kind'] == 'pause'):   # crossfade of the (room-tone) edges of the removed region
        j, d = item['join'], item['join']['add']
        av = min(j.get('avail', d) or d, d)
        pad = d - av
        if pad < 0.002:
            cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', f"{j['lb']:.4f}", '-t', f'{d + 0.05:.4f}', '-i', master,
                   '-ss', f"{j['ra'] - d - 0.05:.4f}", '-t', f'{d + 0.05:.4f}', '-i', master,
                   '-filter_complex', f'[0:a][1:a]acrossfade=d={d:.4f}:c1=tri:c2=tri,atrim=0:{d:.4f}[a]', '-map', '[a]']
        else:
            cmd = ['ffmpeg', '-y', '-v', 'error', '-ss', f"{j['lb']:.4f}", '-t', f'{av:.4f}', '-i', master,
                   '-ss', f"{j['ra'] - av:.4f}", '-t', f'{av:.4f}', '-i', master, '-stream_loop', '-1', '-i', roomtone,
                   '-filter_complex',
                   f'[2:a]atrim=0:{pad + 0.05:.4f},asetpts=PTS-STARTPTS[ra];[2:a]atrim=0:{pad + 0.05:.4f},asetpts=PTS-STARTPTS[rb];'
                   f'[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a0];[1:a]aformat=sample_rates=48000:channel_layouts=stereo[b0];'
                   f'[a0][ra]concat=n=2:v=0:a=1[A];[rb][b0]concat=n=2:v=0:a=1[B];'
                   f'[A][B]acrossfade=d={d:.4f}:c1=tri:c2=tri,atrim=0:{d:.4f},apad=whole_dur={d:.4f}[a]', '-map', '[a]']
    else:                                # hold: room tone
        d = item['join']['add']
        cmd = ['ffmpeg', '-y', '-v', 'error', '-stream_loop', '-1', '-i', roomtone, '-t', f'{d:.4f}',
               '-af', f'afade=t=in:d={FADE},afade=t=out:st={max(0.0, d - FADE):.4f}:d={FADE}']
    cmd += ['-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le', dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"audio item {item['type']} failed: {r.stderr[-1000:]}")
    return dst


def render(lecture_dir, out=None, workers=4, clip=None, tighten=False):
    """Export the full video or one named clip from the ORIGINAL files to `out` (default multicam_output.mp4 / clips/<name>.mp4); returns the path.
    kept intervals (cuts.json skips + tighten.json removals) -> join plan, pose-matched -> video and audio items encoded in parallel
    into _multicam/render/pieces -> concat with explicit durations -> mux -> verify_output; also writes the .txt transcript and
    render/status.json throughout. If a hardware encoder fails mid-way the whole export is redone with libx264."""
    from joins import migrate_tighten, join_plan, room_tone, cam_at
    from transcribe import load_words, words_in
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    rd = os.path.join(wd, 'render')
    pd = os.path.join(rd, 'pieces')
    shutil.rmtree(pd, ignore_errors=True)
    os.makedirs(pd, exist_ok=True)
    status_path = os.path.join(rd, 'status.json')

    def status(**kw):
        """Write render/status.json with the given fields plus a timestamp."""
        save_json(status_path, {**kw, 'time': time.time()})

    try:
        enc_kind = use_encoder()
        print(f'video encoder: {enc_kind}', flush=True)
        proj = load_json(os.path.join(wd, 'project.json'))
        layout = load_json(os.path.join(wd, 'layout.json'))
        cutsdoc = load_json(os.path.join(wd, 'cuts.json')) or {}
        cuts = sorted(cutsdoc.get('cuts', []), key=lambda c: c['t'])
        DUR = float(layout['duration'])
        in_pt = qf(cutsdoc.get('in') or 0.0)
        out_pt = qf(cutsdoc.get('out') or DUR)
        fallback = cutsdoc.get('fallback') or layout['angles'][0]['id']
        if not cuts:
            raise RuntimeError('no cuts defined')
        skips = cutsdoc.get('skips', [])
        target = 'full video'
        if clip:
            c = next((c for c in cutsdoc.get('clips', []) if c.get('name') == clip), None)
            if c is None:
                raise RuntimeError(f'clip {clip!r} not found in cuts.json')
            in_pt, out_pt = qf(c['a']), qf(c['b'])
            target = f'clip "{clip}"'
            out = out or os.path.join(ld, 'clips', safe_name(clip) + '.mp4')
            clip_edges = True
        out = out or os.path.join(ld, 'multicam_output.mp4')
        os.makedirs(os.path.dirname(out), exist_ok=True)

        # lecture-wide word-level edits (Tighten tab), migrated to the current scheme
        all_words = load_words(wd)['words']
        import envelope as envmod
        u8 = envmod.load(wd)
        tdoc, _mig = migrate_tighten(load_json(os.path.join(wd, 'tighten.json'), {}) or {}, all_words, u8)
        rem_docs = tdoc['removals']
        if clip:
            # breathing room: keep real footage after the last word (and a little before the first) while the audio is quiet,
            # never reaching into the neighbouring words; pause-shortening removals inside that room are ignored
            from posematch import Envelope
            env = Envelope(wd)
            lead, hold = float(tdoc.get('startLead', 0.3) or 0), float(tdoc.get('endHold', 1.0) or 0)
            nxt = min([w['s'] for w in all_words if w['s'] > out_pt + 0.02] or [DUR])
            prv = max([w['e'] for w in all_words if w['e'] < in_pt - 0.02] or [0.0])
            ext_out = min(hold, max(0.0, nxt - 0.06 - out_pt), env.quiet_extent(out_pt, +1, max_win=hold) if env.ok else hold)
            ext_in = min(lead, max(0.0, in_pt - (prv + 0.06)), env.quiet_extent(in_pt, -1, max_win=lead) if env.ok else lead)
            new_out, new_in = qf(out_pt + ext_out), qf(in_pt - ext_in)
            rem_docs = [r for r in rem_docs if not (r.get('kind') == 'pause' and ((out_pt - 0.1 <= r['a'] < new_out + 0.1) or (new_in - 0.1 <= r['b'] <= in_pt + 0.1)))]
            print(f'clip room: +{ext_in:.2f}s before the first word, +{ext_out:.2f}s after the last word')
            in_pt, out_pt = new_in, new_out
        removals = [{'a': r['a'], 'b': r['b']} for r in rem_docs]
        words = words_in({'words': all_words}, in_pt, out_pt)
        if removals:
            skips = skips + removals
            target += ' (tightened)'
        intervals = kept_intervals(in_pt, out_pt, skips, min_keep=0.15)
        if not intervals:
            raise RuntimeError('nothing left to render after cut-outs')
        plan = join_plan(intervals, cuts, all_words, tdoc, layout, fallback, u8=u8)
        # pose-matched cut points (Berthouzoz et al. 2012): move each same-camera join to the best-matching frame pair
        n_moved = 0
        if tdoc.get('poseMatch', True) and any(j['type'] != 'cut' for j in plan):
            from joins import apply_fixes
            from posematch import refine_all
            status(state='matching', target=target, progress=0.02, message='pose-matching joins (finding frames where the speaker stands the same way)')
            fixes = refine_all(ld, plan, all_words, status_cb=lambda n, tot, el: status(state='matching', target=target, progress=0.02 + 0.06 * n / tot, message=f'pose-matching joins {n}/{tot}'))
            from posematch import motions_from_fixes
            motions = motions_from_fixes(plan, fixes)
            intervals, ov2, n_moved = apply_fixes(intervals, plan, fixes, tdoc.get('joinOverrides') or {})
            # re-plan on the adjusted intervals; motions are keyed by the ORIGINAL join region, so re-key by index
            plan2 = join_plan(intervals, cuts, all_words, {**tdoc, 'joinOverrides': ov2}, layout, fallback, u8=u8)
            m2 = {}
            for j0, j1 in zip(plan, plan2):
                m = motions.get((round(j0['lb'], 3), round(j0['ra'], 3), j0['camL']))
                if m is not None:
                    m2[(round(j1['lb'], 3), round(j1['ra'], 3), j1['camL'])] = m
            plan = join_plan(intervals, cuts, all_words, {**tdoc, 'joinOverrides': ov2}, layout, fallback, motions=m2, u8=u8)
            # natural-cut rule: the speaker moved too much across a same-camera word cut -> switch camera there (automatic)
            switches = [j for j in plan if j.get('switch_to') and not any(abs(c['t'] - j['ra']) < 0.3 for c in cuts)]
            if switches:
                for j in switches:
                    cuts.append({'t': j['ra'], 'cam': j['switch_to'], 'auto': True})
                cuts = sorted(cuts, key=lambda c: c['t'])
                print(f'auto camera switch at {len(switches)} jerky join(s): ' + ', '.join(f"{j['ra']:.1f}s→{j['switch_to']}" for j in switches[:8]))
                plan = join_plan(intervals, cuts, all_words, {**tdoc, 'joinOverrides': ov2}, layout, fallback, motions=m2, u8=u8)
        kept_len = sum(b - a for a, b in intervals)
        added = sum(j['add'] for j in plan)
        total_len = kept_len + added
        fade_in = max(0.0, float(tdoc.get('fadeIn', 0.3) or 0))
        tail_frames = int(round(max(0.0, float(tdoc.get('tailOut', 0.8) or 0)) * OFPS))
        tail_len = tail_frames / OFPS
        total_len += tail_len
        roomtone = room_tone(ld, all_words) if (any(j['type'] != 'cut' for j in plan) or tail_frames) else None

        # ---- timeline items: video pieces per kept interval, transition pieces between intervals; audio per item
        vitems, aitems, warnings = [], [], []
        images = [im for im in (cutsdoc.get('images') or []) if im.get('src') and im['b'] > in_pt and im['a'] < out_pt]
        for k, (a, b) in enumerate(intervals):
            pieces, w = expand_pieces(cuts, layout, [(a, b)], fallback, images=images)
            warnings += w
            for p in split_pieces(pieces):
                vitems.append({'type': 'video', 'piece': p, 'len': p['b'] - p['a']})
            aitems.append({'type': 'audio', 'a': a, 'b': b, 'len': b - a})
            if k < len(plan) and plan[k]['type'] != 'cut':
                j = plan[k]
                vitems.append({'type': j['type'], 'join': j, 'len': j['add']})
                aitems.append({'type': j['type'], 'join': j, 'len': j['add']})
        if vitems:
            vitems[0]['fadeIn'] = fade_in
            aitems[0]['fadeIn'] = fade_in
        if tail_frames:
            # fade the voice only over real room after the last word — never over the word itself
            room = float(locals().get('ext_out', 0.4) or 0) if clip else 0.4
            aitems[-1]['fadeOut'] = max(0.12, min(0.6, room - 0.05))   # >= 120 ms: hides the sliver of the next word when speech was continuous
            last_cam = cam_at(cuts, intervals[-1][1] - 1 / OFPS, fallback)
            vitems.append({'type': 'tail', 'cam': last_cam, 't_end': intervals[-1][1], 'frames': tail_frames, 'len': tail_len})
            aitems.append({'type': 'tail', 'len': tail_len})
        # never list a piece that cannot have a file: drop zero-frame slivers (they carry no time)
        vitems = [it for it in vitems if not (it['type'] == 'video' and round((it['piece']['b'] - it['piece']['a']) * OFPS) <= 0)]
        if any((im.get('credit') or '').strip() for im in (images or [])) and not has_drawtext():
            warnings.append("this ffmpeg has no drawtext filter: image credits are not burned into the picture (they are listed in the .txt)")
        warn = sorted(set(warnings))
        for w in warn:
            print('WARNING:', w)
        n_dis = sum(1 for j in plan if j['type'] == 'dissolve'); n_hold = sum(1 for j in plan if j['type'] == 'hold'); n_morph = sum(1 for j in plan if j['type'] == 'morph')
        n_ins = sum(1 for j in plan if j.get('rkind') == 'words' and j.get('kept') is not None and j['kept'] < 0.30 and j['add'] > 0)
        print(f'natural cuts: {n_ins} word joins got a pause inserted, {sum(1 for j in plan if j.get("rkind") == "words") - n_ins} kept their natural pause')
        print(f'{target}: {len(intervals)} kept interval(s), {len(vitems)} video pieces ({n_dis} dissolves, {n_hold} holds, {n_morph} morphs, +{added:.1f}s micro pauses; {n_moved} joins pose-matched; fade-in {fade_in:.1f}s, ending {tail_len:.1f}s), {total_len:.1f}s of output -> {out}')

        prog = [0.0] * len(vitems)
        done = [0]
        t_start = time.time()
        lock = threading.Lock()

        def write_status():
            """Publish encoding progress (seconds encoded, speed, ETA, join counts, warnings) to status.json."""
            enc = sum(min(prog[i], vitems[i]['len']) for i in range(len(vitems)))
            el = time.time() - t_start
            rate = enc / el if el > 2 else 0
            eta = (total_len - enc) / rate if rate > 0 else None
            status(state='encoding', target=target, progress=(enc / total_len) * 0.9 if total_len else 0,
                   total=len(vitems), done=done[0], encoded_s=enc, total_s=total_len, eta_s=eta, speed=rate,
                   joins={'dissolve': n_dis, 'hold': n_hold, 'morph': n_morph, 'added_s': added, 'pose_matched': n_moved},
                   message=f'encoded {enc / 60:.1f} of {total_len / 60:.1f} min ({done[0]}/{len(vitems)} pieces, {rate:.2f}x realtime)'
                           + (f', ~{eta / 60:.0f} min left' if eta is not None else ''),
                   warnings=warn)

        stop = threading.Event()
        def ticker():
            """Background loop: refresh status.json every 2 s until `stop` is set."""
            while not stop.wait(2.0):
                with lock:
                    write_status()
        threading.Thread(target=ticker, daemon=True).start()
        write_status()

        def vjob(i_item):
            """Encode video item i (camera piece, transition or ending tail) to pieces/piece_XXXX.mp4; returns the path."""
            i, it = i_item
            dst = os.path.join(pd, f'piece_{i:04d}.mp4')
            def on_prog(sec):
                """Record the seconds encoded so far for item i (read by the progress ticker)."""
                prog[i] = sec
            if it['type'] == 'video':
                fi = float(it.get('fadeIn', 0) or 0)
                encode_piece(ld, it['piece'], dst, on_prog, vf_extra=(f',fade=t=in:st=0:d={fi:.3f}' if fi > 0 else ''))
                p = it['piece']
                print(f'  piece {i:04d} {p["angle"]:8s} {p["a"]:8.2f}-{p["b"]:8.2f}', flush=True)
            elif it['type'] == 'tail':
                encode_tail(ld, layout, it['cam'], it['t_end'], it['frames'], dst, it['len'], cuts=cuts, images=images)
                on_prog(it['len'])
                print(f'  piece {i:04d} ending: hold last frame {it["frames"]} frames + fade out', flush=True)
            else:
                encode_transition(ld, layout, it['join'], dst, on_prog, cuts=cuts, images=images)
                print(f'  piece {i:04d} {it["type"]:8s} join @ {it["join"]["ra"]:8.2f} ({it["join"]["frames"]} frames)', flush=True)
            with lock:
                done[0] += 1
            return dst
        def ajob(i_item):
            """Render audio item i from the master audio / room tone to pieces/audio_XXXX.wav; returns the path."""
            i, it = i_item
            return audio_item(os.path.join(ld, proj['master_audio']['name']), roomtone, it, os.path.join(pd, f'audio_{i:04d}.wav'))
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                afiles = list(ex.map(ajob, enumerate(aitems)))
                vfiles = list(ex.map(vjob, enumerate(vitems)))
        except RuntimeError as e:
            if enc_kind != 'libx264' and not _FORCE_ENCODER:      # hardware encoder broke mid-way: redo the whole export on the CPU
                stop.set()
                print(f'hardware encoder {enc_kind} failed ({str(e)[-400:]}); restarting the export with the CPU encoder (libx264)', flush=True)
                globals()['_FORCE_ENCODER'] = 'libx264'
                return render(lecture_dir, out=out, workers=workers, clip=clip, tighten=tighten)
            raise
        finally:
            stop.set()

        status(state='muxing', target=target, progress=0.92, message='concatenating + muxing audio')
        def frames_of(it):
            """Exact frame count of a video item (piece, tail or transition) for the concat durations and the verification."""
            if it['type'] == 'video':
                return round((it['piece']['b'] - it['piece']['a']) * OFPS)
            if it['type'] == 'tail':
                return it['frames']
            return it['join']['frames']
        # NOTE: explicit per-piece durations are essential (some sources yield wrong container durations)
        missing = [(i, it['type']) for i, (path, it) in enumerate(zip(vfiles, vitems)) if not path or not os.path.exists(path)]
        if missing:
            raise RuntimeError(f'internal error: {len(missing)} piece file(s) were not produced: {missing[:5]}')
        vlst = os.path.join(rd, 'concat.txt')
        with open(vlst, 'w') as f:
            f.write('ffconcat version 1.0\n')
            for path, it in zip(vfiles, vitems):
                f.write("file '%s'\nduration %.6f\n" % (path.replace("'", r"'\''"), frames_of(it) / OFPS))
        expected_frames = sum(frames_of(it) for it in vitems)
        alst = os.path.join(rd, 'audio_concat.txt')
        with open(alst, 'w') as f:
            f.write('ffconcat version 1.0\n')
            for path in afiles:
                f.write("file '%s'\n" % path.replace("'", r"'\''"))
        awav = os.path.join(rd, 'audio.wav')
        run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', alst, '-c:a', 'pcm_s16le', awav])
        tmp = out + '.part.mp4'
        run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', vlst, '-i', awav,
             '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
             '-movflags', '+faststart', tmp])
        vinfo = verify_output(tmp, expected_frames, total_len)
        # captions: .srt/.ass sidecars are always written; when enabled the .ass is burned in (one finishing pass, re-verified)
        if words:
            import captions as cap
            capdoc = cap.load(wd)
            segs, tt = [], 0.0
            for it in vitems:
                ln = frames_of(it) / OFPS
                if it['type'] == 'video':
                    segs.append((it['piece']['a'], it['piece']['b'], tt))
                tt += ln
            # all_words, not the clip-scoped list: caption overrides are keyed by the word's index in words.json
            cblocks = cap.blocks(cap.kept(all_words, intervals, tdoc.get('corrections', {}), capdoc, segs), capdoc)
            if cblocks:
                from hw import ffmpeg_filters
                burn = capdoc.get('enabled') and 'ass' in ffmpeg_filters()
                # burned exports keep their sidecars in a captions/ subfolder: next to the .mp4 a video player
                # auto-loads the .srt and draws a SECOND set of subtitles over the burned ones
                if burn:
                    capdir = os.path.join(os.path.dirname(out) or '.', 'captions')
                    os.makedirs(capdir, exist_ok=True)
                    base = os.path.join(capdir, os.path.splitext(os.path.basename(out))[0])
                    for ext in ('.srt', '.ass'):   # a sidecar from an earlier non-burned render would still double the captions
                        try:
                            os.remove(os.path.splitext(out)[0] + ext)
                        except FileNotFoundError:
                            pass
                else:
                    base = os.path.splitext(out)[0]
                cap.write_srt(base + '.srt', cblocks)
                asspath = cap.write_ass(base + '.ass', cblocks, capdoc)
                print(f'captions: {len(cblocks)} blocks -> {base}.srt/.ass')
                if capdoc.get('enabled'):
                    if burn:
                        status(state='captions', target=target, progress=0.95, message='burning captions into the picture')
                        tmp2 = out + '.cc.mp4'
                        # run from the .ass directory so the filter argument is a simple relative name
                        run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-vf', 'ass=' + cap.filter_escape(os.path.basename(asspath)) + VTAIL,
                             *X264, '-c:a', 'copy', '-movflags', '+faststart', tmp2], cwd=os.path.dirname(asspath) or '.')
                        verify_output(tmp2, expected_frames, total_len)
                        os.replace(tmp2, tmp)
                        print(f'captions burned in ({len(cblocks)} blocks)')
                    else:
                        warn.append('this ffmpeg cannot burn captions (built without libass) — the .srt/.ass files are next to the export')
        os.replace(tmp, out)
        print('verified:', vinfo)
        if words:
            kw = kept_words(words, intervals, tdoc.get('corrections', {}))
            txt = os.path.splitext(out)[0] + '.txt'
            with open(txt, 'w', encoding='utf-8') as f:
                f.write(words_to_text(kw))
                used = [im for im in images if any(a < im['b'] and b > im['a'] for a, b in intervals)]
                if used:
                    f.write('\n\nImages / credits\n')
                    for im in used:
                        f.write(f"- {im.get('query') or im.get('src')}: {im.get('credit') or 'own image'}" + (f" — {im['source_url']}" if im.get('source_url') else '') + '\n')
            print(f'transcript {txt} ({len(kw)} words)')
        removed = (out_pt - in_pt) - kept_len
        print(f'{out_pt - in_pt:.1f}s -> {total_len:.1f}s (removed {removed:.1f}s, added back {added:.1f}s in {n_dis} dissolves + {n_hold} holds; {len(removals)} word-level removals, {len(skips) - len(removals)} cut-outs)')
        status(state='done', target=target, progress=1, message=f'done: {out}', output=out, warnings=warn,
               joins={'dissolve': n_dis, 'hold': n_hold, 'morph': n_morph, 'added_s': added, 'pose_matched': n_moved})
        return out
    except Exception as e:
        status(state='error', target=locals().get('target', ''), progress=0, message=str(e)[-2000:])
        raise
