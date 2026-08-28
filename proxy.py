"""Proxies: per-file 640x360 -> per-angle aligned -> multiview grid with master audio."""
import os, math
from concurrent.futures import ThreadPoolExecutor
from common import run, load_json, save_json, work_dir

PW, PH, PFPS = 640, 360, 25
X264 = ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-g', str(PFPS * 2), '-pix_fmt', 'yuv420p']


def fresh(dst, *srcs):
    """True when dst exists, is non-empty and is newer (mtime) than every source in srcs."""
    return os.path.exists(dst) and os.path.getsize(dst) > 0 and all(os.path.getmtime(dst) > os.path.getmtime(s) for s in srcs)


def file_proxy(src, dst):
    """Encode the 640x360 25 fps silent proxy of one source file to dst (skipped when fresh); written as .part.mp4 then renamed."""
    if fresh(dst, src):
        print(f'  proxy cached: {os.path.basename(dst)}')
        return
    print(f'  encoding proxy: {os.path.basename(src)} ...', flush=True)
    tmp = dst + '.part.mp4'
    run(['ffmpeg', '-y', '-v', 'error', '-i', src, '-map', '0:v:0', '-an',
         '-vf', f'fps={PFPS},scale={PW}:{PH}:force_original_aspect_ratio=decrease,pad={PW}:{PH}:-1:-1,format=yuv420p',
         *X264, '-movflags', '+faststart', tmp])
    os.replace(tmp, dst)
    print(f'  done: {os.path.basename(dst)}', flush=True)


def black_clip(dst, dur):
    """Create a black PWxPH clip of `dur` seconds at dst (fills coverage gaps); no-op when the file already exists."""
    if os.path.exists(dst):
        return
    run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', f'color=c=black:s={PW}x{PH}:r={PFPS}', '-t', f'{dur:.3f}',
         *X264, dst])


def coverage_for(angle, sync, DUR):
    """Per-file [start,end] on the master clock, clipped, non-overlapping, in time order."""
    segs = []
    for f in angle['files']:
        off = sync['files'][f['name']]['offset']
        segs.append((off, f))
    segs.sort(key=lambda s: s[0])
    cov, cursor = [], 0.0
    for off, f in segs:
        start, end = max(off, 0.0), min(off + f['duration'], DUR)
        start = max(start, cursor)
        if end - start < 0.5:
            continue
        cov.append({'start': start, 'end': end, 'file': f['name'], 'offset': off})
        cursor = end
    return cov


def angle_proxy(angle, cov, DUR, wd, files_dir):
    """Assemble the master-clock-aligned proxy of one angle: its file proxies trimmed with inpoint/outpoint, black clips in the gaps,
    concatenated to exactly DUR seconds. Writes proxies/<id>.concat.txt and proxies/<id>.mp4 (cached while newer than its
    inputs and sync.json); returns the mp4 path."""
    dst = os.path.join(wd, 'proxies', f'{angle["id"]}.mp4')
    srcs = [os.path.join(files_dir, c['file'] + '.mp4') for c in cov]
    if fresh(dst, *srcs) and os.path.getmtime(dst) > os.path.getmtime(os.path.join(wd, 'sync.json')):
        print(f'  angle proxy cached: {angle["id"]}')
        return dst
    print(f'  assembling angle proxy: {angle["id"]} ...', flush=True)
    gaps_dir = os.path.join(wd, 'proxies', 'gaps')
    os.makedirs(gaps_dir, exist_ok=True)
    lines, cursor = ['ffconcat version 1.0'], 0.0
    def q(p):
        """Escape single quotes for an ffconcat file line."""
        return p.replace("'", r"'\''")
    for c in cov:
        if c['start'] - cursor > 0.02:
            g = os.path.join(gaps_dir, f'black_{c["start"] - cursor:.3f}.mp4')
            black_clip(g, c['start'] - cursor)
            lines.append(f"file '{q(g)}'")
        lines.append(f"file '{q(os.path.join(files_dir, c['file'] + '.mp4'))}'")
        lines.append(f"inpoint {c['start'] - c['offset']:.3f}")
        lines.append(f"outpoint {c['end'] - c['offset']:.3f}")
        cursor = c['end']
    if DUR - cursor > 0.02:
        g = os.path.join(gaps_dir, f'black_{DUR - cursor:.3f}.mp4')
        black_clip(g, DUR - cursor)
        lines.append(f"file '{q(g)}'")
    lst = os.path.join(wd, 'proxies', f'{angle["id"]}.concat.txt')
    open(lst, 'w').write('\n'.join(lines) + '\n')
    tmp = dst + '.part.mp4'
    run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-an',
         '-vf', f'fps={PFPS},format=yuv420p', *X264, '-t', f'{DUR:.3f}', '-movflags', '+faststart', tmp])
    os.replace(tmp, dst)
    return dst


def build_proxies(lecture_dir, files_only=False, workers=3):
    """Build every proxy of a lecture: A) per-file proxies in parallel, B) aligned per-angle proxies + layout.json (grid geometry,
    coverage), C) multiview.mp4 (xstack grid with the master audio). files_only stops after A; B and C need sync.json."""
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    proj = load_json(os.path.join(wd, 'project.json'))
    files_dir = os.path.join(wd, 'proxies', 'files')
    os.makedirs(files_dir, exist_ok=True)

    jobs = [(os.path.join(ld, f['name']), os.path.join(files_dir, f['name'] + '.mp4'))
            for a in proj['angles'] for f in a['files']]
    print(f'step A: {len(jobs)} file proxies ({workers} parallel)')
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: file_proxy(*j), jobs))
    if files_only:
        return

    sync = load_json(os.path.join(wd, 'sync.json'))
    if not sync:
        raise SystemExit('run `sync` first')
    DUR = float(proj['master_audio']['duration'])

    print('step B: aligned angle proxies')
    layout = {'width': 0, 'height': 0, 'tile_w': PW, 'tile_h': PH, 'fps': PFPS, 'duration': DUR, 'angles': []}
    n = len(proj['angles'])
    cols = 2 if n <= 4 else 3
    rows = max(1, math.ceil(n / cols))
    layout['width'], layout['height'] = cols * PW, rows * PH
    angle_files = []
    for i, a in enumerate(proj['angles']):
        cov = coverage_for(a, sync, DUR)
        angle_files.append(angle_proxy(a, cov, DUR, wd, files_dir))
        layout['angles'].append({
            'id': a['id'], 'name': a['name'], 'key': a['key'], 'color': a['color'],
            'x': (i % cols) * PW, 'y': (i // cols) * PH, 'w': PW, 'h': PH,
            'coverage': cov,
        })
    save_json(os.path.join(wd, 'layout.json'), layout)

    print('step C: multiview')
    mv = os.path.join(wd, 'multiview.mp4')
    master = os.path.join(ld, proj['master_audio']['name'])
    if fresh(mv, *angle_files, master):
        print('  multiview cached')
        return
    inputs = []
    for p in angle_files:
        inputs += ['-i', p]
    inputs += ['-i', master]
    lay = '|'.join(f'{a["x"]}_{a["y"]}' for a in layout['angles'])
    if n == 1:
        fc = '[0:v]copy[v]'
    else:
        fc = ''.join(f'[{i}:v]' for i in range(n)) + f'xstack=inputs={n}:layout={lay}:fill=black[v]'
    tmp = mv + '.part.mp4'
    run(['ffmpeg', '-y', '-v', 'error', *inputs, '-filter_complex', fc, '-map', '[v]', '-map', f'{n}:a:0',
         *X264, '-c:a', 'aac', '-b:a', '128k', '-t', f'{DUR:.3f}', '-movflags', '+faststart', tmp])
    os.replace(tmp, mv)
    print(f'wrote {mv}')
