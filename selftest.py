"""Integration test: serve a (synthetic) project, exercise the API, render, verify. Usage: selftest.py <lecture dir> [port]"""
import json, os, subprocess, sys, time, urllib.request, shutil

here = os.path.dirname(os.path.abspath(__file__))


def make_synthetic(T):
    """Build a tiny fake lecture: 120 s master audio (pink noise + beeps every 7 s), a Zoom-like screen video, camera A
    (+10 s, 100 s long) and camera B in two segments (+20..+50, +60..+90), a VTT transcript; then init + sync + proxy."""
    def ff(*args):
        """Run ffmpeg quietly inside T, raising on failure."""
        subprocess.run(['ffmpeg', '-v', 'error', '-y', *args], check=True, cwd=T)
    shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
    ff('-f', 'lavfi', '-i', 'anoisesrc=color=pink:seed=7:r=48000:d=120', '-f', 'lavfi', '-i', 'sine=frequency=880:r=48000:d=120',
       '-filter_complex', "[1:a]volume='if(lt(mod(t,7),0.2),1,0)':eval=frame[b];[0:a][b]amix=inputs=2:normalize=0[a]", '-map', '[a]',
       '-c:a', 'aac', '-b:a', '128k', 'zoom audio.m4a')
    ff('-f', 'lavfi', '-i', 'testsrc2=size=640x400:rate=25:duration=120', '-i', 'zoom audio.m4a', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-c:a', 'copy', '-shortest', 'screen_zoom.mp4')
    ff('-f', 'lavfi', '-i', 'testsrc=size=640x360:rate=30:duration=100', '-ss', '10', '-t', '100', '-i', 'zoom audio.m4a', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-c:a', 'aac', '-shortest', 'camA.mp4')
    ff('-f', 'lavfi', '-i', 'smptebars=size=640x360:rate=30:duration=30', '-ss', '20', '-t', '30', '-i', 'zoom audio.m4a', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-c:a', 'aac', '-shortest', 'rec1_B.mp4')
    ff('-f', 'lavfi', '-i', 'smptehdbars=size=640x360:rate=30:duration=30', '-ss', '60', '-t', '30', '-i', 'zoom audio.m4a', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-c:a', 'aac', '-shortest', 'rec2_B.mp4')
    with open(os.path.join(T, 'transcript.vtt'), 'w', newline='') as f:
        f.write('WEBVTT\r\n\r\n1\r\n00:00:01.000 --> 00:00:05.000\r\nTester: Hello and welcome.\r\n\r\n2\r\n00:00:30.000 --> 00:00:40.500\r\n'
                'Tester: Second sentence here.\r\n\r\n3\r\n00:01:30.000 --> 00:01:50.000\r\nTester: Final words.\r\n')
    for cmd in (['init', T], ['sync', T], ['proxy', T, '--workers', '2']):
        subprocess.run([sys.executable, os.path.join(here, 'multicam.py'), *cmd], check=True, stdout=subprocess.DEVNULL)
    print('synthetic project ready:', T)


args = [a for a in sys.argv[1:] if a != '--make']
ld = os.path.abspath(args[0]); port = int(args[1]) if len(args) > 1 else 8799
if '--make' in sys.argv or not os.path.isdir(os.path.join(ld, '_multicam')):
    make_synthetic(ld)
base = f'http://127.0.0.1:{port}'
# start from a clean per-run state (caches from a previous run would change join plans / suggestions)
for rel in ('joinfix.json', 'words.json', 'tighten.json', 'workspace.json'):
    try: os.remove(os.path.join(ld, '_multicam', rel))
    except FileNotFoundError: pass
for d in ('broll', 'gaze', 'checkpoints', 'posematch', 'transcribe', 'topics', 'render'):
    shutil.rmtree(os.path.join(ld, '_multicam', d), ignore_errors=True)
os.makedirs(os.path.join(ld, '_multicam', 'logs'), exist_ok=True)
SRV_LOG = os.path.join(ld, '_multicam', 'logs', 'selftest-server.log')       # server output (tracebacks land here; printed on failure)
srv = subprocess.Popen([sys.executable, os.path.join(here, 'multicam.py'), 'serve', ld, '--port', str(port)],
                       stdout=open(SRV_LOG, 'w'), stderr=subprocess.STDOUT)
def req(path, method='GET', body=None, headers={}):
    """HTTP request to the local server (body JSON-encoded); returns (status, headers dict, body bytes)."""
    r = urllib.request.Request(base + path, method=method, data=(json.dumps(body).encode() if body is not None else None),
                               headers={'Content-Type': 'application/json', **headers})
    with urllib.request.urlopen(r) as resp:
        return resp.status, dict(resp.headers), resp.read()
try:
    for _ in range(50):
        try: req('/'); break
        except Exception: time.sleep(0.2)
    st, h, b = req('/'); assert st == 200 and b'Multicam Cutter' in b, 'index'
    st, h, b = req('/api/project'); P = json.loads(b); assert P['layout']['angles'], 'project'
    print('angles:', [(a['key'], a['id']) for a in P['layout']['angles']], 'cues:', len(P['cues']), 'has_multiview:', P['has_multiview'])
    st, h, b = req('/media/multiview.mp4', headers={'Range': 'bytes=100-199'})
    assert st == 206 and len(b) == 100 and h['Content-Range'].startswith('bytes 100-199/'), ('range', st, h)
    print('range request OK:', h['Content-Range'])
    ids = [a['id'] for a in P['layout']['angles']]
    cuts = {'cuts': [{'t': 2.0, 'cam': ids[2]}, {'t': 15.0, 'cam': ids[0], 'crop': {'x': 0.15, 'y': 0.1, 'w': 0.7}}, {'t': 45.0, 'cam': ids[1]},   # framed shot; runs into B gap at 50 -> fallback
                     {'t': 70.5, 'cam': ids[0]}, {'t': 100.0, 'cam': ids[2]}], 'in': 1.0, 'out': 110.0, 'fallback': ids[2], 'lead': 0.4, 'snapLive': False,
            'skips': [{'a': 0.0, 'b': 12.5}, {'a': 60.0, 'b': 66.0}],            # 11.5 s + 6 s removed from the full video
            'clips': [{'a': 55.0, 'b': 80.0, 'name': 'History of memory', 'id': 'c_test01'}]}      # 25 s clip, minus the 6 s skip inside = 19 s
    st, h, b = req('/api/cuts', 'PUT', cuts); assert st == 200
    assert json.load(open(os.path.join(ld, '_multicam', 'cuts.json')))['cuts'][2]['t'] == 45.0
    print('cuts saved OK (block 2 framed at 1.43x zoom)')
    out = os.path.join(ld, 'multicam_output.mp4')
    if os.path.exists(out): os.remove(out)
    st, h, b = req('/api/tighten', 'PUT', {'removals': [], 'corrections': {}, 'edgeSnap': 3, 'startLead': 0, 'endHold': 0}); assert st == 200   # clean slate for the length checks below
    # start + cancel
    st, h, b = req('/api/render', 'POST', {}); assert st == 200
    time.sleep(3)
    st, h, b = req('/api/render/cancel', 'POST', {}); assert st == 200, (st, b)
    st, h, b = req('/api/render/status'); S = json.loads(b); assert S['state'] == 'cancelled' and not S['running'], S
    for _ in range(40):
        left = subprocess.run(['pgrep', '-fc', f'ffmpeg .*{os.path.basename(ld)}.*piece_'], capture_output=True, text=True).stdout.strip()
        if left in ('', '0'): break
        time.sleep(0.25)
    assert left in ('', '0'), f'ffmpeg children still running after cancel: {left}'
    print('cancel OK (no orphaned ffmpeg)')
    st, h, b = req('/api/render', 'POST', {}); assert st == 200
    t0 = time.time()
    while True:
        st, h, b = req('/api/render/status'); S = json.loads(b)
        if S.get('state') in ('done', 'error') and not S.get('running'): break
        if time.time() - t0 > 600: raise SystemExit('render timeout')
        time.sleep(1)
    print('render:', S['state'], S.get('target'), '|', S.get('message'), '| pieces:', S.get('total'), '| warnings:', S.get('warnings'))
    assert S['state'] == 'done', open(os.path.join(ld, '_multicam', 'logs', 'render.log')).read()[-3000:]
    pr = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', out], capture_output=True, text=True).stdout)
    v = [s for s in pr['streams'] if s['codec_type'] == 'video'][0]; a = [s for s in pr['streams'] if s['codec_type'] == 'audio'][0]
    added = float((S.get('joins') or {}).get('added_s', 0))
    exp = 110.0 - 12.5 - 6.0 + added + 0.8   # + default 0.8 s ending
    print(f"output: {v['width']}x{v['height']} {v['r_frame_rate']} frames={v.get('nb_frames')} vdur={float(v['duration']):.3f} adur={float(a['duration']):.3f} expected={exp:.3f} (joins {S.get('joins')})")
    assert (S.get('joins') or {}).get('dissolve') == 1 and abs(added - 0.2) < 1e-6, S.get('joins')   # the 60-66 cut-out: same camera, no words -> one 6-frame dissolve
    assert abs(float(v['duration']) - exp) < 0.05 and abs(float(a['duration']) - exp) < 0.1, (v['duration'], a['duration'], exp)
    # framed block (master 15-45 s = output ~2.5-32.5 s): the output frame must equal the 70% centre-ish crop of the source frame, scaled up
    import numpy as np
    def frame(path, ss, vf):
        """One frame of `path` at ss seconds through filter vf, as a flat float32 grayscale array."""
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{ss:.3f}', '-i', path, '-frames:v', '1', '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True).stdout
        return np.frombuffer(raw, np.uint8).astype(np.float32)
    out_f = frame(out, 20.0 - 12.5 + 0.3, 'scale=192:108')                                          # output at master 20.0 s
    src_full = frame(os.path.join(ld, 'camA.mp4'), 20.0 - 10.0, 'scale=1920:1080,scale=192:108')    # camA offset 10 s
    src_crop = frame(os.path.join(ld, 'camA.mp4'), 20.0 - 10.0, 'scale=1920:1080,crop=1344:756:288:108,scale=192:108')
    d_full, d_crop = np.abs(out_f - src_full).mean(), np.abs(out_f - src_crop).mean()
    print(f'framing check: output vs cropped source diff {d_crop:.1f}, vs full source diff {d_full:.1f}')
    assert d_crop < d_full * 0.5, 'framed block was not cropped in the render'
    # clip export
    st, h, b = req('/api/render', 'POST', {'clip': 'History of memory'}); assert st == 200
    t0 = time.time()
    while True:
        st, h, b = req('/api/render/status'); S = json.loads(b)
        if S.get('state') in ('done', 'error') and not S.get('running'): break
        if time.time() - t0 > 600: raise SystemExit('clip render timeout')
        time.sleep(1)
    print('clip render:', S['state'], S.get('target'), S.get('message'))
    assert S['state'] == 'done'
    cout = os.path.join(ld, 'clips', 'History of memory.mp4'); assert os.path.exists(cout), cout
    pr = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', cout], capture_output=True, text=True).stdout)
    v = [s for s in pr['streams'] if s['codec_type'] == 'video'][0]; a = [s for s in pr['streams'] if s['codec_type'] == 'audio'][0]
    added = float((S.get('joins') or {}).get('added_s', 0)); exp = 19.0 + added + 0.8
    print(f"clip: frames={v.get('nb_frames')} vdur={float(v['duration']):.3f} adur={float(a['duration']):.3f} expected={exp:.3f}")
    assert abs(added - 0.2) < 1e-6 and abs(float(v['duration']) - exp) < 0.05 and abs(float(a['duration']) - exp) < 0.1
    # ---- tighten mode: synthetic word-level transcript + edit doc
    words = [{'s': round(55.0 + 0.5 * i, 3), 'e': round(55.3 + 0.5 * i, 3), 't': f'w{i}', 'p': 0.9} for i in range(10)]      # 55.0 .. 59.8
    words += [{'s': round(68.0 + 0.5 * i, 3), 'e': round(68.3 + 0.5 * i, 3), 't': f'v{i}', 'p': 0.9} for i in range(22)]     # 68.0 .. 78.8
    json.dump({'model': 'test', 'ranges': [{'a': 50.0, 'b': 85.0}], 'words': words}, open(os.path.join(ld, '_multicam', 'words.json'), 'w'))
    st, h, b = req('/api/words?a=55&b=80'); W = json.loads(b); assert W['covered'] and len(W['words']) == 32, (W['covered'], len(W['words']))
    print('words endpoint OK (covered, 32 words)')
    # v1-style doc (no version): the server/renderer must migrate it. Word removal v10..v13 sits inside camera 'cama' on both
    # sides -> a HOLD join; the 60-66 cut-out has no words inside and the same camera on both sides -> a DISSOLVE join.
    # manual pause lengths: 0.5 s on the dissolve join (66.000; 6 s of footage available) and 0.5 s on the hold join (75.000)
    tdoc = {'removals': [{'a': 73.0, 'b': 75.0, 'kind': 'words', 'label': 'v10..v13'}], 'corrections': {'68.000': 'CORRECTED'}, 'keepBreath': 0.15,
            'joinOverrides': {'66.000': {'type': 'dissolve', 'frames': 15}, '75.000': {'frames': 15}}, 'edgeSnap': 3, 'startLead': 0, 'endHold': 0}   # noise audio: skip acoustic snapping / clip room
    st, h, b = req('/api/tighten', 'PUT', tdoc); assert st == 200
    st, h, b = req('/api/tighten'); TD = json.loads(b); assert TD['removals'][0]['a'] == 73.0 and TD.get('version') == 2 and TD.get('edgeSnap') == 3 and 'keepBreath' not in TD, TD
    print('tighten migrated view OK (v2, keepBreath dropped)')
    st, h, b = req('/api/envelope'); assert st == 200 and len(b) > 1000, ('envelope', st, len(b)); print(f'envelope endpoint OK ({len(b)} hops = {len(b) * 0.01:.0f}s)')
    st, h, b = req('/api/topics/status'); assert json.loads(b).get('state') in ('idle', 'done', 'error'), b
    st, h, b = req('/api/render', 'POST', {'clip': 'History of memory'}); assert st == 200   # lecture-wide removals apply automatically
    t0 = time.time()
    while True:
        st, h, b = req('/api/render/status'); S = json.loads(b)
        if S.get('state') in ('done', 'error') and not S.get('running'): break
        if time.time() - t0 > 600: raise SystemExit('tighten render timeout')
        time.sleep(1)
    print('tighten render:', S['state'], S.get('target'), '|', S.get('message'))
    assert S['state'] == 'done', open(os.path.join(ld, '_multicam', 'logs', 'render.log')).read()[-3000:]
    pr = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', cout], capture_output=True, text=True).stdout)
    v = [s for s in pr['streams'] if s['codec_type'] == 'video'][0]; a = [s for s in pr['streams'] if s['codec_type'] == 'audio'][0]
    joins = S.get('joins') or {}
    added = float(joins.get('added_s', 0))
    exp = 25.0 - 6.0 - 2.0 + added + 0.8
    print(f"tightened clip: frames={v.get('nb_frames')} vdur={float(v['duration']):.3f} adur={float(a['duration']):.3f} expected={exp:.3f} (25 - 6 skip - 2 removed + {added:.3f} micro pauses; {joins})")
    assert joins.get('dissolve') == 1 and joins.get('hold') == 1, joins
    assert abs(added - 1.0) < 1e-6, added          # two manual 0.5 s pauses
    assert abs(float(v['duration']) - exp) < 0.05 and abs(float(a['duration']) - exp) < 0.1
    assert int(v.get('nb_frames')) == round(exp * 30), (v.get('nb_frames'), exp * 30)
    txt = open(os.path.join(ld, 'clips', 'History of memory.txt')).read()
    toks = set(txt.split())
    print('txt:', txt.strip().replace('\n', ' | ')[:200])
    assert 'CORRECTED' in toks and 'v0' not in toks and not ({'v10', 'v11', 'v12', 'v13'} & toks) and {'v9', 'v14', 'w9'} <= toks, toks
    print('tighten txt OK')
    # audio boundaries must be click-free: no sample-to-sample jump above -30 dBFS anywhere near the joins
    import wave
    wavp = os.path.join(ld, '_multicam', 'render', 'audio.wav')
    with wave.open(wavp) as wf:
        n = wf.getnframes(); ch = wf.getnchannels(); data = wf.readframes(n)
    import array
    smp = array.array('h', data)[::ch]
    # boundaries between audio items (from the concat list) must not introduce jumps larger than the signal itself has
    bounds, pos = [], 0
    for line in open(os.path.join(ld, '_multicam', 'render', 'audio_concat.txt')):
        if line.startswith('file '):
            f = line.strip()[6:-1]
            with wave.open(f) as w2: pos += w2.getnframes()
            bounds.append(pos)
    bounds = bounds[:-1]
    near = set(); [near.update(range(max(1, b - 48), min(len(smp), b + 48))) for b in bounds]
    jumps = [abs(smp[i] - smp[i - 1]) for i in range(1, len(smp))]
    at_b = max((jumps[i - 1] for i in near), default=0); elsewhere = max(j for i, j in enumerate(jumps, 1) if i not in near)
    print(f'audio: {len(bounds)} item boundaries; max jump at boundaries {at_b} vs elsewhere {elsewhere}')
    tail_rms = (sum(v * v for v in smp[-2400:]) / 2400) ** 0.5; body_rms = (sum(v * v for v in smp[48000:96000]) / 48000) ** 0.5
    print(f'ending fade: last 50 ms RMS {tail_rms:.0f} vs body {body_rms:.0f}'); assert tail_rms < body_rms * 0.05, 'ending did not fade out'
    assert at_b <= elsewhere * 1.05, 'audio click at an item boundary'
    # ---- checkpoints: v2 checkpoint embeds words; restore brings back tighten + cuts (+ words if they differ)
    st, h, b = req('/api/checkpoints', 'POST', {'name': 'selftest cp'}); cp = json.loads(b); assert cp.get('words_fingerprint'), cp
    st, h, b = req('/api/tighten', 'PUT', {'removals': [], 'corrections': {}, 'edgeSnap': 3}); assert st == 200
    json.dump({'model': 'other', 'ranges': [{'a': 50.0, 'b': 85.0}], 'words': words[:5]}, open(os.path.join(ld, '_multicam', 'words.json'), 'w'))   # simulate a re-transcription
    st, h, b = req('/api/checkpoints/restore', 'POST', {'file': cp['file']}); R = json.loads(b); assert R['ok'], R
    st, h, b = req('/api/tighten'); TD2 = json.loads(b); assert len(TD2['removals']) == 1 and TD2['removals'][0]['a'] == 73.0, TD2
    W2 = json.load(open(os.path.join(ld, '_multicam', 'words.json'))); assert len(W2['words']) == 32 and 'words restored' in R.get('note', ''), R
    st, h, b = req('/api/checkpoints'); names = [c['name'] for c in json.loads(b)['checkpoints']]; assert 'selftest cp' in names and 'before restore' in names
    # a legacy (v1, no words) checkpoint restores too and is served migrated
    legacy = {'meta': {'name': 'legacy v1', 'time': 1.0, 'stamp': '2000-01-01_00-00-00'}, 'cuts': cuts,
              'tighten': {'removals': [{'a': 60.35, 'b': 65.65, 'kind': 'pause', 'label': 'old fixed pause'}], 'keepBreath': 0.15, 'corrections': {}, 'edgeSnap': 3}}
    json.dump(legacy, open(os.path.join(ld, '_multicam', 'checkpoints', '2000-01-01_00-00-00_legacy v1.json'), 'w'))
    st, h, b = req('/api/checkpoints/restore', 'POST', {'file': '2000-01-01_00-00-00_legacy v1.json'}); assert json.loads(b)['ok']
    st, h, b = req('/api/tighten'); TD3 = json.loads(b); r0 = TD3['removals'][0]
    assert TD3['version'] == 2 and r0['kind'] == 'pause' and r0.get('i') == 9 and 'keepBreath' not in TD3, TD3   # matched the 59.8 -> 68.0 pause after word w9
    print(f"checkpoints OK: v2 restore (words restored: {'words restored' in R.get('note','')}), legacy v1 restore migrated -> pause i={r0['i']} [{r0['a']}, {r0['b']}] {r0['label']}")
    # ---- pose matching job (synthetic test patterns have no person -> image-difference fallback; must complete and stay consistent)
    st, h, b = req('/api/joins/refine', 'POST', {'force': True}); assert st == 200
    t0 = time.time()
    while True:
        st, h, b = req('/api/joins/status'); PS = json.loads(b)
        if PS.get('state') in ('done', 'error') and not PS.get('running'): break
        if time.time() - t0 > 300: raise SystemExit('posematch timeout')
        time.sleep(1)
    assert PS['state'] == 'done', PS
    st, h, b = req('/api/joins/fixes'); FX = json.loads(b); print('posematch:', PS['message'], '| fixes cached:', len(FX), '| methods:', sorted(set(f['method'] for f in FX.values())))
    assert len(FX) >= 1
    # ---- gaze job: synthetic test patterns contain no faces -> must finish with 0 proposals, no error
    st, h, b = req('/api/gaze', 'POST', {'a': 55.0, 'b': 80.0, 'minShot': 3}); assert st == 200
    t0 = time.time()
    while True:
        st, h, b = req('/api/gaze/status'); G = json.loads(b)
        if G.get('state') in ('done', 'error', 'choose_person') and not G.get('running'): break
        if time.time() - t0 > 300: raise SystemExit('gaze timeout')
        time.sleep(1)
    assert G['state'] == 'done', G
    st, h, b = req('/api/gaze/result?a=55.0&b=80.0'); R = json.loads(b); print('gaze job:', G['message'], '| cuts:', len(R.get('cuts', [])), '| cams:', R.get('cams'))
    assert len(R.get('cuts', [])) == 0 and 'samples' in R
    # ---- B-roll: offline suggestion job (keyword fallback + placeholder image), then a render with a full-frame and a PiP image
    json.dump({'model': 'test', 'ranges': [{'a': 50.0, 'b': 85.0}], 'words': words}, open(os.path.join(ld, '_multicam', 'words.json'), 'w'))
    words_b = [{'s': round(56.0 + 0.5 * i, 3), 'e': round(56.3 + 0.5 * i, 3), 't': t, 'p': 0.9} for i, t in enumerate('we used vacuum tubes back then'.split())]
    words_b += [{'s': round(76.0 + 0.5 * i, 3), 'e': round(76.3 + 0.5 * i, 3), 't': t, 'p': 0.9} for i, t in enumerate('and the floppy disk followed'.split())]   # >= 20 s apart -> two suggestions
    json.dump({'model': 'test', 'ranges': [{'a': 50.0, 'b': 85.0}], 'words': words_b}, open(os.path.join(ld, '_multicam', 'words.json'), 'w'))
    st, h, b = req('/api/broll/suggest', 'POST', {'clipId': 'c_test01', 'offline': True, 'llm': False}); assert st == 200
    t0 = time.time()
    while True:
        st, h, b = req('/api/broll/status'); BS = json.loads(b)
        if BS.get('state') in ('done', 'error') and not BS.get('running'): break
        if time.time() - t0 > 120: raise SystemExit('broll timeout')
        time.sleep(1)
    assert BS['state'] == 'done', BS
    st, h, b = req('/api/broll/list/c_test01'); BD = json.loads(b); print('broll:', BS['message'], '|', [(i['query'], i['candidates'][0].get('src')) for i in BD['items']])
    assert len(BD['items']) >= 2 and all(i['candidates'] and i['candidates'][0].get('src') for i in BD['items'])
    st, h, b = req('/api/broll/next', 'POST', {'clipId': 'c_test01', 'itemId': BD['items'][0]['id']})   # offline: openverse fails -> cycles within existing
    st, h, b = req('/api/broll/add', 'POST', {'clipId': 'c_test01', 't': 68.5, 'query': 'ENIAC', 'duration': 3, 'offline': True}); AD = json.loads(b); assert st == 200 and AD['item']['candidates'][0].get('src'), AD
    st, h, b = req('/api/broll/list/c_test01'); assert any(i['query'] == 'ENIAC' and i['source'] == 'manual' for i in json.loads(b)['items']); print('manual image request OK (ENIAC card added at 68.5 s)')
    st, h, b = req('/api/broll/item', 'POST', {'clipId': 'c_test01', 'itemId': BD['items'][1]['id'], 'status': 'ignored'}); assert st == 200
    cuts2 = json.load(open(os.path.join(ld, '_multicam', 'cuts.json')))
    i0 = BD['items'][0]
    cuts2['images'] = [{'id': 'im1', 'sugId': i0['id'], 'clipId': 'c_test01', 'a': 57.0, 'b': 59.0, 'src': i0['candidates'][0]['src'], 'mode': 'full', 'query': i0['query'], 'credit': 'Test credit · CC BY 4.0', 'zoom': 1.0},   # zoom OFF
                      {'id': 'im2', 'sugId': 'x', 'clipId': 'c_test01', 'a': 76.0, 'b': 78.0, 'src': i0['candidates'][0]['src'], 'mode': 'pip', 'pos': 'bl', 'size': 0.25, 'query': 'pip test', 'credit': '', 'zoom': 1.0}]   # PiP bottom-left, 25 %
    st, h, b = req('/api/cuts', 'PUT', cuts2); assert st == 200
    st, h, b = req('/api/tighten', 'PUT', {'removals': [{'a': 73.0, 'b': 75.0, 'kind': 'words', 'label': 'x'}], 'corrections': {}, 'edgeSnap': 3, 'startLead': 0, 'endHold': 0}); assert st == 200   # known tightening state
    st, h, b = req('/api/render', 'POST', {'clip': 'History of memory'}); assert st == 200
    t0 = time.time()
    while True:
        st, h, b = req('/api/render/status'); S = json.loads(b)
        if S.get('state') in ('done', 'error') and not S.get('running'): break
        if time.time() - t0 > 600: raise SystemExit('broll render timeout')
        time.sleep(1)
    assert S['state'] == 'done', open(os.path.join(ld, '_multicam', 'logs', 'render.log')).read()[-2500:]
    pr = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', cout], capture_output=True, text=True).stdout)
    v = [s for s in pr['streams'] if s['codec_type'] == 'video'][0]
    exp_frames = round((25.0 - 6.0 - 2.0 + float((S.get('joins') or {}).get('added_s', 0)) + 0.8) * 30)
    assert int(v['nb_frames']) == exp_frames, (v['nb_frames'], exp_frames)   # images replace, never add, time
    # the frame inside the full-frame image window must NOT look like the camera test pattern any more
    out_img = frame(cout, 57.5 - 55.0 + 0.3, 'scale=192:108'); cam_ref = frame(os.path.join(ld, 'camA.mp4'), 57.5 - 10.0, 'scale=1920:1080,scale=192:108')
    print(f'image piece check: diff vs camera {np.abs(out_img - cam_ref).mean():.1f} (must be large)'); assert np.abs(out_img - cam_ref).mean() > 20
    txt = open(os.path.join(ld, 'clips', 'History of memory.txt')).read(); assert 'Images / credits' in txt and 'Test credit' in txt, txt[-300:]
    # PiP bottom-left: the bottom-left region of a frame inside im2 must differ from the camera, the top-right must not
    pip_t = 76.5 - 55.0 + 0.3 + float((S.get('joins') or {}).get('added_s', 0)) - 6.0   # output time of master 76.5 (skip 60-66 removed)
    out_pip = frame(cout, pip_t, 'scale=192:108'); cam_pip = frame(os.path.join(ld, 'camA.mp4'), 76.5 - 10.0, 'scale=1920:1080,scale=192:108')
    o = out_pip.reshape(108, 192); cam = cam_pip.reshape(108, 192)
    bl = np.abs(o[70:104, 4:52] - cam[70:104, 4:52]).mean(); tr = np.abs(o[4:40, 140:188] - cam[4:40, 140:188]).mean()
    print(f'PiP position check: bottom-left diff {bl:.1f} (image there) vs top-right diff {tr:.1f} (camera there)'); assert bl > 15 and tr < 8, (bl, tr)
    print('broll render OK (full-frame + PiP pieces, credits in txt)')
    print('SELFTEST PASSED')
except BaseException:
    print('\n--- server log (last 60 lines) ---')
    try:
        print(''.join(open(SRV_LOG, errors='replace').readlines()[-60:]))
    except OSError:
        pass
    raise
finally:
    srv.terminate()
