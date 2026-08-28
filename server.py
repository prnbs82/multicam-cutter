"""Local HTTP server: static web app, range-capable media, JSON API."""
import json, os, re, subprocess, sys, threading, time, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from common import TOOL_DIR, load_json, save_json, work_dir
from vtt import parse_vtt

CTYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.u8': 'application/octet-stream',
          '.mp4': 'video/mp4', '.json': 'application/json', '.svg': 'image/svg+xml'}


def make_handler(lecture_dir):
    """Build the request-handler class bound to one lecture folder (routes, background jobs, locks live in this closure)."""
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    lock = threading.Lock()
    state = {'render_proc': None, 'transcribe_proc': None, 'topics_proc': None}
    ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,40}$')
    TIGHTEN = os.path.join(wd, 'tighten.json')

    def migrate_tighten():
        """Fold legacy per-clip tighten/<id>.json files into the lecture-wide tighten.json."""
        d = os.path.join(wd, 'tighten')
        if not os.path.isdir(d):
            return
        doc = load_json(TIGHTEN, {'removals': [], 'corrections': {}})
        changed = False
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json'):
                continue
            old = load_json(os.path.join(d, fn), {})
            doc['removals'] += old.get('removals', [])
            doc['corrections'].update(old.get('corrections', {}))
            for k in ('pauseThreshold', 'keepBreath', 'fillers'):
                if k in old and k not in doc:
                    doc[k] = old[k]
            os.replace(os.path.join(d, fn), os.path.join(d, fn + '.migrated'))
            changed = True
        if changed:
            doc['removals'].sort(key=lambda r: r['a'])
            save_json(TIGHTEN, doc)
            print(f'migrated per-clip tighten files into {TIGHTEN}')
    migrate_tighten()

    def words_fingerprint(words):
        """Short identity string of a words.json (model, count, first/last time) used to detect stale checkpoints."""
        if not words or not words.get('words'):
            return None
        w = words['words']
        return f"{words.get('model')}|{len(w)}|{w[0]['s']:.3f}|{w[-1]['e']:.3f}"

    def kill_group(proc, grace=8.0):
        """SIGTERM the whole process group (worker + ffmpeg children) and wait until every member is gone; SIGKILL stragglers."""
        import signal
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        t0 = time.time()
        while time.time() - t0 < grace:
            try:
                os.killpg(proc.pid, 0)      # raises when no member of the group is left
            except ProcessLookupError:
                return
            time.sleep(0.2)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def asr_running():
        """True if our subprocess runs, or a transcriber started elsewhere (CLI / before a server restart) updated status recently."""
        proc = state['transcribe_proc']
        if proc is not None and proc.poll() is None:
            return True
        st = load_json(os.path.join(wd, 'transcribe', 'status.json'), {})
        return st.get('state') in ('starting', 'loading', 'transcribing') and time.time() - st.get('time', 0) < 90

    class H(BaseHTTPRequestHandler):
        """HTTP handler: static UI, range-capable media, and the JSON API under /api/*."""
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            """Quieter access log: skip the polling endpoints so the server log stays readable."""
            if '/media/' in (args[0] if args else ''):
                return
            sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

        # ---- helpers
        def send_json(self, obj, code=200):
            """Serialise `obj` as a JSON response with the given status code."""
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path):
            """Serve a file with HTTP Range support (video seeking) and the right content type; 404 if missing."""
            if not os.path.isfile(path):
                return self.send_json({'error': 'not found', 'path': path}, 404)
            size = os.path.getsize(path)
            ctype = CTYPES.get(os.path.splitext(path)[1].lower(), 'application/octet-stream')
            start, end = 0, size - 1
            rng = self.headers.get('Range')
            if rng and rng.startswith('bytes='):
                a, _, b = rng[6:].partition('-')
                if a:
                    start = int(a)
                    if b:
                        end = min(int(b), size - 1)
                elif b:  # suffix range
                    start = max(0, size - int(b))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            else:
                self.send_response(200)
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                with open(path, 'rb') as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def read_body(self):
            """Parse the JSON request body ({} when empty)."""
            n = int(self.headers.get('Content-Length') or 0)
            return json.loads(self.rfile.read(n) or b'{}')

        # ---- routes
        def do_GET(self):
            """Read-only API: project/layout, cuts, tighten (migrated), words, envelope, job statuses, topics, keys, AI/hardware, capacity, B-roll lists, checkpoints, media files."""
            p = urlparse(self.path).path
            if p == '/':
                return self.send_file(os.path.join(TOOL_DIR, 'web', 'index.html'))
            if p.startswith('/media/'):
                rel = p[len('/media/'):]
                parts = [x for x in rel.split('/') if x and x not in ('.', '..')]
                if len(parts) == 2 and parts[0] in ('gaze', 'broll'):
                    return self.send_file(os.path.join(wd, parts[0], parts[1]))
                return self.send_file(os.path.join(wd, os.path.basename(rel)))
            if p == '/api/project':
                proj = load_json(os.path.join(wd, 'project.json'))
                layout = load_json(os.path.join(wd, 'layout.json'))
                if not layout:
                    return self.send_json({'error': 'layout.json missing - run `multicam.py proxy` first'}, 503)
                cues = parse_vtt(os.path.join(ld, proj['transcript'])) if proj.get('transcript') else []
                cuts = load_json(os.path.join(wd, 'cuts.json'), None)
                return self.send_json({'name': os.path.basename(ld), 'layout': layout, 'cues': cues, 'cuts': cuts,
                                       'has_multiview': os.path.exists(os.path.join(wd, 'multiview.mp4'))})
            if p == '/api/words':
                from transcribe import load_words, coverage_missing, words_in
                q = parse_qs(urlparse(self.path).query)
                a, b = float(q.get('a', [0])[0]), float(q.get('b', [0])[0])
                doc = load_words(wd)
                return self.send_json({'covered': not coverage_missing(doc['ranges'], a, b), 'model': doc.get('model'),
                                       'ranges': doc['ranges'], 'words': words_in(doc, a, b)})
            if p == '/api/transcribe/status':
                st = load_json(os.path.join(wd, 'transcribe', 'status.json'), {'state': 'idle'})
                proc = state['transcribe_proc']
                st['running'] = asr_running()
                if not st['running'] and st.get('state') in ('starting', 'loading', 'transcribing'):
                    st['state'] = 'error'
                    st['message'] = 'transcription stopped unexpectedly (see logs/transcribe.log)'
                if proc is not None and proc.poll() not in (None, 0) and st.get('state') not in ('done', 'error', 'cancelled'):
                    st['state'] = 'error'
                    st['message'] = f'transcription exited with code {proc.poll()} (see logs/transcribe.log)'
                return self.send_json(st)
            if p == '/api/tighten':
                from joins import migrate_tighten
                from transcribe import load_words
                import envelope as envmod
                doc, stats = migrate_tighten(load_json(TIGHTEN, {'removals': [], 'corrections': {}}), load_words(wd)['words'], envmod.load(wd))
                doc['migration'] = stats
                return self.send_json(doc)
            if p == '/api/envelope':
                import envelope as envmod
                try:
                    path = envmod.build(wd)
                except Exception as e:
                    return self.send_json({'error': str(e)}, 503)
                return self.send_file(path)
            if p == '/api/workspace':
                return self.send_json(load_json(os.path.join(wd, 'workspace.json'), {}))
            if p == '/api/checkpoints':
                d = os.path.join(wd, 'checkpoints')
                items = []
                if os.path.isdir(d):
                    for fn in sorted(os.listdir(d), reverse=True):
                        if fn.endswith('.json'):
                            meta = load_json(os.path.join(d, fn), {}).get('meta', {})
                            items.append({'file': fn, **meta})
                return self.send_json({'checkpoints': items})
            if p == '/api/topics':
                return self.send_json(load_json(os.path.join(wd, 'topics.json'), {'segments': [], 'topics': [], 'recurrences': []}))
            if p == '/api/keys':
                from hw import load_config
                k = {**(load_json(os.path.join(wd, 'keys.json'), {}) or {}), **(load_config('keys.json', {}) or {})}
                return self.send_json({name: (v[:4] + '…' + v[-4:] if isinstance(v, str) and len(v) > 8 else ('set' if v else '')) for name, v in k.items()})
            if p == '/api/ai':
                from llm import provider_status
                return self.send_json(provider_status())
            if p == '/api/hw':
                from hw import report as hw_report
                return self.send_json(hw_report(refresh='refresh' in parse_qs(urlparse(self.path).query)))
            if p == '/api/capacity':
                from capacity import report
                return self.send_json(report(ld))
            if p == '/api/broll/status':
                st = load_json(os.path.join(wd, 'broll', 'status.json'), {'state': 'idle'})
                proc = state.get('broll_proc')
                st['running'] = proc is not None and proc.poll() is None
                if proc is not None and proc.poll() not in (None, 0) and st.get('state') not in ('done', 'error'):
                    st['state'] = 'error'
                    st['message'] = f'image search exited with code {proc.poll()} (see logs/broll.log)'
                return self.send_json(st)
            if p.startswith('/api/broll/list/'):
                cid = p[len('/api/broll/list/'):]
                if not ID_RE.match(cid):
                    return self.send_json({'error': 'bad id'}, 400)
                return self.send_json(load_json(os.path.join(wd, 'broll', f'suggest_{cid}.json'), {}))
            if p == '/api/gaze/status':
                st = load_json(os.path.join(wd, 'gaze', 'status.json'), {'state': 'idle'})
                proc = state.get('gaze_proc')
                st['running'] = proc is not None and proc.poll() is None
                if proc is not None and proc.poll() not in (None, 0) and st.get('state') not in ('done', 'error', 'choose_person'):
                    st['state'] = 'error'
                    st['message'] = f'gaze analysis exited with code {proc.poll()} (see logs/gaze.log)'
                return self.send_json(st)
            if p == '/api/gaze/result':
                q = parse_qs(urlparse(self.path).query)
                from gaze import key_for
                key = key_for(float(q.get('a', [0])[0]), float(q.get('b', [0])[0]))
                return self.send_json(load_json(os.path.join(wd, 'gaze', key + '.json'), {'error': 'no result', 'key': key}))
            if p == '/api/joins/fixes':
                return self.send_json(load_json(os.path.join(wd, 'joinfix.json'), {}))
            if p == '/api/joins/status':
                st = load_json(os.path.join(wd, 'posematch', 'status.json'), {'state': 'idle'})
                proc = state.get('posematch_proc')
                st['running'] = proc is not None and proc.poll() is None
                if proc is not None and proc.poll() not in (None, 0) and st.get('state') not in ('done', 'error'):
                    st['state'] = 'error'
                    st['message'] = f'pose matching exited with code {proc.poll()} (see logs/posematch.log)'
                return self.send_json(st)
            if p == '/api/topics/status':
                st = load_json(os.path.join(wd, 'topics', 'status.json'), {'state': 'idle'})
                proc = state['topics_proc']
                st['running'] = proc is not None and proc.poll() is None
                if proc is not None and proc.poll() not in (None, 0) and st.get('state') not in ('done', 'error'):
                    st['state'] = 'error'
                    st['message'] = f'topic analysis exited with code {proc.poll()} (see logs/topics.log)'
                return self.send_json(st)
            if p == '/api/render/status':
                st = load_json(os.path.join(wd, 'render', 'status.json'), {'state': 'idle'})
                proc = state['render_proc']
                st['running'] = proc is not None and proc.poll() is None
                if proc is not None and proc.poll() is not None and st.get('state') not in ('done', 'error', 'cancelled'):
                    st['state'] = 'error'
                    st['message'] = f'render process exited with code {proc.poll()} (see logs/render.log)'
                return self.send_json(st)
            self.send_json({'error': 'not found'}, 404)

        def do_PUT(self):
            """Save endpoints: cuts, tighten, keys, AI settings, encoder choice, workspace (atomic JSON writes under the lock)."""
            p = urlparse(self.path).path
            if p == '/api/cuts':
                data = self.read_body()
                with lock:
                    save_json(os.path.join(wd, 'cuts.json'), data)
                return self.send_json({'ok': True})
            if p == '/api/tighten':
                data = self.read_body()
                with lock:
                    save_json(TIGHTEN, data)
                return self.send_json({'ok': True})
            if p == '/api/keys':                 # PUT {name: value} — empty value removes the key (stored per user, not per lecture)
                data = self.read_body()
                from hw import load_config, save_config
                with lock:
                    k = load_config('keys.json', {}) or {}
                    legacy = load_json(os.path.join(wd, 'keys.json'), {}) or {}
                    for name, v in data.items():
                        name = re.sub(r'[^a-z0-9_]', '', str(name).lower())
                        if v:
                            k[name] = str(v).strip()
                        else:
                            k.pop(name, None); legacy.pop(name, None)
                    save_config('keys.json', k)
                    if os.path.exists(os.path.join(wd, 'keys.json')):
                        save_json(os.path.join(wd, 'keys.json'), legacy)
                return self.send_json({'ok': True})
            if p == '/api/ai':                   # PUT {provider, base_url, model, api_key?}  (api_key omitted = keep, '' = remove)
                data = self.read_body()
                from llm import save as ai_save
                with lock:
                    st = ai_save(str(data.get('provider') or 'auto'), str(data.get('base_url') or ''), str(data.get('model') or ''),
                                 None if 'api_key' not in data else str(data.get('api_key') or ''))
                return self.send_json(st)
            if p == '/api/hw':                   # PUT {encoder: auto|libx264|h264_nvenc|...}
                data = self.read_body()
                from hw import set_encoder, report as hw_report
                with lock:
                    set_encoder(str(data.get('encoder') or 'auto'))
                return self.send_json(hw_report())
            if p == '/api/workspace':
                data = self.read_body()
                with lock:
                    save_json(os.path.join(wd, 'workspace.json'), data)
                return self.send_json({'ok': True})
            self.send_json({'error': 'not found'}, 404)

        def do_POST(self):
            """Actions: start/cancel background jobs (render, transcribe, topics, gaze, joins, B-roll), checkpoints, AI test, project setup."""
            p = urlparse(self.path).path
            def ram_guard(what):
                """Refuse to start an analysis when free RAM is below the streaming threshold (HTTP 507 with an explanation)."""
                from capacity import report
                rep = report(ld)
                if not rep['analysis_ok_now']:
                    return self.send_json({'error': f'not enough free RAM for {what} right now ({rep["ram_available_gb"]} GB available) — close other programs and retry'}, 507)
                return None
            if p in ('/api/gaze', '/api/joins/refine', '/api/broll/suggest'):
                g = ram_guard({'/api/gaze': 'gaze analysis', '/api/joins/refine': 'join measurement', '/api/broll/suggest': 'image search'}[p])
                if g is not None:
                    return g
            if p == '/api/ai/test':              # POST {} -> round-trip a tiny prompt through the configured AI provider
                from llm import test as ai_test
                return self.send_json(ai_test())
            if p == '/api/render':
                body = self.read_body()
                with lock:
                    proc = state['render_proc']
                    if proc is not None and proc.poll() is None:
                        return self.send_json({'error': 'render already running'}, 409)
                    os.makedirs(os.path.join(wd, 'logs'), exist_ok=True)
                    save_json(os.path.join(wd, 'render', 'status.json'), {'state': 'starting', 'progress': 0, 'message': 'starting'})
                    from capacity import report
                    cmd = [sys.executable, os.path.join(TOOL_DIR, 'multicam.py'), 'render', ld, '--workers', str(report(ld)['recommended_workers'])]
                    if body.get('out'):
                        cmd += ['--out', body['out']]
                    if body.get('clip'):
                        cmd += ['--clip', body['clip']]
                        if body.get('tighten'):
                            cmd += ['--tighten']
                    log = open(os.path.join(wd, 'logs', 'render.log'), 'w')
                    state['render_proc'] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                return self.send_json({'ok': True})
            if p == '/api/transcribe':
                body = self.read_body()
                with lock:
                    if asr_running():
                        return self.send_json({'error': 'transcription already running'}, 409)
                    os.makedirs(os.path.join(wd, 'logs'), exist_ok=True)
                    cmd = [sys.executable, os.path.join(TOOL_DIR, 'multicam.py'), 'transcribe', ld, '--model', str(body.get('model') or 'auto')]
                    if not body.get('all'):
                        cmd += ['--range', str(float(body['a'])), str(float(body['b']))]
                    save_json(os.path.join(wd, 'transcribe', 'status.json'), {'state': 'starting', 'progress': 0, 'message': 'starting'})
                    log = open(os.path.join(wd, 'logs', 'transcribe.log'), 'a')
                    state['transcribe_proc'] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                return self.send_json({'ok': True})
            if p == '/api/checkpoints':
                body = self.read_body()
                name = re.sub(r'[^\w\- ]+', '_', str(body.get('name') or 'checkpoint')).strip()[:60] or 'checkpoint'
                stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                d = os.path.join(wd, 'checkpoints')
                with lock:
                    cuts = load_json(os.path.join(wd, 'cuts.json'), {})
                    tight = load_json(TIGHTEN, {})
                    words = load_json(os.path.join(wd, 'words.json'), None)
                    doc = {'meta': {'name': name, 'time': time.time(), 'stamp': stamp,
                                    'cuts': len(cuts.get('cuts', [])), 'clips': len(cuts.get('clips', [])),
                                    'removals': len(tight.get('removals', [])), 'words_fingerprint': words_fingerprint(words)},
                           'cuts': cuts, 'tighten': tight, 'words': words}
                    fn = f'{stamp}_{name}.json'
                    save_json(os.path.join(d, fn), doc)
                return self.send_json({'ok': True, 'file': fn, **doc['meta']})
            if p == '/api/checkpoints/restore':
                body = self.read_body()
                fn = os.path.basename(str(body.get('file', '')))
                path = os.path.join(wd, 'checkpoints', fn)
                if not fn.endswith('.json') or not os.path.exists(path):
                    return self.send_json({'error': 'checkpoint not found'}, 404)
                with lock:
                    doc = load_json(path, {})
                    # safety net: snapshot the current state before overwriting it
                    cur = {'meta': {'name': 'before restore', 'time': time.time(), 'stamp': time.strftime('%Y-%m-%d_%H-%M-%S')},
                           'cuts': load_json(os.path.join(wd, 'cuts.json'), {}), 'tighten': load_json(TIGHTEN, {})}
                    cur['meta'].update(cuts=len(cur['cuts'].get('cuts', [])), clips=len(cur['cuts'].get('clips', [])), removals=len(cur['tighten'].get('removals', [])))
                    cur['words'] = load_json(os.path.join(wd, 'words.json'), None)
                    cur['meta']['words_fingerprint'] = words_fingerprint(cur['words'])
                    save_json(os.path.join(wd, 'checkpoints', f"{cur['meta']['stamp']}_before restore.json"), cur)
                    save_json(os.path.join(wd, 'cuts.json'), doc.get('cuts', {}))
                    save_json(TIGHTEN, doc.get('tighten', {}))
                    note = ''
                    if doc.get('words') and words_fingerprint(doc['words']) != words_fingerprint(cur['words']):
                        save_json(os.path.join(wd, 'words.json'), doc['words'])   # the edits refer to these word timings
                        note = 'words restored from checkpoint (they differed from the current transcript)'
                return self.send_json({'ok': True, 'restored': doc.get('meta', {}), 'note': note})
            if p == '/api/broll/suggest':
                body = self.read_body()
                cid = str(body.get('clipId', ''))
                if not ID_RE.match(cid):
                    return self.send_json({'error': 'bad clip id'}, 400)
                with lock:
                    proc = state.get('broll_proc')
                    if proc is not None and proc.poll() is None:
                        return self.send_json({'error': 'image search already running'}, 409)
                    os.makedirs(os.path.join(wd, 'logs'), exist_ok=True)
                    cmd = [sys.executable, os.path.join(TOOL_DIR, 'multicam.py'), 'broll', ld, '--clip', cid] + (['--offline'] if body.get('offline') else []) + (['--no-llm'] if body.get('llm') is False else [])
                    save_json(os.path.join(wd, 'broll', 'status.json'), {'state': 'starting', 'progress': 0, 'message': 'starting', 'clipId': cid})
                    log = open(os.path.join(wd, 'logs', 'broll.log'), 'a')
                    state['broll_proc'] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                return self.send_json({'ok': True})
            if p == '/api/broll/add':           # {clipId, t, query, phrase?, w0?, w1?, duration?}  (synchronous search + download, a few seconds)
                body = self.read_body()
                from broll import add_manual
                try:
                    it = add_manual(ld, str(body['clipId']), float(body['t']), str(body['query']).strip(), body.get('phrase'), body.get('w0'), body.get('w1'), float(body.get('duration') or 4.0), offline=bool(body.get('offline')))
                except Exception as e:
                    return self.send_json({'error': str(e)}, 400)
                return self.send_json({'ok': True, 'item': it})
            if p == '/api/broll/item':          # {clipId, itemId, status?, duration?}
                body = self.read_body()
                from broll import set_status
                try:
                    it = set_status(ld, str(body['clipId']), str(body['itemId']), body.get('status', 'new'), body.get('duration'))
                except Exception as e:
                    return self.send_json({'error': str(e)}, 400)
                return self.send_json({'ok': True, 'item': it})
            if p == '/api/broll/next':
                body = self.read_body()
                from broll import next_candidate
                try:
                    it = next_candidate(ld, str(body['clipId']), str(body['itemId']))
                except Exception as e:
                    return self.send_json({'error': str(e)}, 400)
                return self.send_json({'ok': True, 'item': it})
            if p == '/api/broll/upload':        # {clipId, itemId, name, data: base64}
                body = self.read_body()
                import base64
                from broll import upload_own
                try:
                    raw = base64.b64decode(body['data'].split(',', 1)[-1])
                    it = upload_own(ld, str(body['clipId']), str(body['itemId']), raw, body.get('name') or 'own image')
                except Exception as e:
                    return self.send_json({'error': str(e)}, 400)
                return self.send_json({'ok': True, 'item': it})
            if p == '/api/gaze':
                body = self.read_body()
                with lock:
                    proc = state.get('gaze_proc')
                    if proc is not None and proc.poll() is None:
                        return self.send_json({'error': 'gaze analysis already running'}, 409)
                    os.makedirs(os.path.join(wd, 'logs'), exist_ok=True)
                    cmd = [sys.executable, os.path.join(TOOL_DIR, 'multicam.py'), 'gaze', ld, '--range', str(float(body['a'])), str(float(body['b'])),
                           '--min-shot', str(float(body.get('minShot', 3.0)))]
                    if body.get('person'):
                        cmd += ['--person', json.dumps(body['person'])]
                    save_json(os.path.join(wd, 'gaze', 'status.json'), {'state': 'starting', 'progress': 0, 'message': 'starting'})
                    log = open(os.path.join(wd, 'logs', 'gaze.log'), 'a')
                    state['gaze_proc'] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                return self.send_json({'ok': True})
            if p == '/api/joins/refine':
                body = self.read_body()
                with lock:
                    proc = state.get('posematch_proc')
                    if proc is not None and proc.poll() is None:
                        return self.send_json({'error': 'pose matching already running'}, 409)
                    os.makedirs(os.path.join(wd, 'logs'), exist_ok=True)
                    cmd = [sys.executable, os.path.join(TOOL_DIR, 'multicam.py'), 'posematch', ld] + (['--force'] if body.get('force') else [])
                    save_json(os.path.join(wd, 'posematch', 'status.json'), {'state': 'starting', 'progress': 0, 'message': 'starting'})
                    log = open(os.path.join(wd, 'logs', 'posematch.log'), 'a')
                    state['posematch_proc'] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                return self.send_json({'ok': True})
            if p == '/api/topics':
                body = self.read_body()
                with lock:
                    proc = state['topics_proc']
                    if proc is not None and proc.poll() is None:
                        return self.send_json({'error': 'topic analysis already running'}, 409)
                    os.makedirs(os.path.join(wd, 'logs'), exist_ok=True)
                    cmd = [sys.executable, os.path.join(TOOL_DIR, 'multicam.py'), 'topics', ld, '--granularity', str(float(body.get('granularity', 0.5)))]
                    if body.get('llm') is False:
                        cmd += ['--no-llm']
                    save_json(os.path.join(wd, 'topics', 'status.json'), {'state': 'starting', 'progress': 0, 'message': 'starting'})
                    log = open(os.path.join(wd, 'logs', 'topics.log'), 'a')
                    state['topics_proc'] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                return self.send_json({'ok': True})
            if p == '/api/transcribe/cancel':
                with lock:
                    proc = state['transcribe_proc']
                    if proc is None or proc.poll() is not None:
                        return self.send_json({'error': 'no transcription running'}, 409)
                    kill_group(proc)
                    st = load_json(os.path.join(wd, 'transcribe', 'status.json'), {})
                    st.update(state='cancelled', message='cancelled by user', progress=0)
                    save_json(os.path.join(wd, 'transcribe', 'status.json'), st)
                    state['transcribe_proc'] = None
                return self.send_json({'ok': True})
            if p == '/api/render/cancel':
                with lock:
                    proc = state['render_proc']
                    if proc is None or proc.poll() is not None:
                        return self.send_json({'error': 'no render running'}, 409)
                    kill_group(proc)
                    st = load_json(os.path.join(wd, 'render', 'status.json'), {})
                    st.update(state='cancelled', message='cancelled by user', progress=0)
                    save_json(os.path.join(wd, 'render', 'status.json'), st)
                    state['render_proc'] = None
                return self.send_json({'ok': True})
            self.send_json({'error': 'not found'}, 404)

    return H


def serve(lecture_dir, port=8765, open_browser=False):
    """Run the threaded HTTP server on 127.0.0.1:port for one lecture folder until Ctrl+C."""
    handler = make_handler(lecture_dir)
    httpd = ThreadingHTTPServer(('127.0.0.1', port), handler)
    url = f'http://127.0.0.1:{port}/'
    print(f'Multicam cutter serving {os.path.abspath(lecture_dir)}\n  -> {url}\n(Ctrl+C to stop)')
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
