"""B-roll suggestions per clip: image-ready moments (Claude via the local CLI, keyword fallback) and one relevant, freely
licensed image per moment (Wikipedia page image -> Wikimedia Commons search -> Openverse), downloaded with its credit.

Files: _multicam/broll/suggest_<clipId>.json, _multicam/broll/<hash>.jpg (+ .json sidecar), _multicam/broll/status.json
"""
import hashlib, io, json, os, re, time, urllib.error, urllib.parse, urllib.request
from common import load_json, save_json, work_dir

UA = 'MulticamCutter/1.0 (lecture editing tool; contact: local user)'
MAX_W = 1280   # a standard Wikimedia thumbnail width (others are rate-limited: https://w.wiki/GHai)
KEYWORDS = ['vacuum tube', 'vacuum tubes', 'transistor', 'transistors', 'eniac', 'floppy disk', 'floppy', 'dram', 'nand', 'flash memory',
            'hbm', 'gpu', 'cpu', 'wafer', 'fab', 'samsung', 'micron', 'sk hynix', 'punch card', 'punch cards', 'magnetic tape', 'von neumann',
            'intel', 'nvidia', 'tsmc', 'mainframe', 'microprocessor', 'sram', 'cache', 'motherboard', 'silicon', 'integrated circuit', 'chip',
            'relay', 'abacus', 'turing', 'babbage', 'hard disk', 'ssd', 'data center', 'server rack', 'robot', 'sensor', 'camera']


def _status(wd, **kw):
    save_json(os.path.join(wd, 'broll', 'status.json'), {**kw, 'time': time.time()})


_BLOCKED = {}    # host -> until (epoch): hosts that answered 429 are avoided for a while instead of waiting on retries


def _host(url):
    return urllib.parse.urlparse(url).netloc.lower()


def host_blocked(url):
    return _BLOCKED.get(_host(url), 0) > time.time()


def _get(url, timeout=25, tries=2):
    """GET; a 429 marks the host as throttled for 10 minutes and fails fast so callers move on to another candidate/host."""
    if host_blocked(url):
        raise RuntimeError(f'{_host(url)} is rate-limiting us right now (retry in a few minutes)')
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json,*/*'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                _BLOCKED[_host(url)] = time.time() + 600
                raise RuntimeError(f'{_host(url)} is rate-limiting us (429); using other sources for 10 min')
            if e.code not in (500, 502, 503, 504):
                raise
            time.sleep(1.0)
        except Exception as e:
            last = e
            time.sleep(0.5)
    raise last


def _api(url):
    return json.loads(_get(url).decode('utf-8', 'replace'))


# ------------------------------------------------------------------ 1. image-ready moments
def suggest_moments(words, a, b, use_llm=True):
    """words: the clip's words (dicts with s,e,t). Returns [{w0,w1,phrase,query,why,duration}] (indices into `words`)."""
    if not words:
        return []
    items = None
    from llm import ask_json as claude_json, provider_status, source_name
    if use_llm and provider_status()['available']:
        numbered = ' '.join(f'{i}:{w["t"]}' for i, w in enumerate(words))
        prompt = (
            "You are helping edit a university lecture clip into a short video. Below is the clip's transcript as numbered words "
            "(index:word). Find the moments where showing a PICTURE would help the viewer: concrete, visual things the speaker names — "
            "devices, machines, components, people, places, companies/products, historical computers, materials. Never abstract concepts "
            "(e.g. 'bottleneck', 'performance', 'idea'). At most one suggestion per ~25 seconds of speech (about 60 words), prefer the most "
            "vivid ones, and skip things the speaker only mentions in passing. For each give the word index range that names the thing, the "
            "phrase, a short image-search query (a Wikipedia article title works best, e.g. 'Vacuum tube', 'ENIAC', 'Floppy disk', "
            "'High Bandwidth Memory'), why it helps (max 12 words) and a duration in seconds (3-6). Respond with ONLY a JSON object: "
            '{"items":[{"w0":<int>,"w1":<int>,"phrase":"...","query":"...","why":"...","duration":<number>}]}\n\n' + numbered)
        got = claude_json(prompt)
        if got and isinstance(got.get('items'), list):
            items = []
            for it in got['items']:
                try:
                    w0, w1 = int(it['w0']), int(it['w1'])
                except Exception:
                    continue
                if not (0 <= w0 <= w1 < len(words)):
                    continue
                items.append({'w0': w0, 'w1': min(w1, w0 + 6), 'phrase': str(it.get('phrase') or ' '.join(w['t'] for w in words[w0:w1 + 1]))[:80],
                              'query': str(it.get('query') or it.get('phrase') or '')[:80], 'why': str(it.get('why') or '')[:120],
                              'duration': float(max(2.5, min(8.0, float(it.get('duration') or 4)))), 'source': source_name()})
    if items is None:                                   # keyword fallback
        items = []
        low = [re.sub(r'[^a-z0-9 ]', '', w['t'].lower()) for w in words]
        i = 0
        while i < len(words):
            hit = None
            for kw in KEYWORDS:
                parts = kw.split()
                if low[i:i + len(parts)] == parts:
                    hit = (kw, len(parts)); break
            if hit:
                items.append({'w0': i, 'w1': i + hit[1] - 1, 'phrase': ' '.join(w['t'] for w in words[i:i + hit[1]]), 'query': hit[0].title(),
                              'why': 'named a concrete object', 'duration': 4.0, 'source': 'keywords'})
                i += hit[1]
                continue
            # proper nouns: a run of capitalised words that is not the start of a sentence (e.g. "Von Neumann", "IBM", "Intel Core")
            prev_end = i == 0 or words[i - 1]['t'].strip()[-1:] in '.?!'
            tok = words[i]['t'].strip().strip('.,;:!?"\'()')
            if not prev_end and len(tok) > 2 and tok[0].isupper() and tok.lower() not in ('i', 'okay', 'ok') and not tok.isdigit():
                j = i
                while j + 1 < len(words) and j - i < 3:
                    nx = words[j + 1]['t'].strip().strip('.,;:!?"\'()')
                    if nx[:1].isupper() and len(nx) > 1 and words[j]['t'].strip()[-1:] not in '.?!':
                        j += 1
                    else:
                        break
                phrase = ' '.join(w['t'].strip().strip('.,;:!?"\'()') for w in words[i:j + 1])
                items.append({'w0': i, 'w1': j, 'phrase': phrase, 'query': phrase, 'why': 'a name — probably a person, place, company or product',
                              'duration': 4.0, 'source': 'keywords'})
                i = j + 1
                continue
            i += 1
    # thin out: keep >= 20 s apart, earliest first
    items.sort(key=lambda x: x['w0'])
    out, last_t = [], -1e9
    for it in items:
        t = words[it['w0']]['s']
        if t - last_t >= 20.0:
            out.append(it); last_t = t
    return out


# ------------------------------------------------------------------ 2. images
def _commons_fileinfo(file_title):
    q = urllib.parse.quote(file_title)
    d = _api(f'https://commons.wikimedia.org/w/api.php?action=query&titles={q}&prop=imageinfo&iiprop=url|extmetadata&iiurlwidth={MAX_W}&format=json')
    for p in d.get('query', {}).get('pages', {}).values():
        ii = (p.get('imageinfo') or [None])[0]
        if not ii:
            continue
        md = ii.get('extmetadata', {})
        g = lambda k: re.sub(r'<[^>]+>', '', md.get(k, {}).get('value', '') or '').strip()
        return {'src_url': ii.get('thumburl') or ii.get('url'), 'orig_url': ii.get('url'), 'page_url': ii.get('descriptionurl'), 'artist': g('Artist')[:80],
                'license': g('LicenseShortName') or g('License') or 'unknown', 'license_url': g('LicenseUrl'), 'title': p.get('title', '')}
    return None


def wikipedia_candidates(query, n=3):
    out = []
    try:
        d = _api('https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=' + urllib.parse.quote(query) + f'&gsrlimit={n}&prop=pageimages|pageprops|info&inprop=url&piprop=original|name&format=json')
    except Exception as e:
        print('wikipedia failed:', e); return out
    pages = sorted(d.get('query', {}).get('pages', {}).values(), key=lambda p: p.get('index', 99))
    for p in pages:
        fname = (p.get('pageprops') or {}).get('page_image_free') or p.get('pageimage')
        if not fname:
            continue
        try:
            info = _commons_fileinfo('File:' + fname)
        except Exception as e:
            info = None
        if not info or not info.get('src_url'):
            continue
        out.append({'provider': 'wikipedia', 'title': p.get('title', ''), 'src_url': info['src_url'], 'orig_url': info.get('orig_url'), 'source_url': p.get('fullurl') or info.get('page_url'),
                    'license': info['license'], 'artist': info['artist'], 'credit': _credit(p.get('title', ''), info['artist'], info['license'], 'Wikimedia Commons')})
    return out


def commons_candidates(query, n=4):
    out = []
    try:
        d = _api('https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=' + urllib.parse.quote(query + ' filetype:bitmap') + f'&gsrnamespace=6&gsrlimit={n}&prop=imageinfo&iiprop=url|extmetadata&iiurlwidth={MAX_W}&format=json')
    except Exception as e:
        print('commons failed:', e); return out
    for p in sorted(d.get('query', {}).get('pages', {}).values(), key=lambda p: p.get('index', 99)):
        ii = (p.get('imageinfo') or [None])[0]
        if not ii or not (ii.get('thumburl') or ii.get('url')):
            continue
        md = ii.get('extmetadata', {})
        g = lambda k: re.sub(r'<[^>]+>', '', md.get(k, {}).get('value', '') or '').strip()
        title = re.sub(r'^File:|\.[a-zA-Z]+$', '', p.get('title', ''))
        out.append({'provider': 'commons', 'title': title, 'src_url': ii.get('thumburl') or ii.get('url'), 'orig_url': ii.get('url'), 'source_url': ii.get('descriptionurl'),
                    'license': g('LicenseShortName') or 'unknown', 'artist': g('Artist')[:80], 'credit': _credit(title, g('Artist')[:80], g('LicenseShortName'), 'Wikimedia Commons')})
    return out


def openverse_candidates(query, n=4):
    out = []
    try:
        d = _api('https://api.openverse.org/v1/images/?q=' + urllib.parse.quote(query) + f'&license_type=commercial&page_size={n}')
    except Exception as e:
        print('openverse failed:', e); return out
    for r in d.get('results', []):
        lic = (r.get('license') or '').upper()
        lic = ('CC ' + lic + ' ' + (r.get('license_version') or '')).strip() if lic and lic not in ('PDM', 'CC0') else lic
        out.append({'provider': 'openverse', 'title': (r.get('title') or '')[:80], 'src_url': r.get('url'), 'source_url': r.get('foreign_landing_url'),
                    'license': lic or 'unknown', 'artist': (r.get('creator') or '')[:80], 'credit': _credit(r.get('title') or '', r.get('creator') or '', lic, r.get('source') or 'Openverse')})
    return out


def get_key(wd, name):
    """Lecture-local keys.json (legacy) -> user config ~/.config/multicam/keys.json -> environment."""
    from hw import load_config
    return ((load_json(os.path.join(wd, 'keys.json'), {}) or {}).get(name) or (load_config('keys.json', {}) or {}).get(name)
            or os.environ.get(name.upper()))


def unsplash_candidates(query, n=5, key=None):
    """Unsplash API (needs an access key). Free-licence photos only (Unsplash+ excluded by the API for non-subscribers)."""
    if not key:
        return []
    out = []
    try:
        req = urllib.request.Request('https://api.unsplash.com/search/photos?query=' + urllib.parse.quote(query) + f'&per_page={n}&orientation=landscape&content_filter=high',
                                     headers={'User-Agent': UA, 'Accept-Version': 'v1', 'Authorization': 'Client-ID ' + key})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode('utf-8', 'replace'))
    except Exception as e:
        print('unsplash failed:', str(e)[:120]); return out
    for r in d.get('results', []):
        if r.get('premium') or r.get('plus'):
            continue
        user = r.get('user') or {}
        name = user.get('name') or user.get('username') or ''
        out.append({'provider': 'unsplash', 'title': (r.get('alt_description') or r.get('description') or query)[:80], 'src_url': (r.get('urls') or {}).get('regular') or (r.get('urls') or {}).get('full'),
                    'source_url': (r.get('links') or {}).get('html', ''), 'license': 'Unsplash License', 'artist': name,
                    'credit': f'Photo by {name} on Unsplash' if name else 'Unsplash', 'download_location': (r.get('links') or {}).get('download_location')})
    return out


def unsplash_track_download(wd, cand):
    """Unsplash API terms: trigger the download endpoint when a photo is actually used."""
    key = get_key(wd, 'unsplash_access_key')
    loc = cand.get('download_location')
    if not key or not loc:
        return
    try:
        req = urllib.request.Request(loc, headers={'User-Agent': UA, 'Accept-Version': 'v1', 'Authorization': 'Client-ID ' + key})
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        print('unsplash download tracking failed:', str(e)[:80])


def _credit(title, artist, license, via):
    parts = [title.strip()] if title else []
    if artist:
        parts.append(artist.strip())
    lic = (license or '').strip()
    if lic and lic.lower() not in ('public domain', 'pdm', 'cc0', 'cc0 1.0', 'pd'):
        parts.append(lic)
    return ' · '.join(p for p in parts if p)[:110] + (f' ({via})' if via else '')


def find_images(query, n=6, wd=None):
    """Candidates from all providers (queried in parallel); ordered Unsplash (if a key is set) -> Wikipedia -> Commons -> Openverse,
    except that hosts currently rate-limiting us are moved to the end so the first downloadable picture comes quickly."""
    from concurrent.futures import ThreadPoolExecutor
    key = get_key(wd, 'unsplash_access_key') if wd else None
    with ThreadPoolExecutor(4) as ex:
        futs = ([ex.submit(unsplash_candidates, query, 4, key)] if key else []) + [ex.submit(wikipedia_candidates, query, 2), ex.submit(commons_candidates, query, 3), ex.submit(openverse_candidates, query, 5)]
        cands = []
        for f in futs:
            try:
                cands += f.result(timeout=20)
            except Exception as e:
                print('provider failed:', str(e)[:80])
    seen, out = set(), []
    for c in cands:
        if not c.get('src_url') or c['src_url'] in seen:
            continue
        seen.add(c['src_url']); out.append(c)
    out.sort(key=lambda c: 1 if host_blocked(c['src_url']) else 0)     # stable: keeps provider order within each group
    return out[:max(n, 6)]


def download_first(wd, cands, max_tries=3):
    """Download the first candidate that works (moving it to the front); returns the index of the downloaded one or None."""
    tried = 0
    for i, c in enumerate(list(cands)):
        if c.get('src'):
            return i
        if host_blocked(c['src_url']):
            continue
        try:
            download(wd, c)
            return i
        except Exception as e:
            print('download failed:', str(e)[:90])
            tried += 1
            if tried >= max_tries:
                break
    return None


def download(wd, cand):
    """Fetch the candidate image once, store as JPEG (<= MAX_W) with a sidecar; returns the relative file name."""
    h = hashlib.sha1(cand['src_url'].encode()).hexdigest()[:12]
    d = os.path.join(wd, 'broll')
    os.makedirs(d, exist_ok=True)
    fn = f'{h}.jpg'
    path = os.path.join(d, fn)
    if not os.path.exists(path):
        from PIL import Image
        try:
            data = _get(cand['src_url'], timeout=40)
        except Exception as e:
            alt = cand.get('orig_url')
            if not alt or alt == cand['src_url']:
                raise
            print(f'thumbnail fetch failed ({e}); using the original file')
            data = _get(alt, timeout=60)
        im = Image.open(io.BytesIO(data)).convert('RGB')
        if im.width > MAX_W:
            im = im.resize((MAX_W, int(im.height * MAX_W / im.width)))
        im.save(path, 'JPEG', quality=90)
        save_json(os.path.join(d, f'{h}.json'), cand)
    cand['src'] = fn
    return fn


# ------------------------------------------------------------------ 3. per-clip job
def run_suggest(lecture_dir, clip_id, use_llm=True, offline=False):
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    from transcribe import load_words, words_in
    cuts = load_json(os.path.join(wd, 'cuts.json'), {})
    clip = next((c for c in cuts.get('clips', []) if c.get('id') == clip_id), None)
    if not clip:
        raise RuntimeError(f'clip {clip_id} not found')
    out_path = os.path.join(wd, 'broll', f'suggest_{clip_id}.json')
    prev = load_json(out_path, {}) or {}
    try:
        _status(wd, state='thinking', progress=0.05, message=f'reading "{clip["name"]}" for image-ready moments', clipId=clip_id)
        words = words_in(load_words(wd), clip['a'], clip['b'])
        moments = suggest_moments(words, clip['a'], clip['b'], use_llm=use_llm and not offline)
        items = []
        for k, m in enumerate(moments):
            w0 = words[m['w0']]
            hid = hashlib.sha1(('%.3f%s' % (w0['s'], m['query'])).encode()).hexdigest()[:8]
            it = {'id': 's_' + hid, 't': round(w0['s'], 3), 'w0': m['w0'], 'w1': m['w1'],
                  'phrase': m['phrase'], 'query': m['query'], 'why': m['why'], 'duration': m['duration'], 'status': 'new', 'candidates': [], 'pick': 0, 'source': m['source']}
            old = next((o for o in prev.get('items', []) if abs(o.get('t', -1) - it['t']) < 0.5), None)
            if old:
                it['status'], it['candidates'], it['pick'] = old.get('status', 'new'), old.get('candidates', []), old.get('pick', 0)
                if old.get('duration'): it['duration'] = old['duration']
            items.append(it)
        _status(wd, state='searching', progress=0.3, message=f'{len(items)} moments — looking for pictures', clipId=clip_id)
        done = [0]
        def work(it):
            if not it['candidates']:
                if offline:
                    it['candidates'] = [{'provider': 'placeholder', 'title': it['query'], 'src_url': 'placeholder://' + it['query'], 'source_url': '', 'license': 'own', 'artist': '', 'credit': ''}]
                    it['candidates'][0]['src'] = _placeholder(wd, it['query'])
                else:
                    try:
                        it['candidates'] = find_images(it['query'], wd=wd)
                    except Exception as e:
                        print('search failed for', it['query'], e)
                        it['candidates'] = []
            if it['candidates'] and not it['candidates'][min(it['pick'], len(it['candidates']) - 1)].get('src'):
                i = download_first(wd, it['candidates'])
                if i is not None:
                    it['pick'] = i
            done[0] += 1
            _status(wd, state='searching', progress=0.3 + 0.65 * done[0] / max(1, len(items)), message=f'{done[0]}/{len(items)}: {it["query"]}', clipId=clip_id)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(work, items))
        doc = {'clipId': clip_id, 'a': clip['a'], 'b': clip['b'], 'name': clip['name'], 'created': time.time(), 'items': items,
               'llm': any(i.get('source') in ('claude', 'llm') for i in items)}
        save_json(out_path, doc)
        with_img = sum(1 for i in items if i['candidates'])
        _status(wd, state='done', progress=1, message=f'{len(items)} suggestions, {with_img} with an image', clipId=clip_id)
        print(f'{len(items)} suggestions ({with_img} with an image) for "{clip["name"]}"')
        return doc
    except Exception as e:
        _status(wd, state='error', progress=0, message=str(e)[-1500:], clipId=clip_id)
        raise


def _placeholder(wd, text):
    from PIL import Image, ImageDraw
    h = hashlib.sha1(('placeholder' + text).encode()).hexdigest()[:12]
    d = os.path.join(wd, 'broll'); os.makedirs(d, exist_ok=True)
    fn = f'{h}.jpg'
    im = Image.new('RGB', (1280, 720), (40, 60, 90)); dr = ImageDraw.Draw(im); dr.text((60, 330), text, fill=(255, 255, 255))
    im.save(os.path.join(d, fn), 'JPEG', quality=85)
    return fn


def next_candidate(lecture_dir, clip_id, item_id):
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    path = os.path.join(wd, 'broll', f'suggest_{clip_id}.json')
    doc = load_json(path, {})
    it = next((i for i in doc.get('items', []) if i['id'] == item_id), None)
    if not it:
        raise RuntimeError('item not found')
    if len(it['candidates']) < 6:
        try:
            more = openverse_candidates(it['query'], 6)
            have = {c['src_url'] for c in it['candidates']}
            it['candidates'] += [c for c in more if c['src_url'] not in have]
        except Exception as e:
            print('more candidates failed:', e)
    if not it['candidates']:
        raise RuntimeError('no images found for this query')
    n = len(it['candidates'])
    for step in range(1, n + 1):
        k = (it['pick'] + step) % n
        c = it['candidates'][k]
        if c.get('src'):
            it['pick'] = k; break
        if host_blocked(c['src_url']):
            continue
        try:
            download(wd, c); it['pick'] = k; break
        except Exception as e:
            print('download failed:', str(e)[:80])
    else:
        raise RuntimeError('no other picture could be fetched right now (sources rate-limiting) — try again in a few minutes or use your own image')
    save_json(path, doc)
    return it


def set_status(lecture_dir, clip_id, item_id, status, duration=None):
    wd = work_dir(os.path.abspath(lecture_dir))
    path = os.path.join(wd, 'broll', f'suggest_{clip_id}.json')
    doc = load_json(path, {})
    it = next((i for i in doc.get('items', []) if i['id'] == item_id), None)
    if not it:
        raise RuntimeError('item not found')
    if status == 'used' and it.get('status') != 'used' and it.get('candidates'):
        c = it['candidates'][min(it['pick'], len(it['candidates']) - 1)]
        if c.get('provider') == 'unsplash':
            unsplash_track_download(wd, c)
    it['status'] = status
    if duration:
        it['duration'] = float(duration)
    save_json(path, doc)
    return it


def upload_own(lecture_dir, clip_id, item_id, data, name='own image'):
    """data: raw image bytes from the user -> stored like any candidate, credited as the user's own."""
    wd = work_dir(os.path.abspath(lecture_dir))
    path = os.path.join(wd, 'broll', f'suggest_{clip_id}.json')
    doc = load_json(path, {})
    it = next((i for i in doc.get('items', []) if i['id'] == item_id), None)
    if not it:
        raise RuntimeError('item not found')
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert('RGB')
    if im.width > MAX_W:
        im = im.resize((MAX_W, int(im.height * MAX_W / im.width)))
    h = hashlib.sha1(data).hexdigest()[:12]
    d = os.path.join(wd, 'broll'); os.makedirs(d, exist_ok=True)
    fn = f'{h}.jpg'
    im.save(os.path.join(d, fn), 'JPEG', quality=90)
    cand = {'provider': 'own', 'title': name, 'src_url': 'own://' + fn, 'source_url': '', 'license': 'own', 'artist': '', 'credit': '', 'src': fn}
    it['candidates'].insert(0, cand)
    it['pick'] = 0
    save_json(path, doc)
    return it


def add_manual(lecture_dir, clip_id, t, query, phrase=None, w0=None, w1=None, duration=4.0, offline=False):
    """User-requested image: search + download now, add a 'new' suggestion card at time t (master clock)."""
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    cuts = load_json(os.path.join(wd, 'cuts.json'), {})
    clip = next((c for c in cuts.get('clips', []) if c.get('id') == clip_id), None)
    if not clip:
        raise RuntimeError('clip not found')
    path = os.path.join(wd, 'broll', f'suggest_{clip_id}.json')
    doc = load_json(path, None) or {'clipId': clip_id, 'a': clip['a'], 'b': clip['b'], 'name': clip['name'], 'created': time.time(), 'items': [], 'llm': False}
    if w0 is None:
        from transcribe import load_words, words_in
        ws = words_in(load_words(wd), clip['a'], clip['b'])
        idx = next((i for i, w in enumerate(ws) if w['s'] >= t - 0.02), len(ws) - 1)
        w0 = w1 = max(0, idx)
        if ws:
            t = ws[w0]['s']
            phrase = phrase or ws[w0]['t']
    hid = hashlib.sha1(('%.3f%s' % (t, query)).encode()).hexdigest()[:8]
    it = {'id': 's_' + hid, 't': round(float(t), 3), 'w0': int(w0), 'w1': int(w1 if w1 is not None else w0), 'phrase': (phrase or query)[:80], 'query': query[:80],
          'why': 'requested by you', 'duration': float(duration), 'status': 'new', 'candidates': [], 'pick': 0, 'source': 'manual'}
    if offline:
        it['candidates'] = [{'provider': 'placeholder', 'title': query, 'src_url': 'placeholder://' + query, 'source_url': '', 'license': 'own', 'artist': '', 'credit': '', 'src': _placeholder(wd, query)}]
    else:
        it['candidates'] = find_images(query, wd=wd)
        i = download_first(wd, it['candidates'])
        if i is not None:
            it['pick'] = i
    doc['items'] = [i for i in doc['items'] if i['id'] != it['id']] + [it]
    doc['items'].sort(key=lambda i: i['t'])
    save_json(path, doc)
    return it
