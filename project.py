"""`init`: scan a lecture folder and build/update project.json."""
import os, re
from common import VIDEO_EXT, AUDIO_EXT, COLORS, probe_summary, load_json, save_json, work_dir

# Friendly-name hints for angle naming (substring of filename, lowercase) -> name
NAME_HINTS = [('zoom', 'Slides'), ('sazzad', 'Front'), ('bz', 'Side')]


def group_key(stem):
    """Files like record1_BZ / record2_BZ share a key; digits are stripped."""
    return re.sub(r'\d+', '#', stem).strip().lower()


def suggest_name(files):
    """Human-readable angle name from its file names: the first NAME_HINTS match, else the cleaned file stem (segment digits dropped)."""
    s = ' '.join(files).lower()
    for hint, name in NAME_HINTS:
        if hint in s:
            return name
    stem = os.path.splitext(files[0])[0]
    if len(files) > 1:  # segments of one camera: drop the segment number (record1_BZ -> record BZ)
        stem = re.sub(r'\d+', ' ', stem)
    name = re.sub(r'[\s_]+', ' ', stem).strip()
    return name or stem


def slug(name, taken):
    """Lower-case id for `name` (letters, digits, dashes), made unique against `taken` by appending 2, 3, ..."""
    base = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'angle'
    s, i = base, 2
    while s in taken:
        s, i = f'{base}{i}', i + 1
    return s


def init_project(lecture_dir):
    """Scan lecture_dir for media, adopt the first unknown audio as master, the first .vtt as transcript, group new videos into
    angles (segments share a group_key; each file probed with ffprobe) and save _multicam/project.json. Files already in
    project.json are kept untouched. Returns the project dict."""
    lecture_dir = os.path.abspath(lecture_dir)
    wd = work_dir(lecture_dir)
    ppath = os.path.join(wd, 'project.json')
    proj = load_json(ppath) or {
        'lecture_dir': lecture_dir, 'master_audio': None, 'angles': [], 'transcript': None,
    }
    known = {f['name'] for a in proj['angles'] for f in a['files']}
    if proj['master_audio']:
        known.add(proj['master_audio']['name'])

    entries = sorted(os.listdir(lecture_dir))
    videos, audios, vtts = [], [], []
    for e in entries:
        p = os.path.join(lecture_dir, e)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(e)[1].lower()
        if ext in VIDEO_EXT:
            videos.append(e)
        elif ext in AUDIO_EXT:
            audios.append(e)
        elif ext == '.vtt':
            vtts.append(e)

    if proj['transcript'] is None and vtts:
        proj['transcript'] = vtts[0]

    if proj['master_audio'] is None:
        cands = [a for a in audios if a not in known]
        if cands:
            info = probe_summary(os.path.join(lecture_dir, cands[0]))
            proj['master_audio'] = info
            known.add(cands[0])
            print(f'master audio: {cands[0]} ({info["duration"]:.1f}s)')

    new = [v for v in videos if v not in known]
    groups = {}
    for v in new:
        groups.setdefault(group_key(os.path.splitext(v)[0]), []).append(v)

    taken = {a['id'] for a in proj['angles']}
    used_keys = {a['key'] for a in proj['angles']}
    for key, files in groups.items():
        infos = []
        for f in files:
            print(f'probing {f} ...')
            infos.append(probe_summary(os.path.join(lecture_dir, f)))
        if not any(i['has_video'] for i in infos):
            continue
        infos.sort(key=lambda i: i.get('creation_time') or i['name'])
        name = suggest_name(files)
        aid = slug(name, taken)
        taken.add(aid)
        k = next(str(n) for n in range(1, 10) if str(n) not in used_keys)
        used_keys.add(k)
        proj['angles'].append({
            'id': aid, 'name': name, 'key': k,
            'color': COLORS[(len(proj['angles'])) % len(COLORS)],
            'files': infos,
        })
        print(f'angle [{k}] {name}: {", ".join(files)}')

    save_json(ppath, proj)
    print(f'wrote {ppath}')
    return proj
