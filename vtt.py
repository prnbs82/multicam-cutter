"""Minimal WebVTT parser -> list of {start, end, text, speaker}."""
import re

_TS = re.compile(r'(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})')


def parse_ts(s):
    m = _TS.match(s.strip())
    if not m:
        raise ValueError(f'bad timestamp {s!r}')
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms) / 1000


def parse_vtt(path):
    text = open(path, encoding='utf-8-sig').read().replace('\r\n', '\n').replace('\r', '\n')
    cues = []
    for block in re.split(r'\n\s*\n', text):
        lines = [l for l in block.strip().split('\n') if l.strip()]
        idx = next((i for i, l in enumerate(lines) if '-->' in l), None)
        if idx is None:
            continue
        a, b = [x.strip() for x in lines[idx].split('-->')]
        b = b.split()[0]
        txt = ' '.join(lines[idx + 1:]).strip()
        speaker = None
        m = re.match(r'^([^:<>]{1,40}):\s*(.*)$', txt)
        if m:
            speaker, txt = m.group(1), m.group(2)
        cues.append({'start': parse_ts(a), 'end': parse_ts(b), 'text': txt, 'speaker': speaker})
    cues.sort(key=lambda c: c['start'])
    return cues
