"""Captions: caption blocks from the kept words, a plain .srt and a styled .ass next to every export, and the
text of the .ass that ffmpeg's libass burns into the picture when captions are enabled.

_multicam/captions.json:
  enabled   burn the captions into the exported video (the .srt/.ass sidecars are written either way)
  sizePct   font size as % of the frame height (4.5 -> ~49 px at 1080p)
  color     base text colour '#RRGGBB'; position 'bottom'|'top'; outline (black rim); bg (dark box behind the text)
  font      family name resolved by fontconfig at burn time; maxChars per caption line (a block holds ~2 lines)
  mode      'all' = captions everywhere; 'ranges' = only inside the marked `ranges` [{a,b} master seconds] —
            attention captions for chosen parts, like shorts/reels
  words     {"<index into words.json>": {"text": "shown instead of the spoken word ('' hides it)", "color": "#RRGGBB"}}
The transcript corrections (double-click in the UI) apply to captions too; `words` overrides win over both.
"""
import json, os
from common import load_json, save_json, work_dir

DEFAULTS = {'enabled': False, 'mode': 'all', 'ranges': [], 'sizePct': 4.5, 'color': '#FFFFFF', 'position': 'bottom',
            'outline': True, 'bg': False, 'font': 'DejaVu Sans', 'maxChars': 42, 'words': {}}


def load(wd):
    """captions.json merged over DEFAULTS (missing file -> defaults, captions disabled)."""
    return {**DEFAULTS, **(load_json(os.path.join(wd, 'captions.json'), {}) or {})}


def save(wd, doc):
    """Store the caption settings/overrides (only known keys, so stray fields never accumulate)."""
    save_json(os.path.join(wd, 'captions.json'), {k: doc[k] for k in DEFAULTS if k in doc})


def kept(words, intervals, corrections, doc, segs):
    """The words that survive the edit, as caption entries {i, a, b (output-timeline s), t (display text), color}.
    segs = [(master_a, master_b, out_offset)] of the rendered video pieces (from the render plan); a word whose time
    falls between pieces (inside a transition) snaps to the nearest piece edge. Overridden text '' drops the word."""
    ov = {int(k): v for k, v in (doc.get('words') or {}).items() if str(k).lstrip('-').isdigit()}

    def out_time(m, end=False):
        prev_end = 0.0
        for a, b, o in segs:
            if m < a:
                return prev_end if end else o
            if m <= b + 1e-6:
                return o + min(max(m - a, 0.0), b - a)
            prev_end = o + (b - a)
        return prev_end

    marked = [(float(r['a']), float(r['b'])) for r in (doc.get('ranges') or [])] if doc.get('mode') == 'ranges' else None
    out = []
    for i, w in enumerate(words):
        mid = (w['s'] + w['e']) / 2
        if not any(a <= mid < b for a, b in intervals):
            continue
        if marked is not None and not any(a <= mid < b for a, b in marked):
            continue                             # 'only marked parts' mode: captions appear just where the user marked them
        o = ov.get(i, {})
        t = (o['text'] if 'text' in o else corrections.get(f"{w['s']:.3f}", w['t'])).strip()
        if not t:
            continue
        a = out_time(w['s'])
        out.append({'i': i, 'a': a, 'b': max(out_time(w['e'], end=True), a + 0.05), 't': t, 'color': o.get('color')})
    return out


def blocks(kws, doc):
    """Group caption entries into timed blocks: break at pauses > 0.7 s, after a sentence once the block has some
    length, and at ~2 lines of maxChars. Each block ends 0.35 s after its last word but never into the next block."""
    mx = max(20, int(doc.get('maxChars') or 42)) * 2
    groups, cur = [], []
    for w in kws:
        if cur:
            gap = w['a'] - cur[-1]['b']
            ln = len(' '.join(x['t'] for x in cur))
            if gap > 0.7 or ln + 1 + len(w['t']) > mx or (cur[-1]['t'][-1:] in '.?!' and ln >= 24 and gap > 0.15):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    out = []
    for gi, g in enumerate(groups):
        a = g[0]['a']
        b = g[-1]['b'] + 0.35
        if gi + 1 < len(groups):
            b = min(b, groups[gi + 1][0]['a'] - 0.02)
        out.append({'a': a, 'b': max(b, a + 0.2), 'words': g})
    return out


def _srt_t(t):
    """SRT timestamp HH:MM:SS,mmm."""
    ms = max(0, int(round(t * 1000)))
    return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}'


def _ass_t(t):
    """ASS timestamp H:MM:SS.cc (centiseconds)."""
    cs = max(0, int(round(t * 100)))
    return f'{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}'


def _ass_color(hexs, alpha='00'):
    """'#RRGGBB' -> ASS '&HAABBGGRR' (little-endian colour, AA = transparency)."""
    h = (hexs or '#FFFFFF').lstrip('#')
    h = h if len(h) == 6 else 'FFFFFF'
    return f'&H{alpha}{h[4:6]}{h[2:4]}{h[0:2]}'.upper()


def _esc(t):
    """Make a word safe inside an ASS dialogue line (braces start override blocks)."""
    return t.replace('{', '(').replace('}', ')').replace('\n', ' ')


def write_srt(path, bl):
    """Plain SubRip sidecar (no styling — for YouTube upload or players)."""
    with open(path, 'w', encoding='utf-8') as f:
        for n, b in enumerate(bl, 1):
            f.write(f"{n}\n{_srt_t(b['a'])} --> {_srt_t(b['b'])}\n{' '.join(w['t'] for w in b['words'])}\n\n")
    return path


def write_ass(path, bl, doc):
    """Styled ASS sidecar: base style from the settings, per-word colours as inline override tags ({\\1c...}word{\\r})."""
    size = max(12, int(round(1080 * float(doc.get('sizePct') or 4.5) / 100)))
    border = 3 if doc.get('bg') else 1                 # 3 = opaque box behind the text, 1 = outline + shadow
    outline = 3 if (doc.get('outline') or doc.get('bg')) else 0
    align = 8 if doc.get('position') == 'top' else 2   # 8 = top centre, 2 = bottom centre
    style = (f"Style: Default,{doc.get('font') or 'DejaVu Sans'},{size},{_ass_color(doc.get('color'))},&H000000FF,"
             f"&H00000000,{_ass_color('#000000', '60')},-1,0,0,0,100,100,0,0,{border},{outline},0,{align},60,60,45,1")
    lines = ['[Script Info]', 'ScriptType: v4.00+', 'PlayResX: 1920', 'PlayResY: 1080', 'WrapStyle: 0',
             'ScaledBorderAndShadow: yes', '', '[V4+ Styles]',
             'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, '
             'Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, '
             'MarginL, MarginR, MarginV, Encoding', style, '', '[Events]',
             'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    for b in bl:
        parts = []
        for w in b['words']:
            if w.get('color'):
                parts.append('{\\1c' + _ass_color(w['color'])[2:] + '&}' + _esc(w['t']) + '{\\r}')
            else:
                parts.append(_esc(w['t']))
        lines.append(f"Dialogue: 0,{_ass_t(b['a'])},{_ass_t(b['b'])},Default,,0,0,0,," + ' '.join(parts))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def filter_escape(p):
    """Escape a filename for use inside an ffmpeg filter argument (ass=...)."""
    return p.replace('\\', '\\\\').replace("'", "\\'").replace(':', '\\:').replace(',', '\\,')


if __name__ == '__main__':
    import sys
    print(json.dumps(load(work_dir(sys.argv[1])), indent=2))
