"""Word-level transcription of the master audio with faster-whisper, cached in _multicam/words.json.

words.json = {model, ranges:[{a,b}], words:[{s,e,t,p}]}  (s/e = master-clock seconds)
"""
import os, time, tempfile
from common import run, load_json, save_json, work_dir

MARGIN = 3.0        # seconds of context added on both sides of a requested range
CHUNK = 600.0       # process long ranges in chunks so progress is meaningful


def _status(wd, **kw):
    """Write transcribe/status.json with the given fields plus a timestamp."""
    save_json(os.path.join(wd, 'transcribe', 'status.json'), {**kw, 'time': time.time()})


def load_words(wd):
    """words.json of the work dir, or an empty document {model: None, ranges: [], words: []}."""
    return load_json(os.path.join(wd, 'words.json'), {'model': None, 'ranges': [], 'words': []})


def words_in(doc, a, b):
    """Words of doc whose start time lies in [a, b)."""
    return [w for w in doc['words'] if a <= w['s'] < b]


def coverage_missing(ranges, a, b):
    """Sub-ranges of [a,b] not covered by `ranges`."""
    cur, out = a, []
    for r in sorted(ranges, key=lambda r: r['a']):
        if r['b'] <= cur:
            continue
        if r['a'] >= b:
            break
        if r['a'] > cur:
            out.append((cur, min(r['a'], b)))
        cur = max(cur, r['b'])
        if cur >= b:
            break
    if cur < b:
        out.append((cur, b))
    return [(x, y) for x, y in out if y - x > 0.05]


def merge_range(doc, a, b, words):
    """Replace the words starting in [a, b) with `words`, keep them sorted, and merge [a, b] into doc['ranges'] (all in place)."""
    doc['words'] = [w for w in doc['words'] if not (a <= w['s'] < b)] + words
    doc['words'].sort(key=lambda w: w['s'])
    rs = sorted(doc['ranges'] + [{'a': a, 'b': b}], key=lambda r: r['a'])
    merged = []
    for r in rs:
        if merged and r['a'] <= merged[-1]['b'] + 0.01:
            merged[-1]['b'] = max(merged[-1]['b'], r['b'])
        else:
            merged.append(dict(r))
    doc['ranges'] = merged


def extract_wav(master, a, b, dst):
    """Cut [a, b) of the master audio to a mono 16 kHz 16-bit wav at dst (Whisper input)."""
    run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{a:.3f}', '-t', f'{b - a:.3f}', '-i', master,
         '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', dst])


def transcribe_range(lecture_dir, a=None, b=None, model='large-v3', force=False):
    """Transcribe the not-yet-covered parts of [a, b] of the master audio (whole lecture by default) with faster-whisper and update
    _multicam/words.json; returns the document. Works in CHUNK-second pieces with MARGIN seconds of context, saving after each;
    device/model from hw.whisper_choice (falls back to CPU int8 when the GPU fails); force re-does the range; progress in
    transcribe/status.json."""
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    proj = load_json(os.path.join(wd, 'project.json'))
    master = os.path.join(ld, proj['master_audio']['name'])
    DUR = float(proj['master_audio']['duration'])
    a = 0.0 if a is None else max(0.0, float(a))
    b = DUR if b is None else min(DUR, float(b))
    doc = load_words(wd)
    if force:
        doc['ranges'] = [r for r in doc['ranges'] if r['b'] <= a or r['a'] >= b]
        doc['words'] = [w for w in doc['words'] if not (a <= w['s'] < b)]
    missing = coverage_missing(doc['ranges'], a, b)
    if not missing:
        _status(wd, state='done', progress=1, message=f'already transcribed {a:.0f}-{b:.0f}s', range=[a, b])
        print('already covered')
        return doc
    total = sum(y - x for x, y in missing)
    from hw import whisper_choice, physical_cores
    device, compute_type, model = whisper_choice(None if model in (None, '', 'auto') else model)
    _status(wd, state='loading', progress=0, message=f'loading whisper {model} on {device} (first use downloads the model)', range=[a, b], model=model)
    from faster_whisper import WhisperModel
    t_load = time.time()
    try:
        m = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=max(4, physical_cores()))
    except Exception as e:                       # e.g. CUDA libraries missing/broken -> CPU still works
        if device == 'cpu':
            raise
        print(f'whisper on {device} failed ({e}); falling back to CPU int8')
        device, compute_type = 'cpu', 'int8'
        m = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=max(4, physical_cores()))
    print(f'model {model} loaded on {device} ({compute_type}) in {time.time() - t_load:.0f}s; transcribing {total:.0f}s of audio')
    done = 0.0
    t0 = time.time()
    try:
        for x, y in missing:
            t = x
            while t < y:
                u = min(y, t + CHUNK)
                xa, xb = max(0.0, t - MARGIN), min(DUR, u + MARGIN)
                with tempfile.TemporaryDirectory() as td:
                    wav = os.path.join(td, 'chunk.wav')
                    extract_wav(master, xa, xb, wav)
                    # a punctuated prompt nudges Whisper to emit punctuation + capitalisation (it otherwise
                    # sometimes falls into an all-lowercase, unpunctuated mode on lecture audio)
                    segments, _info = m.transcribe(wav, language='en', word_timestamps=True, vad_filter=True,
                                                   beam_size=5, condition_on_previous_text=False,
                                                   initial_prompt='Welcome back, everyone. Today we are going to talk about memory, compute, and why it matters. Okay? Let\'s begin.')
                    words = []
                    for seg in segments:
                        for w in (seg.words or []):
                            s, e = w.start + xa, w.end + xa
                            if s < t or s >= u:        # words in the margins belong to the neighbouring chunk
                                continue
                            txt = w.word.strip()
                            if txt:
                                words.append({'s': round(s, 3), 'e': round(max(e, s + 0.02), 3), 't': txt, 'p': round(float(w.probability), 3)})
                        prog = (done + min(max(seg.end + xa - t, 0), u - t)) / total
                        el = time.time() - t0
                        speed = (done + max(seg.end + xa - t, 0)) / el if el > 1 else 0
                        _status(wd, state='transcribing', progress=prog, model=model, range=[a, b],
                                message=f'{prog * 100:.0f}% · {speed:.2f}x realtime · ~{((total - prog * total) / speed / 60) if speed else 0:.0f} min left')
                merge_range(doc, t, u, words)
                doc['model'] = model
                save_json(os.path.join(wd, 'words.json'), doc)
                done += u - t
                print(f'  {t:8.1f}-{u:8.1f}s: {len(words)} words', flush=True)
                t = u
        _status(wd, state='done', progress=1, model=model, range=[a, b],
                message=f'done: {total:.0f}s of audio in {(time.time() - t0) / 60:.1f} min')
    except Exception as e:
        _status(wd, state='error', progress=0, message=str(e)[-1500:], range=[a, b])
        raise
    return doc
