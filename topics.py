"""Topic analysis of the word-level transcript.

sentences -> local sentence embeddings (MiniLM via transformers) -> TextTiling-style boundaries -> segments
-> agglomerative clustering into topics -> recurrences -> labels (claude CLI, fallback: TF-IDF keywords)
Writes _multicam/topics.json and topics/status.json.
"""
import json, os, time
import numpy as np
from common import save_json, work_dir
from transcribe import load_words

EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
PALETTE = ['#58a6ff', '#3fb950', '#f0883e', '#d2a8ff', '#f778ba', '#e3b341', '#79c0ff', '#56d364',
           '#ffa657', '#ff7b72', '#a5d6ff', '#7ee787']
MIN_SEG_S = 45.0
CLUSTER_DIST = 0.35


def _status(wd, **kw):
    save_json(os.path.join(wd, 'topics', 'status.json'), {**kw, 'time': time.time()})


# ------------------------------------------------------------------ text units
def sentences_from_words(words, max_gap=1.0, max_words=40):
    sents, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        end = w['t'].rstrip()[-1:] in '.?!' or len(cur) >= max_words
        if not end and i + 1 < len(words) and words[i + 1]['s'] - w['e'] > max_gap:
            end = True
        if end:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    out = []
    for idx in sents:
        out.append({'a': words[idx[0]]['s'], 'b': words[idx[-1]]['e'], 'text': ' '.join(words[i]['t'] for i in idx), 'w0': idx[0], 'w1': idx[-1]})
    return out


# ------------------------------------------------------------------ embeddings
def embed(texts, batch=64):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from hw import torch_device
    dev = torch_device()
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    model = AutoModel.from_pretrained(EMB_MODEL).eval().to(dev)
    if dev == 'cpu':
        batch = min(batch, 16)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors='pt').to(dev)
            h = model(**enc).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            e = torch.nn.functional.normalize(e, dim=1)
            out.append(e.cpu().numpy())
    return np.vstack(out)


# ------------------------------------------------------------------ segmentation
def boundaries(sents, E, granularity=0.5, block=3):
    """TextTiling depth scores over sentence gaps; returns sorted list of sentence indices where a new segment starts."""
    n = len(sents)
    if n < 2 * block + 2:
        return []
    sims = np.zeros(n - 1)
    for g in range(n - 1):
        l = E[max(0, g - block + 1):g + 1].mean(0)
        r = E[g + 1:min(n, g + 1 + block)].mean(0)
        sims[g] = float(l @ r / (np.linalg.norm(l) * np.linalg.norm(r) + 1e-9))
    depth = np.zeros(n - 1)
    for g in range(n - 1):
        lp = sims[g]
        k = g
        while k > 0 and sims[k - 1] >= lp:
            lp = sims[k - 1]; k -= 1
        rp = sims[g]
        k = g
        while k < n - 2 and sims[k + 1] >= rp:
            rp = sims[k + 1]; k += 1
        depth[g] = (lp - sims[g]) + (rp - sims[g])
    thr = depth.mean() + granularity * depth.std()
    cands = sorted([g for g in range(n - 1) if depth[g] > thr], key=lambda g: -depth[g])
    chosen = []
    for g in cands:
        t = sents[g + 1]['a']
        if all(abs(t - sents[c + 1]['a']) >= MIN_SEG_S for c in chosen) and t - sents[0]['a'] >= MIN_SEG_S and sents[-1]['b'] - t >= MIN_SEG_S:
            chosen.append(g)
    return sorted(g + 1 for g in chosen)


def make_segments(sents, E, starts):
    bounds = [0] + starts + [len(sents)]
    segs = []
    for i, (x, y) in enumerate(zip(bounds, bounds[1:])):
        if y <= x:
            continue
        segs.append({'i': len(segs), 'a': sents[x]['a'], 'b': sents[y - 1]['b'], 's0': x, 's1': y - 1,
                     'text': ' '.join(s['text'] for s in sents[x:y]), 'emb': E[x:y].mean(0)})
    return segs


def cluster(segs, dist=CLUSTER_DIST):
    if len(segs) == 1:
        return [0]
    from sklearn.cluster import AgglomerativeClustering
    X = np.vstack([s['emb'] / (np.linalg.norm(s['emb']) + 1e-9) for s in segs])
    try:
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=dist, metric='cosine', linkage='average')
    except TypeError:  # older sklearn
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=dist, affinity='cosine', linkage='average')
    return list(ac.fit_predict(X))


# ------------------------------------------------------------------ labels
def keyword_labels(topic_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    ids = list(topic_texts)
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, max_features=5000)
    X = vec.fit_transform([topic_texts[i] for i in ids])
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for row, tid in enumerate(ids):
        v = X[row].toarray().ravel()
        top = terms[np.argsort(-v)[:3]]
        out[tid] = {'label': ' · '.join(t for t in top if v[np.where(terms == t)[0][0]] > 0) or f'topic {tid + 1}', 'summary': ''}
    return out


def claude_labels(topics_payload, timeout=180):
    """Ask the configured AI assistant (any provider, see llm.py) for names/summaries. Returns dict or None."""
    from llm import ask_json as claude_json
    prompt = (
        "You are labelling topic segments of a university lecture transcript (subject: computer memory / compute). "
        "For EACH topic below give a short label (max 5 words, title case, no trailing period) and a one-sentence summary. "
        "For each recurrence listed, write one short sentence saying how the later passage returns to the idea. "
        "Respond with ONLY a JSON object of the form "
        '{"topics":[{"id":<int>,"label":"...","summary":"..."}],"recurrences":[{"topic":<int>,"to":<int>,"note":"..."}]}'
        " and nothing else.\n\n" + json.dumps(topics_payload, ensure_ascii=False)
    )
    return claude_json(prompt, timeout=timeout)


# ------------------------------------------------------------------ main
def analyse(lecture_dir, granularity=0.5, use_llm=True, a=None, b=None):
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    t0 = time.time()
    try:
        words = load_words(wd)['words']
        if a is not None or b is not None:
            words = [w for w in words if (a is None or w['s'] >= a) and (b is None or w['s'] < b)]
        if len(words) < 50:
            raise RuntimeError('not enough transcribed words (run transcribe first)')
        _status(wd, state='embedding', progress=0.05, message=f'{len(words)} words → sentences + embeddings')
        sents = sentences_from_words(words)
        E = embed([s['text'] for s in sents])
        _status(wd, state='segmenting', progress=0.5, message=f'{len(sents)} sentences embedded; finding topic boundaries')
        starts = boundaries(sents, E, granularity)
        segs = make_segments(sents, E, starts)
        labels_idx = cluster(segs)
        # merge adjacent segments with the same topic
        merged = []
        for s, t in zip(segs, labels_idx):
            if merged and merged[-1]['topic'] == t:
                m = merged[-1]
                m['b'] = s['b']; m['s1'] = s['s1']; m['text'] += ' ' + s['text']
                m['emb'] = (m['emb'] + s['emb']) / 2
            else:
                merged.append({**s, 'topic': int(t)})
        for i, m in enumerate(merged):
            m['i'] = i
        # renumber topics by first appearance
        order = {}
        for m in merged:
            order.setdefault(m['topic'], len(order))
        for m in merged:
            m['topic'] = order[m['topic']]
        topics = []
        for tid in range(len(order)):
            ss = [m for m in merged if m['topic'] == tid]
            topics.append({'id': tid, 'color': PALETTE[tid % len(PALETTE)], 'segments': [m['i'] for m in ss],
                           'duration': sum(m['b'] - m['a'] for m in ss)})
        recurrences = []
        for t in topics:
            segs_t = t['segments']
            for j in segs_t[1:]:
                recurrences.append({'topic': t['id'], 'from': segs_t[0], 'to': j, 'note': ''})
        from llm import provider_status, source_name
        ai = provider_status()
        use_llm = use_llm and ai['available']
        _status(wd, state='labelling', progress=0.8, message=f'{len(merged)} segments, {len(topics)} topics, {len(recurrences)} recurrences; naming' + (f" with {ai['detail'].split(',')[0]}" if use_llm else ' by keywords'))
        topic_texts = {t['id']: ' '.join(merged[i]['text'] for i in t['segments']) for t in topics}
        labels = keyword_labels(topic_texts)
        source = 'keywords'
        if use_llm:
            payload = {'topics': [{'id': t['id'], 'minutes': round(t['duration'] / 60, 1),
                                   'excerpts': [' '.join(merged[i]['text'].split()[:150]) for i in t['segments'][:3]]} for t in topics],
                       'recurrences': [{'topic': r['topic'], 'to': r['to'], 'later_excerpt': ' '.join(merged[r['to']]['text'].split()[:80])} for r in recurrences[:40]]}
            got = claude_labels(payload)
            if got and got.get('topics'):
                for t in got['topics']:
                    if isinstance(t, dict) and t.get('id') in labels:
                        labels[t['id']] = {'label': str(t.get('label', '')).strip() or labels[t['id']]['label'], 'summary': str(t.get('summary', '')).strip()}
                for r in got.get('recurrences', []) or []:
                    for rec in recurrences:
                        if rec['topic'] == r.get('topic') and rec['to'] == r.get('to'):
                            rec['note'] = str(r.get('note', '')).strip()
                source = source_name()
        for t in topics:
            t.update(labels[t['id']])
        out = {'params': {'granularity': granularity, 'min_segment_s': MIN_SEG_S, 'cluster_distance': CLUSTER_DIST, 'model': EMB_MODEL},
               'labels_source': source, 'created': time.time(),
               'segments': [{'i': m['i'], 'a': round(m['a'], 3), 'b': round(m['b'], 3), 'topic': m['topic'],
                             'preview': ' '.join(m['text'].split()[:25]) + '…'} for m in merged],
               'topics': topics, 'recurrences': recurrences}
        save_json(os.path.join(wd, 'topics.json'), out)
        _status(wd, state='done', progress=1, message=f'{len(merged)} segments, {len(topics)} topics ({source} labels), {len(recurrences)} recurrences in {time.time() - t0:.0f}s')
        for t in topics:
            print(f"[{t['id']}] {t['label']:40s} {t['duration'] / 60:5.1f} min  segs {t['segments']}  {t.get('summary', '')}")
        for r in recurrences:
            print(f"  ↩ topic {r['topic']} returns in segment {r['to']} (from {r['from']}): {r['note']}")
        return out
    except Exception as e:
        _status(wd, state='error', progress=0, message=str(e)[-1500:])
        raise
