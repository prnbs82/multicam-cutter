"""One door to a chat model, whichever the user has — or none.

Providers (config in ~/.config/multicam/ai.json, never inside the repo or the lecture folder):
  none        — the tool works without any AI: keyword topic labels, noun-phrase image moments
  claude-cli  — the `claude` command installed on this machine (what the original setup used)
  anthropic   — Anthropic Messages API with an API key
  xai         — Grok through xAI's OpenAI-compatible API (https://api.x.ai/v1) with an xAI API key
  openai      — any other OpenAI-compatible chat API: OpenAI, Ollama, LM Studio, Groq, ... (base_url + model + key)

Auto-detection when nothing is configured: `claude` on PATH -> claude-cli; ANTHROPIC_API_KEY -> anthropic;
XAI_API_KEY / OPENAI_API_KEY -> openai; otherwise none.
"""
import json, os, re, shutil, subprocess, urllib.error, urllib.request
from hw import load_config, save_config

PRESETS = {   # provider -> (base_url, default model)
    'anthropic': ('https://api.anthropic.com', 'claude-sonnet-5'),
    'xai': ('https://api.x.ai/v1', 'grok-4-fast'),
    'openai': ('https://api.openai.com/v1', 'gpt-4o-mini'),
    'ollama': ('http://localhost:11434/v1', 'llama3.1'),
}
PROVIDERS = ('none', 'claude-cli', 'anthropic', 'xai', 'openai')


def config():
    """Stored config merged with auto-detection for anything left blank."""
    c = dict(load_config('ai.json', {}) or {})
    prov = c.get('provider') or 'auto'
    if prov == 'auto':
        if shutil.which('claude'):
            prov = 'claude-cli'
        elif os.environ.get('ANTHROPIC_API_KEY'):
            prov = 'anthropic'
        elif os.environ.get('XAI_API_KEY'):
            prov = 'xai'
        elif os.environ.get('OPENAI_API_KEY'):
            prov = 'openai'
        else:
            prov = 'none'
        c['auto'] = True
    c['provider'] = prov if prov in PROVIDERS else 'none'
    if c['provider'] == 'anthropic':
        c.setdefault('base_url', PRESETS['anthropic'][0]); c.setdefault('model', PRESETS['anthropic'][1])
        c['api_key'] = c.get('api_key') or os.environ.get('ANTHROPIC_API_KEY', '')
    if c['provider'] == 'xai':
        c['base_url'] = c.get('base_url') or PRESETS['xai'][0]; c['model'] = c.get('model') or PRESETS['xai'][1]
        c['api_key'] = c.get('api_key') or os.environ.get('XAI_API_KEY', '')
    if c['provider'] == 'openai':
        c['base_url'] = c.get('base_url') or PRESETS['openai'][0]; c['model'] = c.get('model') or PRESETS['openai'][1]
        c['api_key'] = c.get('api_key') or os.environ.get('OPENAI_API_KEY', '')
    return c


def save(provider='auto', base_url='', model='', api_key=None):
    """Store the AI settings in ~/.config/multicam/ai.json and return provider_status().
    api_key=None keeps the stored key, '' removes it, anything else replaces it."""
    c = dict(load_config('ai.json', {}) or {})
    c['provider'] = provider if provider in PROVIDERS + ('auto',) else 'auto'
    c['base_url'] = (base_url or '').strip().rstrip('/')
    c['model'] = (model or '').strip()
    if api_key is not None:                      # None = keep the stored key; '' = remove it
        if api_key.strip():
            c['api_key'] = api_key.strip()
        else:
            c.pop('api_key', None)
    save_config('ai.json', c)
    return provider_status()


def provider_status():
    """What the UI/CLI show: effective provider, model, whether a key is present, and whether calls can succeed."""
    c = config()
    p = c['provider']
    has_key = bool(c.get('api_key'))
    if p == 'none':
        avail, detail = False, 'no AI assistant — automatic keyword labels and image moments (set one up with the 🤖 AI button)'
    elif p == 'claude-cli':
        avail = bool(shutil.which('claude'))
        detail = 'Claude CLI on this machine' if avail else 'Claude CLI selected but `claude` is not installed'
    elif p == 'anthropic':
        avail = has_key
        detail = f"Anthropic API, model {c.get('model')}" + ('' if has_key else ' — no API key')
    elif p == 'xai':
        avail = has_key
        detail = f"Grok (xAI), model {c.get('model')}" + ('' if has_key else ' — no xAI API key')
    else:
        avail = has_key or 'localhost' in (c.get('base_url') or '') or '127.0.0.1' in (c.get('base_url') or '')
        detail = f"OpenAI-compatible API at {c.get('base_url')}, model {c.get('model')}" + ('' if avail else ' — no API key')
    stored = load_config('ai.json', {}) or {}
    return {'provider': p, 'configured': stored.get('provider') or 'auto', 'base_url': c.get('base_url', ''), 'model': c.get('model', ''),
            'has_key': has_key, 'key_hint': (c['api_key'][:4] + '…' + c['api_key'][-4:]) if has_key and len(c['api_key']) > 8 else ('set' if has_key else ''),
            'available': avail, 'detail': detail, 'presets': PRESETS}


# ------------------------------------------------------------------ back-ends
def _post_json(url, payload, headers, timeout):
    """POST a JSON body and return the parsed JSON reply (urllib only — no extra dependencies)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'MulticamCutter/1.0', **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8', errors='replace'))


def _ask_claude_cli(prompt, timeout):
    """Run `claude -p` and return its text result (raises on a non-zero exit)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE')}   # allow running from inside a Claude Code session
    r = subprocess.run(['claude', '-p', prompt, '--output-format', 'json'], capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(f'claude CLI failed: {r.stderr[-500:]}')
    try:
        outer = json.loads(r.stdout)
        return outer.get('result', '') if isinstance(outer, dict) else r.stdout
    except json.JSONDecodeError:
        return r.stdout


def _ask_anthropic(prompt, timeout, c):
    """One user turn through the Anthropic Messages API; returns the concatenated text blocks."""
    out = _post_json(c['base_url'].rstrip('/') + '/v1/messages',
                     {'model': c['model'], 'max_tokens': 4096, 'messages': [{'role': 'user', 'content': prompt}]},
                     {'x-api-key': c['api_key'], 'anthropic-version': '2023-06-01'}, timeout)
    return ''.join(b.get('text', '') for b in out.get('content', []) if b.get('type') == 'text')


def _ask_openai(prompt, timeout, c):
    """One user turn through an OpenAI-compatible /chat/completions endpoint (xAI Grok, OpenAI, Ollama...)."""
    headers = {'Authorization': f"Bearer {c['api_key']}"} if c.get('api_key') else {}
    out = _post_json(c['base_url'].rstrip('/') + '/chat/completions',
                     {'model': c['model'], 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.2}, headers, timeout)
    return out['choices'][0]['message']['content']


def _ask_raw(prompt, timeout):
    """Call the configured provider; returns the text, or raises RuntimeError with a human-readable reason."""
    c = config()
    p = c['provider']
    try:
        if p == 'none':
            raise RuntimeError('no AI assistant configured')
        if p == 'claude-cli':
            return _ask_claude_cli(prompt, timeout)
        if p == 'anthropic':
            if not c.get('api_key'):
                raise RuntimeError('no Anthropic API key')
            return _ask_anthropic(prompt, timeout, c)
        if p in ('xai', 'openai'):
            if p == 'xai' and not c.get('api_key'):
                raise RuntimeError('no xAI API key (get one at console.x.ai)')
            return _ask_openai(prompt, timeout, c)
        raise RuntimeError(f'unknown provider {p!r}')
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='replace')[:300]
        raise RuntimeError(f'{p}: HTTP {e.code} from {e.url.split("/")[2]} — {body}') from None
    except (OSError, subprocess.TimeoutExpired, urllib.error.URLError, KeyError, ValueError) as e:
        raise RuntimeError(f'{p}: {e}') from None


def ask(prompt, timeout=180):
    """Plain text answer, or None when no provider is usable / the call fails (callers fall back to heuristics)."""
    try:
        return _ask_raw(prompt, timeout)
    except RuntimeError as e:
        print(f'AI provider failed: {e}')
        return None


def test(timeout=60):
    """Round trip a tiny prompt for the Test button / `multicam.py ai --test`; returns {'ok', 'reply'|'error', 'provider', 'model'}."""
    c = config()
    try:
        reply = _ask_raw('Reply with exactly the two words: connection works', timeout)
        return {'ok': True, 'reply': (reply or '').strip()[:200], 'provider': c['provider'], 'model': c.get('model', '')}
    except RuntimeError as e:
        return {'ok': False, 'error': str(e), 'provider': c['provider'], 'model': c.get('model', '')}


def ask_json(prompt, timeout=180):
    """Ask for a JSON object and parse it (None on any failure)."""
    text = ask(prompt, timeout)
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        print('AI returned no JSON:', text[:300])
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print('AI JSON parse error:', e, text[:300])
        return None


def source_name():
    """Short tag for 'who produced this label' fields."""
    p = config()['provider']
    return {'claude-cli': 'claude', 'anthropic': 'claude', 'xai': 'grok', 'openai': 'llm'}.get(p, 'heuristic')


claude_json = ask_json   # backwards compatibility for older call sites
