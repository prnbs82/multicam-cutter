"""One door to a chat model, whichever the user has — or none.

Providers (config in ~/.config/multicam/ai.json, never inside the repo or the lecture folder):
  none        — the tool works without any AI: keyword topic labels, noun-phrase image moments
  claude-cli  — the `claude` command installed on this machine (what the original setup used)
  anthropic   — Anthropic Messages API with an API key
  openai      — any OpenAI-compatible chat API: xAI Grok, OpenAI, Ollama, LM Studio, Groq, ... (base_url + model + key)

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
PROVIDERS = ('none', 'claude-cli', 'anthropic', 'openai')


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
            prov = 'openai'; c.setdefault('base_url', PRESETS['xai'][0]); c.setdefault('model', PRESETS['xai'][1])
        elif os.environ.get('OPENAI_API_KEY'):
            prov = 'openai'
        else:
            prov = 'none'
        c['auto'] = True
    c['provider'] = prov if prov in PROVIDERS else 'none'
    if c['provider'] == 'anthropic':
        c.setdefault('base_url', PRESETS['anthropic'][0]); c.setdefault('model', PRESETS['anthropic'][1])
        c['api_key'] = c.get('api_key') or os.environ.get('ANTHROPIC_API_KEY', '')
    if c['provider'] == 'openai':
        c.setdefault('base_url', PRESETS['openai'][0]); c.setdefault('model', PRESETS['openai'][1])
        c['api_key'] = c.get('api_key') or os.environ.get('XAI_API_KEY') or os.environ.get('OPENAI_API_KEY', '')
    return c


def save(provider='auto', base_url='', model='', api_key=None):
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
    c = config()
    p = c['provider']
    has_key = bool(c.get('api_key'))
    if p == 'none':
        avail, detail = False, 'no AI assistant — automatic keyword labels and image moments (add one under Advanced)'
    elif p == 'claude-cli':
        avail = bool(shutil.which('claude'))
        detail = 'Claude CLI on this machine' if avail else 'Claude CLI selected but `claude` is not installed'
    elif p == 'anthropic':
        avail = has_key
        detail = f"Anthropic API, model {c.get('model')}" + ('' if has_key else ' — no API key')
    else:
        avail = has_key or 'localhost' in (c.get('base_url') or '') or '127.0.0.1' in (c.get('base_url') or '')
        detail = f"OpenAI-compatible API at {c.get('base_url')}, model {c.get('model')}" + ('' if avail else ' — no API key')
    stored = load_config('ai.json', {}) or {}
    return {'provider': p, 'configured': stored.get('provider') or 'auto', 'base_url': c.get('base_url', ''), 'model': c.get('model', ''),
            'has_key': has_key, 'key_hint': (c['api_key'][:4] + '…' + c['api_key'][-4:]) if has_key and len(c['api_key']) > 8 else ('set' if has_key else ''),
            'available': avail, 'detail': detail, 'presets': PRESETS}


# ------------------------------------------------------------------ back-ends
def _post_json(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'MulticamCutter/1.0', **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8', errors='replace'))


def _ask_claude_cli(prompt, timeout):
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
    out = _post_json(c['base_url'].rstrip('/') + '/v1/messages',
                     {'model': c['model'], 'max_tokens': 4096, 'messages': [{'role': 'user', 'content': prompt}]},
                     {'x-api-key': c['api_key'], 'anthropic-version': '2023-06-01'}, timeout)
    return ''.join(b.get('text', '') for b in out.get('content', []) if b.get('type') == 'text')


def _ask_openai(prompt, timeout, c):
    headers = {'Authorization': f"Bearer {c['api_key']}"} if c.get('api_key') else {}
    out = _post_json(c['base_url'].rstrip('/') + '/chat/completions',
                     {'model': c['model'], 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.2}, headers, timeout)
    return out['choices'][0]['message']['content']


def ask(prompt, timeout=180):
    """Plain text answer, or None when no provider is usable / the call fails (callers fall back to heuristics)."""
    c = config()
    p = c['provider']
    try:
        if p == 'claude-cli':
            return _ask_claude_cli(prompt, timeout)
        if p == 'anthropic':
            if not c.get('api_key'):
                return None
            return _ask_anthropic(prompt, timeout, c)
        if p == 'openai':
            return _ask_openai(prompt, timeout, c)
    except (OSError, subprocess.TimeoutExpired, urllib.error.URLError, RuntimeError, KeyError, ValueError) as e:
        msg = e.read().decode(errors='replace')[:300] if isinstance(e, urllib.error.HTTPError) else str(e)
        print(f'AI provider {p} failed: {msg}')
    return None


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
    return {'claude-cli': 'claude', 'anthropic': 'claude', 'openai': 'llm'}.get(p, 'heuristic')


claude_json = ask_json   # backwards compatibility for older call sites
