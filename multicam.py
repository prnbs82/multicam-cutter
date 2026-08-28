#!/usr/bin/env python3
"""Multicam lecture cutter CLI.

  multicam.py init   <lecture dir>            scan files -> _multicam/project.json
  multicam.py sync   <lecture dir> [--force]  audio cross-correlation -> sync.json
  multicam.py proxy  <lecture dir> [--files-only]  build proxies + multiview
  multicam.py serve  <lecture dir> [--port N] [--open]
  multicam.py render <lecture dir> [--out FILE] [--clip NAME [--tighten]]
  multicam.py transcribe <lecture dir> [--range A B] [--model large-v3]
  multicam.py topics <lecture dir> [--granularity 0.5] [--no-llm]
  multicam.py posematch <lecture dir> [--force]        pose-matched cut points for all joins -> joinfix.json
  multicam.py gaze <lecture dir> --range A B [--person JSON] [--min-shot 3]   propose camera angles from head direction
  multicam.py capacity <lecture dir>                   what this machine can handle (RAM/CPU/disk, render speed, clip limits)
  multicam.py doctor [lecture dir] [--refresh]         hardware report: usable encoders, GPU, Whisper device, missing modules, AI provider
  multicam.py setup  <lecture dir>                     init + sync + proxy in one go (what the `multicam` launcher runs on a new folder)
"""
import argparse, sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('doctor'); d.add_argument('lecture_dir', nargs='?'); d.add_argument('--refresh', action='store_true', help='re-probe the encoders')
    s = sub.add_parser('setup'); s.add_argument('lecture_dir'); s.add_argument('--workers', type=int, default=0)
    for c in ('init', 'sync', 'proxy', 'serve', 'render', 'transcribe', 'topics', 'posematch', 'gaze', 'broll', 'capacity'):
        p = sub.add_parser(c)
        p.add_argument('lecture_dir')
        if c == 'sync':
            p.add_argument('--force', action='store_true')
        if c == 'proxy':
            p.add_argument('--files-only', action='store_true')
            p.add_argument('--workers', type=int, default=3)
        if c == 'serve':
            p.add_argument('--port', type=int, default=8765)
            p.add_argument('--open', action='store_true')
        if c == 'transcribe':
            p.add_argument('--range', nargs=2, type=float, metavar=('A', 'B'), help='master-clock seconds; default whole lecture')
            p.add_argument('--model', default='auto', help='auto (by RAM/GPU), large-v3, medium, small')
            p.add_argument('--force', action='store_true', help='re-transcribe even if cached')
        if c == 'broll':
            p.add_argument('--clip', required=True, help='clip id')
            p.add_argument('--no-llm', action='store_true')
            p.add_argument('--offline', action='store_true', help='placeholder images, no network (tests)')
        if c == 'gaze':
            p.add_argument('--range', nargs=2, type=float, required=True, metavar=('A', 'B'))
            p.add_argument('--person', help='JSON {cam: [cx, cy]} anchors for the face to follow')
            p.add_argument('--min-shot', type=float, default=3.0)
        if c == 'posematch':
            p.add_argument('--force', action='store_true', help='recompute cached joins too')
        if c == 'topics':
            p.add_argument('--granularity', type=float, default=0.5, help='boundary sensitivity in std-devs (lower = more segments)')
            p.add_argument('--no-llm', action='store_true', help='keyword labels only; do not call the claude CLI')
        if c == 'render':
            p.add_argument('--tighten', action='store_true', help='with --clip: apply the clip\'s word-level edits (tighten/<id>.json)')
            p.add_argument('--out')
            p.add_argument('--clip', help='render only the named clip range (from cuts.json) to clips/<name>.mp4')
            p.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.cmd == 'doctor':
        from hw import doctor
        sys.exit(1 if doctor(a.lecture_dir, refresh=a.refresh) else 0)
    if a.cmd == 'setup':
        from project import init_project
        from sync import sync_project
        from proxy import build_proxies
        from capacity import report
        init_project(a.lecture_dir)
        sync_project(a.lecture_dir)
        build_proxies(a.lecture_dir, workers=a.workers or report(a.lecture_dir)['recommended_workers'])
        return
    if a.cmd == 'init':
        from project import init_project
        init_project(a.lecture_dir)
    elif a.cmd == 'sync':
        from sync import sync_project
        sync_project(a.lecture_dir, force=a.force)
    elif a.cmd == 'proxy':
        from proxy import build_proxies
        build_proxies(a.lecture_dir, files_only=a.files_only, workers=a.workers)
    elif a.cmd == 'serve':
        from server import serve
        serve(a.lecture_dir, a.port, a.open)
    elif a.cmd == 'render':
        from render import render
        render(a.lecture_dir, out=a.out, workers=a.workers, clip=a.clip, tighten=a.tighten)
    elif a.cmd == 'capacity':
        import json as _json
        from capacity import report, estimate_export
        r = report(a.lecture_dir); print(_json.dumps(r, indent=2))
        for m in (5, 11, 30):
            print(f'{m:3d} min clip ->', estimate_export(m, r))
    elif a.cmd == 'broll':
        from broll import run_suggest
        run_suggest(a.lecture_dir, a.clip, use_llm=not a.no_llm, offline=a.offline)
    elif a.cmd == 'gaze':
        import json as _json
        from gaze import run_gaze
        run_gaze(a.lecture_dir, a.range[0], a.range[1], person=_json.loads(a.person) if a.person else None, min_shot=a.min_shot)
    elif a.cmd == 'posematch':
        from posematch import run_posematch
        run_posematch(a.lecture_dir, force=a.force)
    elif a.cmd == 'topics':
        from topics import analyse
        analyse(a.lecture_dir, granularity=a.granularity, use_llm=not a.no_llm)
    elif a.cmd == 'transcribe':
        from transcribe import transcribe_range
        rng = a.range or (None, None)
        transcribe_range(a.lecture_dir, rng[0], rng[1], model=a.model, force=a.force)


if __name__ == '__main__':
    main()
