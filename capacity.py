"""What this machine can handle: RAM/CPU/disk snapshot, per-job memory needs, safe worker counts, clip-length limits.

The pipeline streams everything, so clip length is limited by DISK (intermediate pieces + output) and by TIME, not RAM.
RAM matters for the number of parallel encoders and for the analysis jobs (pose/gaze/morph use MediaPipe/OpenCV).
"""
import os, shutil, json

# measured on this machine (12 cores, 2026-08-26); refined at runtime from the last render's speed if available
EST_RENDER_SPEED_X = 2.2      # x realtime with 4 parallel x264-medium encoders from HEVC sources
MB_PER_MIN_PIECES = 60.0      # 1080p30 CRF18 intermediates + wav audio, per output minute
MB_PER_MIN_OUTPUT = 60.0
RAM_PER_ENCODER_GB = 0.7      # ffmpeg HEVC decode + libx264 medium 1080p
RAM_ANALYSIS_GB = 2.5         # mediapipe + opencv + one 1080p frame stream
RAM_SERVER_GB = 0.5


def meminfo():
    """RAM/swap in GB via psutil (Linux, macOS, Windows); /proc fallback; generous defaults if neither works."""
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return {'total': vm.total / 1024 ** 3, 'available': vm.available / 1024 ** 3, 'swap_total': sw.total / 1024 ** 3, 'swap_free': sw.free / 1024 ** 3}
    except ImportError:
        pass
    m = {}
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':', 1)
                m[k] = int(v.strip().split()[0]) / 1024 / 1024   # GB
    except OSError:
        return {'total': 8.0, 'available': 4.0, 'swap_total': 0.0, 'swap_free': 0.0}
    return {'total': m.get('MemTotal', 0), 'available': m.get('MemAvailable', 0), 'swap_total': m.get('SwapTotal', 0), 'swap_free': m.get('SwapFree', 0)}


def report(lecture_dir=None):
    cpu = os.cpu_count() or 4
    mem = meminfo()
    disk_free_gb = None
    if lecture_dir:
        try:
            disk_free_gb = shutil.disk_usage(lecture_dir).free / 1024 ** 3
        except OSError:
            pass
    # encoders that fit comfortably in AVAILABLE RAM (leave 3 GB headroom) and CPU (3 cores each for libx264;
    # a hardware encoder needs ~1 core per stream, the decode still runs on the CPU)
    try:
        import hw
        enc = hw.encoder()[0]
        gpus = hw.gpu_names()
    except Exception:
        enc, gpus = 'libx264', []
    per_core = 3 if enc == 'libx264' else 2
    workers = max(1, min(4, int((mem['available'] - 3.0 - RAM_SERVER_GB) // RAM_PER_ENCODER_GB), cpu // per_core))
    speed = EST_RENDER_SPEED_X * workers / 4.0 * (1.0 if enc == 'libx264' else 1.6)
    if lecture_dir:
        st = None
        try:
            st = json.load(open(os.path.join(lecture_dir, '_multicam', 'render', 'status.json')))
        except Exception:
            pass
        if st and st.get('speed') and st.get('state') == 'done':
            speed = float(st['speed'])
    max_min_disk = None
    if disk_free_gb is not None:
        max_min_disk = (disk_free_gb - 5.0) * 1024 / (MB_PER_MIN_PIECES + MB_PER_MIN_OUTPUT)    # keep 5 GB free
    notes = []
    if mem['swap_total'] < 0.5:
        notes.append('No swap: a memory spike cannot spill to disk, so RAM exhaustion freezes the machine instead of slowing it. '
                     'Consider a swap file (e.g. 8 GB) as a safety net.')
    if mem['available'] < 6:
        notes.append(f'Only {mem["available"]:.1f} GB RAM available right now — close other applications before exporting or analysing.')
    analysis_ok = mem['available'] > RAM_ANALYSIS_GB + 2.0
    return {
        'cpu_cores': cpu, 'ram_total_gb': round(mem['total'], 1), 'ram_available_gb': round(mem['available'], 1),
        'swap_gb': round(mem['swap_total'], 1), 'disk_free_gb': None if disk_free_gb is None else round(disk_free_gb, 1),
        'recommended_workers': workers, 'est_render_speed_x': round(speed, 2),
        'max_clip_minutes_by_disk': None if max_min_disk is None else int(max(0, max_min_disk)),
        'ram_limit_on_clip_length': None,     # streaming pipeline: none
        'analysis_ok_now': analysis_ok, 'notes': notes,
        'encoder': enc, 'encoder_name': {'libx264': 'CPU (libx264)', 'h264_nvenc': 'NVIDIA NVENC', 'h264_qsv': 'Intel Quick Sync',
                                         'h264_vaapi': 'VA-API', 'h264_videotoolbox': 'Apple VideoToolbox'}.get(enc, enc), 'gpus': gpus,
    }


def estimate_export(minutes, rep=None, lecture_dir=None):
    rep = rep or report(lecture_dir)
    return {'minutes': round(minutes, 1), 'render_minutes': round(minutes / max(0.2, rep['est_render_speed_x']), 1),
            'temp_gb': round(minutes * MB_PER_MIN_PIECES / 1024, 2), 'output_gb': round(minutes * MB_PER_MIN_OUTPUT / 1024, 2),
            'fits_disk': rep['disk_free_gb'] is None or (minutes * (MB_PER_MIN_PIECES + MB_PER_MIN_OUTPUT) / 1024 + 5.0) < rep['disk_free_gb'],
            'workers': rep['recommended_workers']}


if __name__ == '__main__':
    import sys
    r = report(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps(r, indent=2))
    for m in (5, 11, 30, 79):
        print(m, 'min clip ->', estimate_export(m, r))
