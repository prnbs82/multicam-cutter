"""Optical-flow morph transition (Morph-Cut style): warp frame A toward frame B over N frames instead of blending.

Uses OpenCV DIS optical flow at reduced resolution, checks forward/backward consistency, and returns None when the
flow is unreliable (large or inconsistent motion) so the caller can fall back to a plain dissolve.
"""
import subprocess
import numpy as np

FLOW_W = 640          # flow is computed at this width, then upscaled
MAX_MEDIAN_FLOW = 0.12   # of frame width: beyond this the warp tears -> fallback
MAX_INCONSISTENCY = 0.02  # of frame width: mean forward/backward disagreement -> fallback
MIN_WARP_GAIN = 0.35      # the warp must cut the photometric error in the moving region by at least this much


def _flow(a_gray, b_gray):
    """Dense DIS optical flow (medium preset, spatial propagation) from a_gray to b_gray; returns an HxWx2 float32 field in pixels."""
    import cv2
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    return dis.calc(a_gray, b_gray, None)


def _warp(img, flow):
    """Sample img at x + flow(x)."""
    import cv2
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def morph_frames(A, B, n):
    """A, B: BGR uint8 frames of equal size. Returns list of n BGR frames or None if the flow is unreliable."""
    import cv2
    H, W = A.shape[:2]
    sw = min(FLOW_W, W)
    sh = int(round(H * sw / W))
    a_s = cv2.cvtColor(cv2.resize(A, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b_s = cv2.cvtColor(cv2.resize(B, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    fab, fba = _flow(a_s, b_s), _flow(b_s, a_s)
    med = float(np.median(np.linalg.norm(fab.reshape(-1, 2), axis=1))) / sw
    # consistency: fab(x) + fba(x + fab(x)) should be ~0
    gx, gy = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
    fba_at = cv2.remap(fba, gx + fab[..., 0], gy + fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    incons = float(np.mean(np.linalg.norm(fab + fba_at, axis=2))) / sw
    if med > MAX_MEDIAN_FLOW or incons > MAX_INCONSISTENCY:
        return None, {'median_flow': med, 'inconsistency': incons, 'ok': False, 'why': 'flow too large/inconsistent'}
    # does the warp actually explain the motion? compare photometric error in the MOVING region: warp(A->B) vs plain A
    diff = cv2.absdiff(a_s, b_s)
    mask = cv2.dilate((diff > 25).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if mask.sum() > 50:
        a_warped = _warp(a_s, -fab)                     # A pulled fully onto B
        err_plain = float(np.abs(a_s.astype(np.float32) - b_s.astype(np.float32))[mask].mean())
        err_warp = float(np.abs(a_warped.astype(np.float32) - b_s.astype(np.float32))[mask].mean())
        gain = 1 - err_warp / max(err_plain, 1e-6)
        if gain < MIN_WARP_GAIN:
            return None, {'median_flow': med, 'inconsistency': incons, 'warp_gain': gain, 'ok': False, 'why': 'warp does not explain the motion'}
    else:
        gain = 1.0
    scale = W / sw
    FAB = cv2.resize(fab, (W, H), interpolation=cv2.INTER_LINEAR) * scale
    FBA = cv2.resize(fba, (W, H), interpolation=cv2.INTER_LINEAR) * scale
    out = []
    for i in range(n):
        t = (i + 1) / (n + 1)                        # strictly between A and B
        te = t * t * (3 - 2 * t)                     # ease in/out
        # intermediate seen from A: content at x came from x - te*FAB(x)  (A's pixels move along FAB)
        # intermediate seen from B: content at x will be at x - (1-te)*FBA(x)
        wa = _warp(A, -te * FAB)
        wb = _warp(B, -(1 - te) * FBA)
        out.append(cv2.addWeighted(wa, 1 - te, wb, te, 0))
    return out, {'median_flow': med, 'inconsistency': incons, 'warp_gain': gain, 'ok': True}


def write_video(frames, dst, fps, x264_args, vf_tail=''):
    """Pipe raw BGR frames into ffmpeg with the project's encoder settings (vf_tail: e.g. the VA-API upload step)."""
    H, W = frames[0].shape[:2]
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', 'pipe:0',
           '-vf', 'format=yuv420p' + vf_tail, '-frames:v', str(len(frames)), *x264_args, dst]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError(f'morph encode failed: {err[-1000:].decode(errors="ignore")}')
    return dst
