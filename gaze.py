"""Propose camera angles from the speaker's head direction.

For every camera that films the speaker, frames of the ORIGINAL file are decoded at 2 fps over the clip range, the
speaker's face is found (full-range face detection -> crop -> face mesh) and the head turn is measured as the nose offset
within the face width (0 = frontal, +-0.5 = profile). At each instant the speaker "faces" the camera with the smallest
turn; when no camera sees her near-frontal she has turned to the board/screen -> the Slides angle. Runs are smoothed to a
minimum shot length. Results: _multicam/gaze/<key>.json ; status: _multicam/gaze/status.json
"""
import os, subprocess, time
import numpy as np
from common import load_json, save_json, work_dir

FPS_SAMPLE = 2           # analysed frames per second
FRONTAL = 0.30           # |turn| below this = facing that camera
MULTI_FRAC = 0.15        # fraction of frames with 2+ faces that triggers the "which person?" question


def key_for(a, b):
    """Result-file key of a clip range: start and end seconds with two decimals, e.g. '12.50-98.00'."""
    return f'{a:.2f}-{b:.2f}'


def _status(wd, **kw):
    """Write _multicam/gaze/status.json with the given fields plus the current time (polled by the UI)."""
    save_json(os.path.join(wd, 'gaze', 'status.json'), {**kw, 'time': time.time()})


def decode(src, start, dur, w=1920, h=1080):
    """Yield frames one at a time from ffmpeg (bounded memory: one frame ~6 MB, regardless of clip length).
    The previous version buffered the whole range (an 11-minute clip = ~8 GB per camera) and could exhaust RAM."""
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', src,
                             '-vf', f'fps={FPS_SAMPLE},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:-1:-1',
                             '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


class FaceTurn:
    """MediaPipe face detector + single-face mesh wrapped for head-turn measurement (loaded once per analysis; runs on the CPU)."""
    def __init__(self):
        """Load MediaPipe full-range face detection and a static single-face mesh; GLOG_minloglevel=2 silences their logging."""
        os.environ.setdefault('GLOG_minloglevel', '2')
        import mediapipe as mp
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.3)

    def faces(self, rgb):
        """Detect faces in an RGB frame; returns [{cx, cy, w, h}] in normalised image coordinates (centre and size)."""
        d = self.det.process(rgb).detections or []
        out = []
        for x in d:
            bb = x.location_data.relative_bounding_box
            out.append({'cx': bb.xmin + bb.width / 2, 'cy': bb.ymin + bb.height / 2, 'w': bb.width, 'h': bb.height})
        return out

    def turn(self, rgb, face):
        """Head turn of one detected face: nose x offset from the cheek midpoint divided by the cheek-to-cheek width (0 = frontal,
        about +-0.5 = profile), measured on a square crop of 2.2x the face size; None when the mesh finds no face in the crop."""
        H, W = rgb.shape[:2]
        cx, cy, s = face['cx'] * W, face['cy'] * H, max(face['w'] * W, face['h'] * H) * 2.2
        x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        crop = np.ascontiguousarray(rgb[y0:y0 + int(s), x0:x0 + int(s)])
        if crop.size == 0:
            return None
        ml = self.mesh.process(crop).multi_face_landmarks
        if not ml:
            return None
        P = ml[0].landmark
        nose, lch, rch = P[1], P[234], P[454]
        fw = abs(rch.x - lch.x) + 1e-6
        return float((nose.x - (lch.x + rch.x) / 2) / fw)


def pick_face(faces, anchor):
    """Choose the face to track: the one nearest the normalised (cx, cy) anchor when given, else the widest; None when there are none."""
    if not faces:
        return None
    if anchor:
        return min(faces, key=lambda f: (f['cx'] - anchor[0]) ** 2 + (f['cy'] - anchor[1]) ** 2)
    return max(faces, key=lambda f: f['w'])


def analyse(lecture_dir, a, b, person=None, min_shot=3.0, status_cb=None):
    """person: {cam: [cx, cy]} anchors (normalised) for cameras where a specific face must be tracked."""
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    layout = load_json(os.path.join(wd, 'layout.json'))
    cams = [x for x in layout['angles'] if x['id'] != 'slides' and any(cv['end'] > a and cv['start'] < b for cv in x['coverage'])]
    ft = FaceTurn()
    person = person or {}
    n = int((b - a) * FPS_SAMPLE)
    times = [a + (i + 0.5) / FPS_SAMPLE for i in range(n)]
    turns = {}                       # cam -> list of turn or None
    people = {}                      # cam -> info when several faces are present
    for ci, cam in enumerate(cams):
        turns[cam['id']] = [None] * n
        multi_frames, sample_boxes, sample_img = 0, None, None
        for cv in cam['coverage']:
            s0, s1 = max(a, cv['start']), min(b, cv['end'])
            if s1 <= s0:
                continue
            for k, f in enumerate(decode(os.path.join(ld, cv['file']), s0 - cv['offset'], s1 - s0)):
                t = s0 + (k + 0.5) / FPS_SAMPLE
                i = int(round((t - a) * FPS_SAMPLE - 0.5))
                if not (0 <= i < n):
                    continue
                import cv2
                rgb = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_BGR2RGB)
                faces = ft.faces(rgb)
                if len(faces) > 1:
                    multi_frames += 1
                    if sample_img is None:
                        sample_img, sample_boxes = f.copy(), faces
                face = pick_face(faces, person.get(cam['id']))
                turns[cam['id']][i] = ft.turn(rgb, face) if face else None
                if status_cb and k % 20 == 0:
                    status_cb(ci, len(cams), cam['name'], (t - a) / max(1e-6, b - a))
        frac = multi_frames / max(1, n)
        if frac > MULTI_FRAC and cam['id'] not in person and sample_img is not None:
            import cv2
            img = sample_img.copy()
            H, W = img.shape[:2]
            boxes = []
            for j, fb in enumerate(sorted(sample_boxes, key=lambda x: x['cx'])):
                x0, y0 = int((fb['cx'] - fb['w'] / 2) * W), int((fb['cy'] - fb['h'] / 2) * H)
                cv2.rectangle(img, (x0, y0), (int(x0 + fb['w'] * W), int(y0 + fb['h'] * H)), (0, 220, 255), 3)
                cv2.putText(img, str(j + 1), (x0, max(30, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3)
                boxes.append([round(fb['cx'], 4), round(fb['cy'], 4)])
            os.makedirs(os.path.join(wd, 'gaze'), exist_ok=True)
            path = os.path.join(wd, 'gaze', f'people_{cam["id"]}.jpg')
            cv2.imwrite(path, cv2.resize(img, (960, 540)))
            people[cam['id']] = {'frac': round(frac, 2), 'image': os.path.basename(path), 'boxes': boxes}
    return times, turns, people, [c['id'] for c in cams]


def classify(times, turns, cams, has_slides, min_shot=3.0):
    """Per-sample facing camera, smoothed, then merged into shots >= min_shot. Returns (labels, cuts)."""
    n = len(times)
    lab = []
    for i in range(n):
        best, bt = None, None
        for c in cams:
            v = turns[c][i]
            if v is None:
                continue
            if bt is None or abs(v) < bt:
                best, bt = c, abs(v)
        if best is None or bt > FRONTAL:
            lab.append('slides' if has_slides else best)
        else:
            lab.append(best)
    # median-ish smoothing: majority in a 5-sample window
    sm = []
    for i in range(n):
        win = [x for x in lab[max(0, i - 2):i + 3] if x]
        sm.append(max(set(win), key=win.count) if win else None)
    # runs -> merge short ones into the longer neighbour
    runs = []
    for i, x in enumerate(sm):
        if runs and runs[-1][0] == x:
            runs[-1][2] = i
        else:
            runs.append([x, i, i])
    min_len = int(round(min_shot * FPS_SAMPLE))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, r in enumerate(runs):
            if r[2] - r[1] + 1 < min_len:
                nb = []
                if k > 0: nb.append(k - 1)
                if k + 1 < len(runs): nb.append(k + 1)
                j = max(nb, key=lambda q: runs[q][2] - runs[q][1])
                runs[j][1], runs[j][2] = min(runs[j][1], r[1]), max(runs[j][2], r[2])
                runs.pop(k)
                # re-merge equal neighbours
                merged = []
                for rr in runs:
                    if merged and merged[-1][0] == rr[0]:
                        merged[-1][2] = rr[2]
                    else:
                        merged.append(rr)
                runs = merged
                changed = True
                break
    cuts = []
    for r in runs:
        if r[0]:
            t = times[r[1]] - 0.5 / FPS_SAMPLE
            cuts.append({'t': round(max(times[0] - 0.5 / FPS_SAMPLE, t), 3), 'cam': r[0]})
    return lab, sm, cuts


def run_gaze(lecture_dir, a, b, person=None, min_shot=3.0):
    """Gaze job for the clip [a,b]: analyse -> classify -> save _multicam/gaze/<key>.json; progress and errors go to gaze/status.json.
    Stops early in state 'choose_person' (saving needsPerson) when a camera sees several faces and has no anchor. Returns the saved dict."""
    ld = os.path.abspath(lecture_dir)
    wd = work_dir(ld)
    key = key_for(a, b)
    try:
        _status(wd, state='analysing', progress=0, message='decoding frames and finding the speaker', key=key)
        def cb(ci, nc, name, frac):
            """Progress callback from analyse(): write per-camera progress to status.json."""
            _status(wd, state='analysing', progress=(ci + frac) / nc, message=f'{name}: {frac * 100:.0f}% — head direction from 1080p frames', key=key)
        times, turns, people, cams = analyse(ld, a, b, person=person, min_shot=min_shot, status_cb=cb)
        layout = load_json(os.path.join(wd, 'layout.json'))
        has_slides = any(x['id'] == 'slides' for x in layout['angles'])
        need = {c: p for c, p in people.items() if not (person or {}).get(c)}
        if need:
            out = {'key': key, 'a': a, 'b': b, 'needsPerson': need, 'cams': cams}
            save_json(os.path.join(wd, 'gaze', key + '.json'), out)
            _status(wd, state='choose_person', progress=1, message='more than one person in view — pick whose gaze to follow', key=key, needsPerson=need)
            return out
        found = {c: sum(1 for v in turns[c] if v is not None) for c in cams}
        if sum(found.values()) < max(3, 0.05 * len(times)):
            lab, sm, cuts = [], [], []       # no speaker seen: nothing to propose
        else:
            lab, sm, cuts = classify(times, turns, cams, has_slides, min_shot)
        out = {'key': key, 'a': a, 'b': b, 'cams': cams, 'minShot': min_shot, 'person': person or {},
               'samples': [{'t': round(t, 2), **{c: (None if turns[c][i] is None else round(turns[c][i], 3)) for c in cams}, 'facing': (sm[i] if i < len(sm) else None)} for i, t in enumerate(times)],
               'cuts': cuts, 'found': found, 'people': people}
        save_json(os.path.join(wd, 'gaze', key + '.json'), out)
        summ = ', '.join(f'{c}: face in {found[c]}/{len(times)} frames' for c in cams)
        _status(wd, state='done', progress=1, message=(f'{len(cuts)} shots proposed ({summ})' if cuts else f'no speaker face found ({summ}) — nothing proposed'), key=key, cuts=len(cuts))
        print(f'{len(cuts)} shots proposed; {summ}')
        return out
    except Exception as e:
        _status(wd, state='error', progress=0, message=str(e)[-1500:], key=key)
        raise
