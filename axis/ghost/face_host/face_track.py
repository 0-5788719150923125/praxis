"""Offline face-landmark track for the Masking editor's clown effect.

Reads a video once, runs MediaPipe's face landmarker over it at a fixed sample
rate, and writes a compact binary track the editor reads back BY TIME.

WHY OFFLINE, AND WHY A TRACK FILE
---------------------------------
This is the same shape as the umbra effect's look-ahead track (see
MaskEditor._umb_ensure_track): decode the whole clip once, fit every sampled
frame up front, cache the result, then playback is one array lookup. Three
things fall out of that which a per-frame detector cannot give:

  DETERMINISM  the live preview and the export relaunch are separate processes
               that must agree frame-for-frame. They read the same cached file,
               so they do.
  NO STALL     detection never runs inside the render loop, so it cannot cost
               frames or fight the audio clock.
  LOOK-AHEAD   the whole track exists before playback, so smoothing can use
               frames on BOTH sides of the current one. A live tracker only has
               the past and has to choose between lag and jitter; this doesn't.

WHAT REPLACES WHAT
------------------
This exists because the hand-written detector it replaces could not do the job.
That one built a weighted "this looks like skin" mass, took its centroid and
second moments, and hunted for dark clusters in an axis-aligned band above the
centre. It has no pose model, so a head turned or tilted broke it; its centroid
is dragged by any other skin in frame (a neck, a bare chest); and it returns
points and radii, so a nose could only ever be drawn as a circle. Measured on
the clip this was written for, it placed the nose at (0.599, 0.466) where the
nose really is (0.588, 0.507), after four rounds of corrections. MediaPipe finds
478 points on 24 of 24 sampled frames of the same clip, including the eye that
is turned away from the lens.

THE FORMAT (little-endian, matches GDScript's FileAccess defaults)
-----------------------------------------------------------------
    magic     4s    b"GFT1"
    version   u32   1
    rate      f32   samples per second
    count     u32   number of samples
    points    u32   landmarks per sample (478)
    then `count` samples, each:
        found u8    1 = a face was detected in this sample, 0 = none
        xy    f32 * points * 2   normalized to the frame (0..1), origin top-left

Samples with found = 0 still carry their (stale) coordinates so a reader can
choose between holding the last good fit and skipping - the flag is the truth,
the numbers are a convenience. Coordinates are normalized deliberately: the
editor works in frame UV throughout and a track must not care what resolution
the clip was decoded at.

Usage:
    python face_track.py --video <path> --out <track.bin> --model <model.task>
                         [--rate 15] [--progress <file>]
"""

import argparse
import os
import struct
import sys

MAGIC = b"GFT1"
VERSION = 1
POINTS = 478


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    # 15 Hz: fast enough that a talking head moves only a little between samples
    # (so reading between them is a short interpolation, not a guess) and slow
    # enough that a long clip stays a few minutes of work, not an hour.
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--progress", default="")
    args = ap.parse_args()

    import cv2  # imported here so --help works without the venv populated
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    if not os.path.exists(args.model):
        print("face_track: model not found: %s" % args.model, file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("face_track: could not open %s" % args.video, file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / src_fps if total > 0 else 0.0

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )

    # DECODE SEQUENTIALLY, SAMPLE BY TIME. Seeking per sample (CAP_PROP_POS_MSEC)
    # is accurate on some containers and wildly not on others, and a track whose
    # samples are at the wrong times is worse than no track. Walking the file
    # once and taking the frames whose timestamp has crossed the next sample
    # boundary is exact for every container and no slower.
    step = 1.0 / max(args.rate, 1.0)
    samples = []
    next_t = 0.0
    idx = 0
    last_report = -1.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = idx / src_fps
        idx += 1
        if t + 1e-9 < next_t:
            continue
        next_t += step
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(image, int(t * 1000.0))
        if res.face_landmarks:
            pts = res.face_landmarks[0]
            row = [1] + [c for p in pts[:POINTS] for c in (p.x, p.y)]
            # A model revision with fewer points would silently write short rows
            # and desynchronise every sample after it - refuse instead.
            if len(row) != 1 + POINTS * 2:
                print(
                    "face_track: model returned %d points, expected %d"
                    % (len(pts), POINTS),
                    file=sys.stderr,
                )
                return 2
        else:
            prev = samples[-1][1:] if samples else [0.5] * (POINTS * 2)
            row = [0] + list(prev)
        samples.append(row)
        if args.progress and duration > 0 and t - last_report >= 1.0:
            last_report = t
            _write_progress(args.progress, t / duration)
    cap.release()

    if not samples:
        print("face_track: no frames decoded from %s" % args.video, file=sys.stderr)
        return 2

    # Write to a .part and rename, so a reader can never open a half-written
    # track (the same discipline the clip transcode uses for video.ogv).
    tmp = args.out + ".part"
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IfII", VERSION, args.rate, len(samples), POINTS))
        for row in samples:
            f.write(struct.pack("<B", row[0]))
            f.write(struct.pack("<%df" % (POINTS * 2), *row[1:]))
    os.replace(tmp, args.out)
    if args.progress:
        _write_progress(args.progress, 1.0)

    found = sum(1 for r in samples if r[0])
    print(
        "face_track: %d samples at %.1f Hz, face found in %d (%.0f%%) -> %s"
        % (len(samples), args.rate, found, 100.0 * found / len(samples), args.out)
    )
    return 0


def _write_progress(path: str, frac: float) -> None:
    try:
        with open(path, "w") as f:
            f.write("%.4f\n" % max(0.0, min(1.0, frac)))
    except OSError:
        pass  # progress is a convenience; never fail the run over it


if __name__ == "__main__":
    sys.exit(main())
