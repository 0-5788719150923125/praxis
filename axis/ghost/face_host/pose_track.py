"""Offline body-silhouette + pose track for the Masking editor's umbra effect.

Reads a video once, runs MediaPipe's pose landmarker over it at a fixed sample
rate, and writes a compact binary track the editor reads back BY TIME. Each
sample carries two things the umbra needs and nothing else could give it:

  THE SILHOUETTE  a per-pixel person mask, downsampled to the editor's grid.
                  This is the ghost's BODY. The effect it replaces grew its
                  shape out of the footage's own cast shadow, which meant it
                  only worked when the room happened to have one big soft
                  shadow on a chromatically uniform wall - and even there it
                  needed a colour hypothesis, two flood fills and a confidence
                  gate that drew nothing about a third of the time. A person
                  mask is the same answer with none of the conditions.

  THE POSE        33 landmarks with visibility. This is the ghost's SKELETON:
                  where her head is, where her shoulders are, where her eyes
                  are - so the ghost's own head lands somewhere real at any
                  size, and its eyes land on that head rather than in the
                  middle of the mass and hoping.

Same architecture and the same three reasons as face_host/face_track.py: the
live preview and the export relaunch are separate PROCESSES that must agree
frame-for-frame and do because they read one cached file; no detection ever
runs inside the render loop; and because the whole track exists before
playback, the effect can read the frame she has NOT REACHED YET. That last one
is the entire point of the umbra - a ghost that moves a beat before she does
reads as a puppeteer, and no live tracker can ever supply it.

THE CAST DIRECTION, and why it is measured over the whole clip
--------------------------------------------------------------
The ghost is thrown away from her along the direction the room's key light
throws her real shadow. That is a property of the ROOM, not of the frame, so
it is measured once here rather than re-derived every tick (re-deriving it per
frame is exactly how the clown earned its uniform twitch, and the effect this
replaces EMA'd it at 0.06 for the same reason).

It is measured TEMPORALLY, which needs no colour model at all: her shadow is
the part of the background that CHANGES AS SHE MOVES. Cells the person mask
never covers, ranked by their luminance variance over the clip, are her moving
shadow plus whatever else flickers; the vector from her mean centroid to that
variance mass is the direction the light throws her. Compare this with asking
"which side of her is darker", which on the reference clip answers correctly
for the wrong reason - camera-left is a cream door and camera-right is a teal
wall, so the luminance question is really a question about paint.

THE FORMAT (little-endian, matches GDScript's FileAccess defaults)
-----------------------------------------------------------------
    magic     4s    b"GST2"
    version   u32   2
    rate      f32   samples per second
    count     u32   number of samples IN THIS FILE
    points    u32   landmarks per sample (33)
    mask_w    u32   silhouette grid width
    mask_h    u32   silhouette grid height
    start     u32   the GLOBAL sample index of this file's first sample
    dir_x     f32   cast direction, aspect-corrected and unit length
    dir_y     f32
    dir_conf  f32   0 = the measurement told us nothing, use your own default
    then `count` samples, each:
        found u8              1 = a pose was detected in this sample
        xy    f32 * points*2  normalized to the frame (0..1), origin top-left
        vis   f32 * points    visibility 0..1 (a landmark off-frame reads ~0)
        mask  u8  * mask_w*mask_h   silhouette coverage, 0..255

Samples with found = 0 carry a zeroed mask and held coordinates: the flag is
the truth, the numbers are a convenience.

At the default 96x54 / 12 Hz a sample is 5581 bytes, so a ten-minute clip
caches about 40 MB in user://pose_tracks. That is deliberate - a byte per cell
keeps the mask soft-edged, and a shadow's edge is the one part of it a viewer
actually looks at.

Usage:
    python pose_track.py --video <path> --out <track.bin> --model <model.task>
                         [--rate 12] [--mask-w 96] [--mask-h 54]
                         [--progress <file>]
"""

import argparse
import math
import os
import struct
import sys

MAGIC = b"GST2"
VERSION = 2
POINTS = 33
HEADER = 44  # 4 magic + u32 version + f32 rate + u32 count + u32 points
             #   + u32 mask_w + u32 mask_h + u32 start
             #   + f32 dir_x + f32 dir_y + f32 conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    # 12 Hz rather than the face track's 15: a body is a much lower-frequency
    # thing than a mouth, and every sample here costs 5 KB of cache instead of
    # 4 bytes. The editor interpolates between samples anyway.
    ap.add_argument("--rate", type=float, default=12.0)
    # The editor's own coarse grid. A shadow is a REGION and regions survive
    # downsampling in a way the clown's eye sockets never did.
    ap.add_argument("--mask-w", type=int, default=96)
    ap.add_argument("--mask-h", type=int, default=54)
    ap.add_argument("--progress", default="")
    # ONE WINDOW OF THE CLIP, not the whole thing. The editor asks for the chunk
    # the playhead is in and the one after it, so an umbra layer starts drawing
    # about twenty seconds after it is placed instead of after the whole clip has
    # been read. Omit both and this reads the lot, which is what the format check
    # and any offline use want.
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    args = ap.parse_args()

    import cv2  # imported here so --help works without the venv populated
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    if not os.path.exists(args.model):
        print("pose_track: model not found: %s" % args.model, file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("pose_track: could not open %s" % args.video, file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / src_fps if total > 0 else 0.0
    # Read BEFORE the capture is released - the properties of a released
    # VideoCapture read back as 0 and the cast direction would then be
    # measured in a 1:1 space on 16:9 footage.
    src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0

    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=True,
        )
    )

    mw, mh = max(8, args.mask_w), max(8, args.mask_h)
    # The running statistics the cast direction is measured from. Accumulated
    # as we go so a long clip never holds more than one frame at a time.
    lum_sum = np.zeros((mh, mw), np.float64)
    lum_sq = np.zeros((mh, mw), np.float64)
    occupancy = np.zeros((mh, mw), np.float64)
    centroid_sum = np.zeros(2, np.float64)
    centroid_n = 0
    frames_seen = 0

    tmp = args.out + ".part"
    f = open(tmp, "wb")
    # A placeholder header, rewritten at the end once the count and the cast
    # direction are known. Writing to a .part and renaming means a reader can
    # never open a half-written track (the discipline the clip transcode uses).
    f.write(b"\0" * HEADER)

    # SAMPLE TIMES ARE ON ONE GLOBAL GRID, whoever produced them. A chunk's
    # first sample is `start` on that grid, so two chunks written by two runs
    # butt together exactly and the editor can index by a single number.
    step = 1.0 / max(args.rate, 1.0)
    start_index = int(math.ceil(args.start * args.rate - 1e-9))
    next_t = start_index * step
    stop_t = args.start + args.duration if args.duration > 0.0 else 1e18
    # SEEK ONLY WHEN ASKED, and then TRUST THE FRAME'S OWN TIMESTAMP rather than
    # the frame count. Seeking is accurate on some containers and wildly not on
    # others (the reason face_track.py walks the whole file); reading the
    # timestamp back per frame makes the sample times right whatever the seek
    # actually did, which is what lets a chunk be produced without decoding
    # everything before it.
    seeked = args.start > 0.0
    if seeked:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    idx = 0
    count = 0
    found_n = 0
    last_report = -1.0
    prev_xy = [0.5] * (POINTS * 2)
    zero_mask = bytes(mw * mh)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0 if seeked else idx / src_fps
        idx += 1
        if t >= stop_t:
            break
        if t + 1e-9 < next_t:
            continue
        next_t += step
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(image, int(t * 1000.0))

        # INTER_AREA, not the default: downsampling a mask with bilinear
        # samples a scattering of source pixels and leaves a mask that is
        # noisy at exactly the scale we are about to magnify. Area averaging
        # is a box filter, which is what "coverage of this cell" means.
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lum = cv2.resize(grey, (mw, mh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        lum_sum += lum
        lum_sq += lum.astype(np.float64) ** 2
        frames_seen += 1

        if res.pose_landmarks and res.segmentation_masks:
            pts = res.pose_landmarks[0]
            if len(pts) < POINTS:
                print(
                    "pose_track: model returned %d landmarks, expected %d"
                    % (len(pts), POINTS),
                    file=sys.stderr,
                )
                f.close()
                os.unlink(tmp)
                return 2
            xy = [c for p in pts[:POINTS] for c in (p.x, p.y)]
            vis = [float(p.visibility) for p in pts[:POINTS]]
            raw = res.segmentation_masks[0].numpy_view()
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            small = cv2.resize(
                np.clip(raw, 0.0, 1.0), (mw, mh), interpolation=cv2.INTER_AREA
            )
            occupancy += small
            m = np.sum(small)
            if m > 1.0:
                ys, xs = np.mgrid[0:mh, 0:mw]
                centroid_sum += (
                    np.sum(small * (xs + 0.5)) / m / mw,
                    np.sum(small * (ys + 0.5)) / m / mh,
                )
                centroid_n += 1
            mask_bytes = (small * 255.0).astype(np.uint8).tobytes()
            prev_xy = xy
            found = 1
            found_n += 1
        else:
            xy = list(prev_xy)
            vis = [0.0] * POINTS
            mask_bytes = zero_mask
            found = 0

        f.write(struct.pack("<B", found))
        f.write(struct.pack("<%df" % (POINTS * 2), *xy))
        f.write(struct.pack("<%df" % POINTS, *vis))
        f.write(mask_bytes)
        count += 1
        if args.progress and t - last_report >= 1.0:
            last_report = t
            span = args.duration if args.duration > 0.0 else duration
            if span > 0:
                _write_progress(args.progress, (t - args.start) / span)
    cap.release()

    if count == 0:
        f.close()
        os.unlink(tmp)
        print("pose_track: no frames decoded from %s" % args.video, file=sys.stderr)
        return 2

    aspect = src_w / src_h if src_w > 0 and src_h > 0 else 16.0 / 9.0
    dx, dy, conf = _cast_direction(
        np, lum_sum, lum_sq, occupancy, centroid_sum, centroid_n, frames_seen, aspect
    )

    f.seek(0)
    f.write(MAGIC)
    f.write(
        struct.pack(
            "<IfIIIIIfff",
            VERSION, args.rate, count, POINTS, mw, mh, start_index, dx, dy, conf,
        )
    )
    f.close()
    os.replace(tmp, args.out)
    if args.progress:
        _write_progress(args.progress, 1.0)

    print(
        "pose_track: %d samples from index %d at %.1f Hz, pose in %d (%.0f%%), "
        "mask %dx%d, cast (%.3f, %.3f) conf %.2f -> %s"
        % (
            count,
            start_index,
            args.rate,
            found_n,
            100.0 * found_n / count,
            mw,
            mh,
            dx,
            dy,
            conf,
            args.out,
        )
    )
    return 0


def _cast_direction(
    np, lum_sum, lum_sq, occupancy, centroid_sum, centroid_n, frames, aspect
):
    """Which way the room throws her shadow, from the whole clip at once.

    HER SHADOW IS THE PART OF THE BACKGROUND THAT MOVES WHEN SHE DOES. So:
    take the cells the person mask essentially never covers, rank them by how
    much their luminance varied over the clip, keep the ones well above the
    background's own median variation, and point from her mean centroid at
    what is left. No colour model, no surface hypothesis, no per-frame
    decision - and nothing here can be fooled by a wall that simply happens to
    be painted darker than the door opposite it, which is what defeats every
    "which side of her is dimmer" formulation on the reference clip.

    Returns (dx, dy, conf) with (dx, dy) unit length in ASPECT-CORRECTED space
    (x already multiplied by the frame aspect), because that is the space the
    editor and the field shader both do their geometry in.
    """
    if frames <= 1 or centroid_n <= 0:
        return 1.0, 0.0, 0.0
    mh, mw = lum_sum.shape
    mean = lum_sum / frames
    var = np.maximum(lum_sq / frames - mean * mean, 0.0)
    sd = np.sqrt(var)
    # Never-her cells only. A cell she passes through has a huge variance that
    # has nothing to do with any shadow.
    bg = (occupancy / frames) < 0.02
    if np.count_nonzero(bg) < 32:
        return 1.0, 0.0, 0.0
    # Above the background's OWN typical wobble: compression noise, grain and
    # the camera's auto-exposure move every cell a little, and subtracting the
    # median removes all of that at once without having to model any of it.
    base = float(np.median(sd[bg]))
    wgt = np.where(bg, np.maximum(sd - base * 1.5, 0.0), 0.0)
    total = float(np.sum(wgt))
    if total <= 1e-6:
        return 1.0, 0.0, 0.0
    ys, xs = np.mgrid[0:mh, 0:mw]
    px = float(np.sum(wgt * (xs + 0.5))) / total / mw
    py = float(np.sum(wgt * (ys + 0.5))) / total / mh
    cx = centroid_sum[0] / centroid_n
    cy = centroid_sum[1] / centroid_n
    v = np.array([(px - cx) * aspect, py - cy], np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-4:
        return 1.0, 0.0, 0.0
    # Confidence = how much of the moving background is actually on one side.
    # A clip where she moves in front of a busy scene spreads the weight all
    # round her and this collapses toward 0, which is the honest answer: there
    # is no single direction to be had, so the editor should use its default.
    unit = v / n
    proj = 0.0
    for_sum = 0.0
    dxs = ((xs + 0.5) / mw - cx) * aspect
    dys = (ys + 0.5) / mh - cy
    r = np.sqrt(dxs * dxs + dys * dys) + 1e-6
    proj = float(np.sum(wgt * (dxs * unit[0] + dys * unit[1]) / r))
    for_sum = total
    conf = max(0.0, min(1.0, proj / for_sum))
    return float(unit[0]), float(unit[1]), conf


def _write_progress(path: str, frac: float) -> None:
    try:
        with open(path, "w") as f:
            f.write("%.4f\n" % max(0.0, min(1.0, frac)))
    except OSError:
        pass  # progress is a convenience; never fail the run over it


if __name__ == "__main__":
    sys.exit(main())
