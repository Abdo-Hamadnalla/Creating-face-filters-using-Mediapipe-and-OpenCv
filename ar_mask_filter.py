#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ar_mask_filter.py  -  Snapchat/Instagram-style AR face-mask filter
===================================================================
Based on the LearnOpenCV article "Create Snapchat/Instagram Filters Using
Mediapipe" (MediaPipe Face Mesh + OpenCV), rewritten as ONE self-contained file
(the article's faceBlendCommon.py helpers are included below).

Pipeline (same as the article)
------------------------------
  1. MediaPipe Face Mesh finds 468 landmarks on the live face.
  2. We keep only 75 of them (SELECTED_KEYPOINT_INDICES).  The position in that
     list IS the point's label:  label 0 = mesh point 127, label 1 = 93, ...
  3. YOU annotate the mask PNG with the same 75 labels (e.g. in makesense.ai)
     and export a CSV.  Label N on the mask = same facial spot as label N on a face.
  4. Optical flow + detection are blended to remove landmark jitter.
  5. Delaunay-triangulate the mask points, then affine-warp every triangle of the
     mask onto the matching triangle of the live face, and alpha-blend.

Sub-commands (run them in this order the first time)
----------------------------------------------------
  python ar_mask_filter.py labels                      # writes labels.txt (0..74) for makesense.ai
  python ar_mask_filter.py view                        # live webcam: see the 75 numbered landmarks
  python ar_mask_filter.py view  --image face.jpg      # same, on a photo -> face_landmarks.png
  python ar_mask_filter.py auto  masks/mask.png        # OPTIONAL: auto-write mask.csv (realistic faces only)
  python ar_mask_filter.py check masks/mask.png        # verify your annotations visually
  python ar_mask_filter.py run                         # the live AR filter (webcam)
  python ar_mask_filter.py run --image me.jpg --out out.png   # no webcam needed
  python ar_mask_filter.py run --video in.mp4 --out out.mp4   # no webcam needed

Masks are auto-discovered: put  masks/<name>.png  (RGBA) +  masks/<name>.csv  side by side.

Requires the LEGACY MediaPipe API:   pip install "mediapipe==0.10.21"
(mediapipe >= 0.10.30 removed `mp.solutions`, which this code - and the article - uses.)
"""

import argparse
import csv
import glob
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

# Fail early with a clear message if the legacy API is missing.
if not hasattr(mp, "solutions"):
    sys.exit(
        "ERROR: mediapipe %s has no `mp.solutions` (legacy API removed in 0.10.30+).\n"
        "Fix:   pip install 'mediapipe==0.10.21' 'numpy<2'" % getattr(mp, "__version__", "?")
    )


# =============================================================================
# 1.  CONFIGURATION  -  every tweakable value lives here (ALL_CAPS = constant)
# =============================================================================

# ---- Camera -----------------------------------------------------------------
CAMERA_INDEX = 0                    # /dev/video0 -> 0, /dev/video1 -> 1, ...
CAMERA_BACKEND = cv2.CAP_V4L2       # Linux webcam backend. If it fails, try cv2.CAP_ANY
REQUEST_FRAME_WIDTH = 1280          # Requested capture size (driver picks the nearest)
REQUEST_FRAME_HEIGHT = 720
REQUEST_FOURCC = "MJPG"             # Many USB webcams only reach ~10 fps at 720p in raw YUYV mode. "" = leave default
MIRROR_VIEW = True                  # Flip horizontally so it behaves like a mirror (selfie view)

# ---- MediaPipe Face Mesh ----------------------------------------------------
MAX_NUM_FACES = 1                   # Filter is built for one face
MIN_DETECTION_CONFIDENCE = 0.5      # Same values as the article
MIN_TRACKING_CONFIDENCE = 0.5

# ---- Landmark stabilisation (optical flow + detection blend) ----------------
USE_STABILIZATION = True
STABILIZATION_SIGMA = 50.0          # Bigger -> trusts the tracker more (smoother, slightly laggier)
LK_WIN_SIZE = (101, 101)            # Lucas-Kanade search window (article values)
LK_MAX_LEVEL = 15                   # Pyramid levels (OpenCV clamps this to what the image allows)
LK_MAX_ITERATIONS = 20
LK_EPSILON = 0.001

# ---- Rendering --------------------------------------------------------------
MASK_OPACITY = 1.0                  # 0.0 = invisible .. 1.0 = fully opaque
EDGE_BLUR_KSIZE = 3                 # Odd number; softens mask edges. 0 or 1 disables
MASK_MAX_SIDE_PX = 1024             # Big mask images are shrunk (annotations are scaled too) for speed

# ---- Files / UI -------------------------------------------------------------
MASKS_DIR = "masks"                 # Folder scanned for <name>.png + <name>.csv pairs
LABELS_FILE = "labels.txt"          # Written by the `labels` command
WINDOW_NAME = "AR Mask Filter"
WINDOW_WARMUP_FRAMES = 10           # Ignore the "window closed" check for the first N frames
SHOW_LANDMARKS_AT_START = False     # Toggle at runtime with the L key
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================================
# 2.  THE 75 LANDMARKS  -  the heart of the whole project
# =============================================================================
# Face Mesh returns 468 points.  The article keeps 75 of them, listed below in a
# FIXED ORDER.  The list position is the LABEL you must use when annotating:
#
#      label (0..74)  ->  Face Mesh index (0..467)
#      label 0   ->  127      label 8  ->  152 (chin)      label 28 -> 6 ...
#
# Labels follow the classic dlib-68 layout (0-67) plus 7 extra forehead points.
# "Screen-left" = the left side of the image as YOU see it (not anatomical left).
SELECTED_KEYPOINT_INDICES = [
    127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356,  # 0-16  jaw
    70, 63, 105, 66, 55,                                                                # 17-21 brow (screen-left)
    285, 296, 334, 293, 300,                                                            # 22-26 brow (screen-right)
    168, 6, 195, 4,                                                                     # 27-30 nose bridge
    64, 60, 94, 290, 439,                                                               # 31-35 nose base
    33, 160, 158, 173, 153, 144,                                                        # 36-41 eye (screen-left)
    398, 385, 387, 466, 373, 380,                                                       # 42-47 eye (screen-right)
    61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91,                                # 48-59 outer lips
    78, 81, 13, 311, 306, 402, 14, 178,                                                 # 60-67 inner lips
    162, 54, 67, 10, 297, 284, 389,                                                     # 68-74 forehead / temples
]
NUM_LANDMARKS = len(SELECTED_KEYPOINT_INDICES)  # 75

# (name, first label, last label, BGR colour) - used for drawing and for the legend
LANDMARK_GROUPS = [
    ("jaw: screen-left ear -> chin -> screen-right ear", 0, 16, (0, 165, 255)),
    ("eyebrow, screen-left (outer -> inner)",            17, 21, (0, 255, 255)),
    ("eyebrow, screen-right (inner -> outer)",           22, 26, (0, 255, 255)),
    ("nose bridge (top -> tip)",                         27, 30, (255, 200, 0)),
    ("nose base (screen-left -> right)",                 31, 35, (255, 200, 0)),
    ("eye, screen-left",                                 36, 41, (0, 255, 0)),
    ("eye, screen-right",                                42, 47, (0, 255, 0)),
    ("outer lips",                                       48, 59, (80, 80, 255)),
    ("inner lips",                                       60, 67, (200, 100, 255)),
    ("forehead / temples (screen-left -> right)",        68, 74, (255, 120, 120)),
]
LABEL_COLORS = [(255, 255, 255)] * NUM_LANDMARKS
for _name, _first, _last, _bgr in LANDMARK_GROUPS:
    for _i in range(_first, _last + 1):
        LABEL_COLORS[_i] = _bgr


# =============================================================================
# 3.  GEOMETRY HELPERS  (replacement for the article's faceBlendCommon.py)
# =============================================================================
def build_triangulation(points, width, height):
    """Delaunay triangulation of `points` (N x 2) inside a width x height image.

    Returns an (T x 3) int array.  Each row = 3 LABELS (row indices of `points`),
    so the SAME triple can be used on the mask and on the live face.

    Why Delaunay?  It splits the face into well-shaped triangles (no skinny ones),
    and every triangle can then be moved by a simple affine transform.
    """
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for x, y in points:
        subdiv.insert((float(x), float(y)))

    triangles = []
    for t in subdiv.getTriangleList():
        corners = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], np.float32)
        ids = []
        for corner in corners:
            # Subdiv2D returns coordinates, not indices -> find which of our points it is.
            dist = np.linalg.norm(points - corner, axis=1)
            k = int(np.argmin(dist))
            ids.append(k if dist[k] < 1.0 else -1)  # -1 = Subdiv2D's virtual outer vertices
        if -1 in ids or len(set(ids)) < 3:
            continue
        triangles.append(ids)
    return np.array(triangles, np.int32)


def warp_triangle(src, dst, tri_src, tri_dst):
    """Affine-warp ONE triangle of `src` onto the matching triangle of `dst` (in place).

    src, dst : H x W x C uint8 images (same C).   tri_* : 3 x 2 float arrays.
    Works on the triangle's bounding box only (fast) and is safe when the face is
    partly outside the frame (the destination box is clipped).
    """
    frame_h, frame_w = dst.shape[:2]
    x, y, w, h = cv2.boundingRect(np.float32([tri_dst]))      # destination box
    sx, sy, sw, sh = cv2.boundingRect(np.float32([tri_src]))  # source box
    if w < 1 or h < 1 or sw < 1 or sh < 1:
        return
    if w > 4 * frame_w or h > 4 * frame_h:                    # garbage landmarks -> skip
        return

    # Clip the destination box to the frame.
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame_w), min(y + h, frame_h)
    if x0 >= x1 or y0 >= y1:
        return

    # Triangle corners relative to their own bounding boxes.
    src_local = (tri_src - np.float32([sx, sy])).astype(np.float32)
    dst_local = (tri_dst - np.float32([x, y])).astype(np.float32)

    # 3 point pairs fully define an affine transform (rotate/scale/shear/translate).
    matrix = cv2.getAffineTransform(src_local, dst_local)
    warped = cv2.warpAffine(src[sy:sy + sh, sx:sx + sw], matrix, (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Hard triangle mask (LINE_8 includes the border pixels -> neighbours overlap
    # instead of leaving hairline gaps between triangles).
    tri_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(tri_mask, np.round(dst_local).astype(np.int32), 255, cv2.LINE_8)

    # Copy only the triangle's pixels into the (clipped) destination region.
    ox, oy = x0 - x, y0 - y
    inside = tri_mask[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] > 0
    dst[y0:y1, x0:x1][inside] = warped[oy:oy + (y1 - y0), ox:ox + (x1 - x0)][inside]


def similarity_from_two_points(src, dst):
    """2x3 matrix (rotation + uniform scale + translation) mapping 2 points -> 2 points.

    Used by 'anchor' accessories (ears, nose...).  Treating the points as complex
    numbers makes it exact:  z' = a*z + b,  a = (q1-q0)/(p1-p0).
    (The article's helper builds a fake 3rd point and calls estimateAffinePartial2D;
    this closed form gives the same similarity transform without that detour.)
    """
    p0, p1 = np.asarray(src, np.float64)
    q0, q1 = np.asarray(dst, np.float64)
    zs = complex(*(p1 - p0))
    zd = complex(*(q1 - q0))
    if abs(zs) < 1e-6:
        return np.float32([[1, 0, 0], [0, 1, 0]])
    a = zd / zs
    rot = np.array([[a.real, -a.imag], [a.imag, a.real]])
    shift = q0 - rot @ p0
    return np.hstack([rot, shift[:, None]]).astype(np.float32)


def composite(frame, canvas, roi=None):
    """Alpha-blend a PREMULTIPLIED BGRA `canvas` over a BGR `frame`.

    Premultiplied = colour channels were already multiplied by alpha, so blending is
    simply   out = frame * (1 - alpha) + colour.   It also avoids dark fringes at
    the mask edges when pixels are interpolated during warping.
    roi = (x0, y0, x1, y1) limits the work to a sub-rectangle (speed).
    """
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = roi if roi else (0, 0, width, height)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x0 >= x1 or y0 >= y1:
        return frame

    patch = canvas[y0:y1, x0:x1]
    if EDGE_BLUR_KSIZE and EDGE_BLUR_KSIZE > 1:
        patch = cv2.GaussianBlur(patch, (EDGE_BLUR_KSIZE, EDGE_BLUR_KSIZE), 0)

    alpha = patch[..., 3:4].astype(np.float32) * (MASK_OPACITY / 255.0)
    colour = patch[..., :3].astype(np.float32) * MASK_OPACITY
    blended = frame[y0:y1, x0:x1].astype(np.float32) * (1.0 - alpha) + colour

    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


# =============================================================================
# 4.  FACE LANDMARKS FROM MEDIAPIPE   <-- "how do I get the landmarks?" lives here
# =============================================================================
class FaceLandmarker:
    """Thin wrapper around MediaPipe Face Mesh that returns OUR 75 points."""

    def __init__(self, static_image_mode=False):
        # Create the model ONCE.  (The article re-creates it every frame, which
        # reloads the neural nets and is very slow.)
        #   static_image_mode=False -> video mode: detect once, then TRACK (fast)
        #   static_image_mode=True  -> run the detector on every call (photos)
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_bgr):
        """Return a (75 x 2) float32 array of pixel coordinates, or None if no face."""
        height, width = frame_bgr.shape[:2]

        # MediaPipe wants RGB; OpenCV gives BGR.  Convert exactly ONCE.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False            # lets MediaPipe skip a copy
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        # 468 landmarks, each with x, y in 0..1 (fractions of width/height) and a relative z.
        mesh = result.multi_face_landmarks[0].landmark

        # Keep our 75, in our order, converted from 0..1 to PIXELS.
        pts = np.array([(mesh[i].x * width, mesh[i].y * height)
                        for i in SELECTED_KEYPOINT_INDICES], np.float32)
        return pts

    def close(self):
        self._mesh.close()


class LandmarkStabilizer:
    """Removes landmark jitter by mixing detection with optical-flow tracking.

    Per point:  d = distance between "detected now" and "tracked from last frame".
        alpha = exp(-d^2 / sigma)
        final = (1 - alpha) * detected + alpha * tracked
    Small d (they agree)  -> alpha ~ 1 -> trust the smooth tracker (no shaking).
    Large d (face moved fast / tracker lost it) -> alpha ~ 0 -> trust the detector.
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None

    def update(self, frame_bgr, detected):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        smoothed = detected.copy()

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           LK_MAX_ITERATIONS, LK_EPSILON)
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray,
                self.prev_pts.reshape(-1, 1, 2), None,
                winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=lk_criteria)
            tracked = tracked.reshape(-1, 2)

            dist = np.linalg.norm(detected - tracked, axis=1)
            alpha = np.exp(-dist * dist / STABILIZATION_SIGMA)
            alpha[status.reshape(-1) == 0] = 0.0       # tracker lost the point -> use detection
            smoothed = (1.0 - alpha[:, None]) * detected + alpha[:, None] * tracked

        self.prev_gray = gray
        self.prev_pts = smoothed.astype(np.float32)
        return smoothed.astype(np.float32)


# =============================================================================
# 5.  MASK LOADING  (image + annotation CSV)
# =============================================================================
@dataclass
class MaskLayer:
    name: str
    kind: str                   # "morph" (full-face mask)  or  "anchor" (2-point accessory)
    premult_bgra: np.ndarray    # H x W x 4 uint8, colour premultiplied by alpha
    points: np.ndarray          # morph: (75 x 2), row i = label i.   anchor: (2 x 2)
    triangles: np.ndarray = None  # morph only: (T x 3) labels
    anchor_labels: tuple = ()     # anchor only: which 2 face labels it hooks onto


def read_annotation_csv(csv_path, image_name=None):
    """Read a makesense.ai point export:   label_name,x,y,image_name,image_w,image_h

    Returns {label:int -> (x, y)} in file order.  Header lines are skipped.
    """
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                label, x, y = int(row[0].strip()), float(row[1]), float(row[2])
            except ValueError:
                continue                                   # header / junk line
            rows.append((row[3].strip() if len(row) > 3 else "", label, x, y))

    if not rows:
        raise ValueError("%s: no valid 'label,x,y' rows. Labels must be NUMBERS 0..%d."
                         % (csv_path, NUM_LANDMARKS - 1))

    # If several images were exported into one CSV, keep only this image's rows.
    if image_name and len({r[0] for r in rows}) > 1:
        rows = [r for r in rows if r[0] == image_name]

    points = {}
    for _, label, x, y in rows:
        if label in points:
            print("  warning: label %d appears twice in %s (last one wins)" % (label, csv_path))
        points[label] = (x, y)
    return points


def load_mask(png_path, csv_path):
    """Load a mask PNG + its annotations and prepare everything needed to render it."""
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(png_path)
    if img.dtype == np.uint16:                              # 16-bit PNG -> 8-bit
        img = (img // 257).astype(np.uint8)
    if img.ndim == 2:                                       # grayscale -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:                                   # no alpha channel -> fully opaque
        print("  note: %s has no transparency (alpha); mask will be a solid rectangle" % png_path)
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])

    points = read_annotation_csv(csv_path, os.path.basename(png_path))

    # Optional down-scaling for speed.  Annotation coordinates must scale too!
    height, width = img.shape[:2]
    scale = min(1.0, MASK_MAX_SIDE_PX / float(max(height, width)))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        points = {k: (x * scale, y * scale) for k, (x, y) in points.items()}
        height, width = img.shape[:2]

    # Premultiply colour by alpha (see composite()).
    alpha = img[..., 3:4].astype(np.float32) / 255.0
    premult = np.dstack([img[..., :3].astype(np.float32) * alpha, img[..., 3]])
    premult = np.round(premult).astype(np.uint8)

    name = os.path.splitext(os.path.basename(png_path))[0]
    max_xy = np.array([width - 1, height - 1], np.float32)

    if all(i in points for i in range(NUM_LANDMARKS)):
        # ---- full-face mask: all 75 labels present ----
        pts = np.array([points[i] for i in range(NUM_LANDMARKS)], np.float32)
        pts = np.clip(pts, 0, max_xy)
        tri = build_triangulation(pts, width, height)
        if len(tri) == 0:
            raise ValueError("%s: Delaunay triangulation is empty - are the points all different?" % csv_path)
        return MaskLayer(name, "morph", premult, pts, triangles=tri)

    if len(points) == 2:
        # ---- accessory: exactly 2 points; they are the labels of the FACE points it hooks onto ----
        labels = tuple(points.keys())
        if any(not 0 <= l < NUM_LANDMARKS for l in labels):
            raise ValueError("%s: labels must be in 0..%d" % (csv_path, NUM_LANDMARKS - 1))
        pts = np.clip(np.array([points[l] for l in labels], np.float32), 0, max_xy)
        return MaskLayer(name, "anchor", premult, pts, anchor_labels=labels)

    missing = [i for i in range(NUM_LANDMARKS) if i not in points]
    raise ValueError("%s has %d points. A full-face mask needs all %d (missing labels: %s); "
                     "an accessory needs exactly 2." % (csv_path, len(points), NUM_LANDMARKS, missing[:15]))


def discover_masks(masks_dir, only=None):
    """Find every  <name>.png + <name>.csv  pair in masks_dir.  Returns {name: MaskLayer}."""
    layers = {}
    for png in sorted(glob.glob(os.path.join(masks_dir, "*.png"))):
        stem = os.path.splitext(png)[0]
        if stem.endswith("_check"):                         # preview images written by the `check` command
            continue
        if only and os.path.basename(stem) != only:
            continue
        csv_path = stem + ".csv"
        if not os.path.exists(csv_path):
            print("  skipping %s (no matching .csv annotation file)" % png)
            continue
        try:
            layer = load_mask(png, csv_path)
        except Exception as exc:                            # keep going, report the problem
            print("  ERROR loading %s: %s" % (png, exc))
            continue
        layers[layer.name] = layer
        print("  loaded '%s' as %s (%d points)" % (
            layer.name, "FULL-FACE MORPH" if layer.kind == "morph" else "2-POINT ACCESSORY",
            len(layer.points)))
    return layers


# =============================================================================
# 6.  RENDERING
# =============================================================================
def render_morph(frame, layer, face_pts):
    """Full-face mask: warp every mask triangle onto the matching face triangle."""
    height, width = frame.shape[:2]
    canvas = np.zeros((height, width, 4), np.uint8)          # premultiplied BGRA, starts transparent
    for tri in layer.triangles:
        warp_triangle(layer.premult_bgra, canvas, layer.points[tri], face_pts[tri])

    # Only blend inside the face's bounding box (much faster than the whole frame).
    margin = EDGE_BLUR_KSIZE + 2
    x0, y0 = np.floor(face_pts.min(axis=0)).astype(int) - margin
    x1, y1 = np.ceil(face_pts.max(axis=0)).astype(int) + margin
    return composite(frame, canvas, (x0, y0, x1, y1))


def render_anchor(frame, layer, face_pts):
    """Accessory (ears, nose...): one rotation+scale+translation from 2 anchor points."""
    height, width = frame.shape[:2]
    matrix = similarity_from_two_points(layer.points, face_pts[list(layer.anchor_labels)])
    canvas = cv2.warpAffine(layer.premult_bgra, matrix, (width, height),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return composite(frame, canvas)


def apply_layer(frame, layer, face_pts):
    return render_morph(frame, layer, face_pts) if layer.kind == "morph" \
        else render_anchor(frame, layer, face_pts)


# =============================================================================
# 7.  DRAWING / HUD HELPERS
# =============================================================================
def put_text(img, text, org, scale=0.5, colour=(255, 255, 255), thickness=1):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, HUD_FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, HUD_FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_landmarks(img, pts, show_numbers=True, radius=3, scale=0.4):
    """Draw the numbered 75 points, coloured by group."""
    for label, (x, y) in enumerate(pts):
        p = (int(round(x)), int(round(y)))
        cv2.circle(img, p, radius, LABEL_COLORS[label], -1, cv2.LINE_AA)
        if show_numbers:
            put_text(img, str(label), (p[0] + 4, p[1] - 4), scale, LABEL_COLORS[label])


# =============================================================================
# 8.  COMMANDS
# =============================================================================
def window_closed(name, frame_index):
    """True if the user closed the window with the X button (checked only after a short warm-up)."""
    return frame_index > WINDOW_WARMUP_FRAMES and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1


def open_capture(args):
    """Open a video file (--video) or the webcam."""
    if getattr(args, "video", None):
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        if REQUEST_FOURCC:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*REQUEST_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_FRAME_HEIGHT)
    if not cap.isOpened():
        print("ERROR: could not open the video source.\n"
              "  - list cameras:        ls -l /dev/video*\n"
              "  - permission problem?  groups   (you should be in 'video'; fix: sudo usermod -aG video $USER, then re-login)\n"
              "  - camera busy?         close Zoom/Teams/browser tabs using it\n"
              "  - wrong index/backend? edit CAMERA_INDEX / CAMERA_BACKEND at the top of this file")
        sys.exit(1)
    return cap


def cmd_labels(args):
    """Write labels.txt (0..74, one per line) - import it in makesense.ai so you never type labels."""
    with open(args.out, "w") as f:
        f.write("\n".join(str(i) for i in range(NUM_LANDMARKS)) + "\n")
    print("Wrote %s with labels 0..%d" % (args.out, NUM_LANDMARKS - 1))
    print("\nWhat each label means (screen-left = left side of the picture as you see it):")
    for name, first, last, _ in LANDMARK_GROUPS:
        print("  %2d-%2d  %s" % (first, last, name))


def cmd_view(args):
    """Show the 75 numbered landmarks on a photo or live webcam.  This is your annotation reference."""
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit("Cannot read " + args.image)
        landmarker = FaceLandmarker(static_image_mode=True)
        pts = landmarker.detect(img)
        if pts is None:
            sys.exit("No face found in " + args.image)
        scale = max(0.4, max(img.shape[:2]) / 1400.0)
        draw_landmarks(img, pts, True, max(2, int(scale * 4)), scale)
        out = os.path.splitext(args.image)[0] + "_landmarks.png"
        cv2.imwrite(out, img)
        print("Saved", out)
        return

    cap = open_capture(args)
    landmarker = FaceLandmarker(static_image_mode=False)
    show_numbers = True
    frame_index = 0
    print("Keys:  N = toggle numbers   S = save snapshot   Q/ESC = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)
        pts = landmarker.detect(frame)
        if pts is not None:
            draw_landmarks(frame, pts, show_numbers)
        else:
            put_text(frame, "No face detected", (10, 30), 0.7, (0, 0, 255))
        cv2.imshow(WINDOW_NAME + " - landmarks", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")) or window_closed(WINDOW_NAME + " - landmarks", frame_index):
            break
        if key == ord("n"):
            show_numbers = not show_numbers
        if key == ord("s"):
            name = "landmarks_%s.png" % time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(name, frame)
            print("Saved", name)
    cap.release()
    cv2.destroyAllWindows()


def cmd_auto(args):
    """OPTIONAL shortcut: run Face Mesh on the MASK image itself and write its annotation CSV.

    Works only when the mask looks like a real human face (photo-like masks).
    Cartoon / animal / very stylised masks -> annotate manually in makesense.ai.
    """
    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit("Cannot read " + args.image)
    if img.ndim == 3 and img.shape[2] == 4:                 # flatten alpha onto mid-grey for detection
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3] * a + 128 * (1 - a)).astype(np.uint8)
    pts = FaceLandmarker(static_image_mode=True).detect(img)
    if pts is None:
        sys.exit("No face found in the mask image - annotate it manually in makesense.ai instead.")

    out = args.out or os.path.splitext(args.image)[0] + ".csv"
    height, width = img.shape[:2]
    name = os.path.basename(args.image)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for label, (x, y) in enumerate(pts):
            writer.writerow([label, int(round(x)), int(round(y)), name, width, height])
    print("Wrote %s  (%d points).  Now run:  python ar_mask_filter.py check %s" % (out, len(pts), args.image))


def cmd_check(args):
    """Draw your annotations on the mask + the triangulation, so you can spot mistakes."""
    csv_path = args.csv or os.path.splitext(args.image)[0] + ".csv"
    layer = load_mask(args.image, csv_path)
    height, width = layer.premult_bgra.shape[:2]

    background = np.full((height, width, 3), 110, np.uint8)             # grey backdrop shows transparency
    preview = composite(background, layer.premult_bgra)

    scale = max(0.4, max(height, width) / 1500.0)
    radius = max(2, int(scale * 4))
    if layer.kind == "morph":
        for tri in layer.triangles:
            cv2.polylines(preview, [np.round(layer.points[tri]).astype(np.int32)], True,
                          (255, 255, 255), 1, cv2.LINE_AA)
        draw_landmarks(preview, layer.points, True, radius, scale)
        print("OK: full-face mask, %d points, %d triangles." % (len(layer.points), len(layer.triangles)))
    else:
        for label, (x, y) in zip(layer.anchor_labels, layer.points):
            cv2.circle(preview, (int(x), int(y)), radius + 2, (0, 255, 255), -1, cv2.LINE_AA)
            put_text(preview, str(label), (int(x) + 6, int(y) - 6), scale * 1.5, (0, 255, 255))
        print("OK: accessory anchored on face labels %s." % (layer.anchor_labels,))

    out = os.path.splitext(args.image)[0] + "_check.png"
    cv2.imwrite(out, preview)
    print("Saved", out)
    if not args.no_gui:
        try:
            cv2.imshow("annotation check (any key = close)", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("(no GUI available - open the saved PNG instead)")


def cmd_run(args):
    """The AR filter: webcam / video file / single image."""
    print("Loading masks from '%s/' ..." % MASKS_DIR)
    layers = discover_masks(MASKS_DIR, args.mask)
    if not layers:
        sys.exit("\nNo usable masks. Put  %s/<name>.png  and  %s/<name>.csv  there "
                 "(see the `labels`, `check` commands)." % (MASKS_DIR, MASKS_DIR))
    names = list(layers)
    current = 0
    show_points = SHOW_LANDMARKS_AT_START

    # ---------------- single image mode (no webcam needed) ----------------
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit("Cannot read " + args.image)
        landmarker = FaceLandmarker(static_image_mode=True)
        pts = landmarker.detect(frame)
        if pts is None:
            sys.exit("No face found in " + args.image)
        out = apply_layer(frame, layers[names[current]], pts)
        path = args.out or os.path.splitext(args.image)[0] + "_filtered.png"
        cv2.imwrite(path, out)
        print("Saved", path)
        return

    # ---------------- video / webcam mode ----------------
    cap = open_capture(args)
    landmarker = FaceLandmarker(static_image_mode=False)
    stabilizer = LandmarkStabilizer()
    writer = None
    fps_smooth, last_t, frame_index = None, time.time(), 0
    print("Keys:  F = next filter   L = show landmarks   S = screenshot   Q/ESC = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if MIRROR_VIEW and not args.video:                   # never mirror recorded files
            frame = cv2.flip(frame, 1)

        pts = landmarker.detect(frame)
        if pts is None:
            stabilizer.reset()                               # forget old points; show the plain frame
            output = frame
            put_text(output, "No face detected", (10, 55), 0.6, (0, 0, 255))
        else:
            if USE_STABILIZATION:
                pts = stabilizer.update(frame, pts)
            output = apply_layer(frame, layers[names[current]], pts)
            if show_points:
                draw_landmarks(output, pts)

        # FPS + help line
        now = time.time()
        inst_fps = 1.0 / max(now - last_t, 1e-6)
        fps_smooth = inst_fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * inst_fps
        last_t = now
        put_text(output, "[%d/%d] %s | F next  L landmarks  S save  Q quit | %.0f fps"
                 % (current + 1, len(names), names[current], fps_smooth), (10, 20))

        if args.out:
            if writer is None:
                h, w = output.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 30.0, (w, h))
            writer.write(output)

        if args.no_gui:
            continue
        cv2.imshow(WINDOW_NAME, output)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")) or window_closed(WINDOW_NAME, frame_index):
            break
        if key == ord("f"):
            current = (current + 1) % len(names)
        elif key == ord("l"):
            show_points = not show_points
        elif key == ord("s"):
            shot = "screenshot_%s.png" % time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(shot, output)
            print("Saved", shot)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# =============================================================================
# 9.  ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="AR face-mask filter (MediaPipe Face Mesh + OpenCV)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("labels", help="write labels.txt (0..74) for makesense.ai")
    p.add_argument("--out", default=LABELS_FILE)
    p.set_defaults(func=cmd_labels)

    p = sub.add_parser("view", help="show the 75 numbered landmarks (webcam or --image)")
    p.add_argument("--image", help="draw the landmarks on a photo instead of the webcam")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("auto", help="auto-annotate a realistic-face mask PNG (writes <name>.csv)")
    p.add_argument("image")
    p.add_argument("--out")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser("check", help="draw annotations + triangulation on a mask PNG")
    p.add_argument("image")
    p.add_argument("--csv")
    p.add_argument("--no-gui", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="run the AR filter")
    p.add_argument("--mask", help="only load masks/<MASK>.png")
    p.add_argument("--image", help="filter one photo instead of the webcam")
    p.add_argument("--video", help="filter a video file instead of the webcam")
    p.add_argument("--out", help="save result (png for --image, mp4 for video/webcam)")
    p.add_argument("--no-gui", action="store_true", help="do not open a window (headless)")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
