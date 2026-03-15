"""
detector.py — Core detection engine

Engine priority:
  1. OpenCV DNN  (res10 SSD — accurate, sub-pixel confidence)
  2. Haar Cascade fallback (if model files not downloaded yet)

New in v2:
  · DNN detection with confidence scores
  · Presence heatmap accumulation
  · Recording-active flag for HUD indicator
"""

import cv2
import numpy as np
import time
import os

from model_manager import PROTOTXT_PATH, CAFFEMODEL_PATH, models_exist

# ─── BGR colour constants ─────────────────────────────────────────────────────
C_GREEN      = (0, 255, 65)
C_GREEN2     = (0, 200, 45)
C_GREEN_DIM  = (0, 120, 30)
C_YELLOW     = (0, 240, 200)
C_CYAN       = (200, 255, 80)
C_RED        = (50, 50, 255)
C_WHITE      = (220, 220, 220)
C_GRAY       = (90,  90,  90)
C_DARK       = (20,  20,  20)

CORNER_LEN   = 20
CORNER_THICK = 2
HUD_FONT     = cv2.FONT_HERSHEY_SIMPLEX


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def _corner_box(img, x, y, w, h, color, thick=CORNER_THICK):
    for (px, py), (dx, dy) in [
        ((x,     y    ), ( 1,  1)),
        ((x + w, y    ), (-1,  1)),
        ((x,     y + h), ( 1, -1)),
        ((x + w, y + h), (-1, -1)),
    ]:
        cv2.line(img, (px, py), (px + dx * CORNER_LEN, py),       color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px,       py + dy * CORNER_LEN), color, thick, cv2.LINE_AA)


def _label(img, text, x, y, color, bg=(0, 0, 0)):
    s, t = 0.42, 1
    (tw, th), _ = cv2.getTextSize(text, HUD_FONT, s, t)
    p = 3
    cv2.rectangle(img, (x, y - th - p * 2), (x + tw + p * 2, y), bg, -1)
    cv2.putText(img, text, (x + p, y - p), HUD_FONT, s, color, t, cv2.LINE_AA)


def _conf_bar(img, x, y, conf, w=50, h=4):
    """Tiny horizontal confidence bar under face label."""
    cv2.rectangle(img, (x, y), (x + w, y + h), C_DARK, -1)
    fill = int(conf * w)
    bar_color = C_GREEN if conf > 0.75 else (C_YELLOW if conf > 0.5 else C_RED)
    cv2.rectangle(img, (x, y), (x + fill, y + h), bar_color, -1)


def _scanlines(img, alpha=0.03):
    h, w = img.shape[:2]
    ov = img.copy()
    for row in range(0, h, 4):
        cv2.line(ov, (0, row), (w, row), (0, 0, 0), 1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _hud(img, fps, faces, mode, elapsed, engine, recording):
    h, w = img.shape[:2]
    mins, secs = divmod(int(elapsed), 60)

    # ── top-left info panel ───────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (230, 100), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (230, 100), C_GREEN_DIM, 1)

    engine_color = C_CYAN if engine == "DNN" else C_YELLOW
    rows = [
        ("FACE SENTINEL v2.0", C_GREEN,   0.40),
        (f"ENGINE : {engine}",  engine_color, 0.40),
        (f"FPS    : {fps:>5.1f}", C_WHITE, 0.40),
        (f"FACES  : {faces:>5d}", C_GREEN, 0.40),
        (f"MODE   : {mode.upper():<10}", C_YELLOW, 0.40),
    ]
    for i, (txt, col, sc) in enumerate(rows):
        cv2.putText(img, txt, (8, 18 + i * 16), HUD_FONT, sc, col, 1, cv2.LINE_AA)

    # ── bottom-right timestamp ────────────────────────────────────────────────
    ts  = f"SESSION  {mins:02d}:{secs:02d}"
    (tw, _), _ = cv2.getTextSize(ts, HUD_FONT, 0.38, 1)
    cv2.putText(img, ts, (w - tw - 8, h - 8), HUD_FONT, 0.38, C_GRAY, 1, cv2.LINE_AA)

    # ── corner accents ────────────────────────────────────────────────────────
    cv2.line(img, (w - 1, 0), (w - 30, 0),  C_GREEN, 2)
    cv2.line(img, (w - 1, 0), (w - 1,  30), C_GREEN, 2)
    cv2.line(img, (0, h - 1), (30, h - 1),  C_GREEN_DIM, 1)
    cv2.line(img, (0, h - 1), (0,  h - 30), C_GREEN_DIM, 1)

    # ── REC indicator ─────────────────────────────────────────────────────────
    if recording:
        cv2.circle(img, (w - 18, 18), 7, C_RED, -1)
        cv2.putText(img, "REC", (w - 48, 23), HUD_FONT, 0.40, C_RED, 1, cv2.LINE_AA)


# ─── Heatmap helpers ──────────────────────────────────────────────────────────

def _build_green_lut() -> np.ndarray:
    """256-entry BGR LUT: black → forest green → bright green → yellow-green."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.35:
            g = int(120 * (t / 0.35))
            lut[i] = [0, g, 0]
        elif t < 0.70:
            tt = (t - 0.35) / 0.35
            lut[i] = [0, int(120 + 135 * tt), int(30 * tt)]
        else:
            tt = (t - 0.70) / 0.30
            lut[i] = [int(220 * tt), 255, int(30 + 180 * tt)]
    return lut

_GREEN_LUT = _build_green_lut()          # pre-built once at import


def _apply_heatmap(frame: np.ndarray, heat: np.ndarray, alpha=0.45) -> np.ndarray:
    """Blend the accumulated heatmap onto *frame*."""
    norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # apply custom green LUT
    colored = np.empty((*norm.shape, 3), dtype=np.uint8)
    colored[..., 0] = _GREEN_LUT[norm, 0]
    colored[..., 1] = _GREEN_LUT[norm, 1]
    colored[..., 2] = _GREEN_LUT[norm, 2]
    # mask: only blend where heat is meaningful (> 5 %)
    mask = norm > 12
    out  = frame.copy()
    out[mask] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[mask]
    return out


# ─── Main detector class ──────────────────────────────────────────────────────

class FaceDetector:
    """
    Dual-engine face detector (DNN preferred, Haar Cascade fallback).

    Public attributes
    -----------------
    mode              : 'face' | 'eyes' | 'smile' | 'all'
    confidence        : float — DNN confidence threshold (0–1)
    scale             : float — Haar scale factor
    min_neighbors     : int   — Haar min-neighbors
    show_heatmap      : bool
    scanlines         : bool
    is_recording      : bool  — drives HUD REC indicator
    """

    # ── Haar cascade paths ────────────────────────────────────────────────────
    _HAAR = {
        "face"   : cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        "profile": cv2.data.haarcascades + "haarcascade_profileface.xml",
        "eyes"   : cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml",
        "smile"  : cv2.data.haarcascades + "haarcascade_smile.xml",
    }

    def __init__(self):
        # ── engine ────────────────────────────────────────────────────────────
        self._net    = None
        self._engine = "HAAR"
        self._haar   = {k: cv2.CascadeClassifier(v) for k, v in self._HAAR.items()}
        self._try_load_dnn()

        # ── settings ──────────────────────────────────────────────────────────
        self.mode          = "face"
        self.confidence    = 0.50
        self.scale         = 1.15
        self.min_neighbors = 5
        self.show_heatmap  = False
        self.scanlines     = True
        self.is_recording  = False

        # ── heatmap state ─────────────────────────────────────────────────────
        self._heat: np.ndarray | None = None
        self._heat_size: tuple[int, int] = (0, 0)
        self.HEAT_DECAY = 0.982       # per-frame decay (≈ 3-4 s memory)

        # ── stats ─────────────────────────────────────────────────────────────
        self._fps_buf: list[float] = []
        self._t_prev   = time.time()
        self._t_start  = time.time()
        self.total_detections = 0
        self.max_faces        = 0
        self.frames_processed = 0

    # ── private ───────────────────────────────────────────────────────────────

    def _try_load_dnn(self):
        if models_exist():
            try:
                self._net    = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
                self._engine = "DNN"
            except Exception as e:
                print(f"[detector] DNN load failed: {e}  — falling back to Haar.")

    def _fps(self) -> float:
        now = time.time()
        dt  = now - self._t_prev
        self._t_prev = now
        self._fps_buf.append(1.0 / dt if dt > 0 else 0.0)
        if len(self._fps_buf) > 30:
            self._fps_buf.pop(0)
        return float(np.mean(self._fps_buf))

    # ── DNN detection ─────────────────────────────────────────────────────────

    def _detect_dnn(self, frame: np.ndarray) -> list[tuple[int,int,int,int,float]]:
        """Returns list of (x, y, w, h, confidence)."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0), swapRB=False)
        self._net.setInput(blob)
        dets = self._net.forward()  # shape (1,1,N,7)

        faces = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < self.confidence:
                continue
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1, conf))
        return faces

    # ── Haar detection ────────────────────────────────────────────────────────

    def _detect_haar(self, gray: np.ndarray) -> list[tuple[int,int,int,int,float]]:
        raw = self._haar["face"].detectMultiScale(
            gray, scaleFactor=self.scale,
            minNeighbors=self.min_neighbors, minSize=(28, 28),
            flags=cv2.CASCADE_SCALE_IMAGE)
        faces: list = []
        if len(raw):
            faces = [(x, y, w, h, 1.0) for (x, y, w, h) in raw]

        if len(faces) < 2:
            prof = self._haar["profile"].detectMultiScale(
                gray, scaleFactor=self.scale,
                minNeighbors=self.min_neighbors, minSize=(28, 28))
            if len(prof):
                for (x, y, w, h) in prof:
                    faces.append((x, y, w, h, 0.85))
        return faces

    # ── Heatmap ───────────────────────────────────────────────────────────────

    def _update_heatmap(self, frame_h: int, frame_w: int,
                        faces: list[tuple]) -> None:
        if self._heat_size != (frame_h, frame_w):
            self._heat      = np.zeros((frame_h, frame_w), dtype=np.float32)
            self._heat_size = (frame_h, frame_w)

        # decay
        self._heat *= self.HEAT_DECAY

        # add blobs
        for (fx, fy, fw, fh, *_) in faces:
            cx, cy = fx + fw // 2, fy + fh // 2
            r      = max(fw, fh) // 2
            cv2.circle(self._heat, (cx, cy), r, 1.0, -1)

    # ── Public API ────────────────────────────────────────────────────────────

    def reload_dnn(self):
        """Call after models are downloaded to switch engine without restart."""
        self._try_load_dnn()

    @property
    def engine(self) -> str:
        return self._engine

    def reset_stats(self):
        self.total_detections = 0
        self.max_faces        = 0
        self.frames_processed = 0
        self._fps_buf.clear()
        self._t_prev  = time.time()
        self._t_start = time.time()
        if self._heat is not None:
            self._heat[:] = 0.0

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Run full detection pipeline on *frame* (BGR).
        Returns (annotated_frame, stats_dict).
        """
        if frame is None:
            return frame, {}

        fps     = self._fps()
        elapsed = time.time() - self._t_start
        h, w    = frame.shape[:2]
        result  = frame.copy()

        # ── detect faces ──────────────────────────────────────────────────────
        if self._engine == "DNN":
            faces = self._detect_dnn(frame)
        else:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = self._detect_haar(gray)

        face_count = len(faces)
        self.total_detections += face_count
        self.frames_processed += 1
        if face_count > self.max_faces:
            self.max_faces = face_count

        # ── heatmap update ────────────────────────────────────────────────────
        self._update_heatmap(h, w, faces)

        # ── draw heatmap (before boxes so boxes sit on top) ───────────────────
        if self.show_heatmap and self._heat is not None and self._heat.max() > 0:
            result = _apply_heatmap(result, self._heat)

        # ── draw each face ────────────────────────────────────────────────────
        gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.equalizeHist(gray_roi)

        for i, (fx, fy, fw, fh, *conf_list) in enumerate(faces):
            conf = conf_list[0] if conf_list else 1.0

            # subtle fill
            ov = result.copy()
            cv2.rectangle(ov, (fx, fy), (fx+fw, fy+fh), C_GREEN_DIM, -1)
            cv2.addWeighted(ov, 0.07, result, 0.93, 0, result)

            # corner box
            _corner_box(result, fx, fy, fw, fh, C_GREEN)

            # label + confidence bar
            label = f"FACE #{i+1}  {conf:.0%}" if self._engine == "DNN" else f"FACE #{i+1}"
            _label(result, label, fx, fy, C_GREEN)
            if self._engine == "DNN":
                _conf_bar(result, fx, fy - 10, conf)

            # sub-detections inside face ROI
            roi_g = gray_roi[fy:fy+fh, fx:fx+fw]
            roi_c = result  [fy:fy+fh, fx:fx+fw]

            if self.mode in ("eyes", "all"):
                eyes = self._haar["eyes"].detectMultiScale(
                    roi_g, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
                for (ex, ey, ew, eh) in eyes:
                    cx2, cy2 = ex + ew // 2, ey + eh // 2
                    cv2.circle(roi_c, (cx2, cy2), max(ew, eh)//2, C_YELLOW, 1)
                    cv2.circle(roi_c, (cx2, cy2), 2,               C_YELLOW, -1)

            if self.mode in ("smile", "all"):
                smiles = self._haar["smile"].detectMultiScale(
                    roi_g, scaleFactor=1.7, minNeighbors=22, minSize=(25, 15))
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_c, (sx, sy), (sx+sw, sy+sh), C_RED, 1)
                    _label(roi_c, "SMILE", sx, sy, C_RED)

        # ── atmospheric scanlines ─────────────────────────────────────────────
        if self.scanlines:
            _scanlines(result)

        # ── HUD overlay ───────────────────────────────────────────────────────
        _hud(result, fps, face_count, self.mode, elapsed,
             self._engine, self.is_recording)

        stats = {
            "fps"        : fps,
            "face_count" : face_count,
            "elapsed"    : elapsed,
            "total_det"  : self.total_detections,
            "max_faces"  : self.max_faces,
            "engine"     : self._engine,
        }
        return result, stats
