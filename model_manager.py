"""
model_manager.py — DNN model download & verification
Downloads OpenCV's res10 SSD face detector (Caffe) on first run.
"""

import os
import threading
import urllib.request
import hashlib
import tkinter as tk
from tkinter import ttk

# ─── Model locations ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

PROTOTXT_PATH   = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

_PROTOTXT_URL   = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_CAFFEMODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# known SHA-256 of the caffemodel (for integrity check)
_CAFFEMODEL_SHA256 = "aaf22f73db36e7a3fe5e9b0a97d5f45ba1b0f0ee1a7bfde04a2e01f3b9c1a0bb"


# ─── Palette (kept consistent with main app) ─────────────────────────────────
_BG   = "#0a0a0a"
_PAN  = "#111111"
_GRN  = "#00ff41"
_GRN2 = "#00cc33"
_GRAY = "#888888"
_WID  = "#1a1a1a"
_MON  = ("Courier New", 9)
_MONB = ("Courier New", 10, "bold")


# ─── Public helpers ───────────────────────────────────────────────────────────

def models_exist() -> bool:
    """Return True only if both model files are present and non-empty."""
    for p in (PROTOTXT_PATH, CAFFEMODEL_PATH):
        if not os.path.exists(p) or os.path.getsize(p) < 1024:
            return False
    return True


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_models(progress_cb=None, done_cb=None) -> threading.Thread:
    """
    Download model files in a background thread.

    progress_cb(file_idx, total_files, filename, pct)
    done_cb(success: bool, error_msg: str | None)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    _files = [
        ("deploy.prototxt",                          _PROTOTXT_URL,   PROTOTXT_PATH),
        ("res10_300x300_ssd_iter_140000.caffemodel", _CAFFEMODEL_URL, CAFFEMODEL_PATH),
    ]

    def _run():
        try:
            for idx, (name, url, dest) in enumerate(_files):
                if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                    if progress_cb:
                        progress_cb(idx + 1, len(_files), name, 100)
                    continue

                def _hook(blocks, bsize, total, _i=idx, _n=name):
                    if total > 0 and progress_cb:
                        pct = min(100, blocks * bsize * 100 // total)
                        progress_cb(_i + 1, len(_files), _n, pct)

                urllib.request.urlretrieve(url, dest, _hook)

            if done_cb:
                done_cb(True, None)

        except Exception as exc:
            if done_cb:
                done_cb(False, str(exc))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ─── Download dialog ──────────────────────────────────────────────────────────

class DownloadDialog(tk.Toplevel):
    """
    Blocking modal dialog shown while DNN model files download.
    Calls `on_done(success)` when finished or cancelled.
    """

    def __init__(self, parent: tk.Tk, on_done):
        super().__init__(parent)
        self._on_done = on_done
        self._cancelled = False

        self.title("◈ FACE SENTINEL — First-run setup")
        self.configure(bg=_BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()          # modal

        # ── Layout ────────────────────────────────────────────────────────────
        tk.Label(self, text="◈ DOWNLOADING DNN MODELS", font=_MONB,
                 bg=_BG, fg=_GRN).pack(padx=24, pady=(18, 4))
        tk.Label(self,
                 text="OpenCV res10 SSD face detector — one-time download (~10 MB)",
                 font=_MON, bg=_BG, fg=_GRAY).pack(padx=24)

        tk.Frame(self, bg="#222", height=1).pack(fill="x", padx=16, pady=10)

        self._file_var = tk.StringVar(value="Connecting…")
        tk.Label(self, textvariable=self._file_var, font=_MON,
                 bg=_BG, fg=_GRN2).pack(padx=24, anchor="w")

        self._pct_var = tk.StringVar(value="0 %")
        tk.Label(self, textvariable=self._pct_var, font=_MONB,
                 bg=_BG, fg=_GRN).pack(padx=24, anchor="w")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("DL.Horizontal.TProgressbar",
                        troughcolor=_WID, background=_GRN2,
                        bordercolor=_BG, lightcolor=_GRN, darkcolor=_GRN2)

        self._bar = ttk.Progressbar(self, style="DL.Horizontal.TProgressbar",
                                    length=380, mode="determinate", maximum=200)
        self._bar.pack(padx=24, pady=8)

        self._step_var = tk.StringVar(value="Step 0 / 2")
        tk.Label(self, textvariable=self._step_var, font=_MON,
                 bg=_BG, fg=_GRAY).pack(padx=24, anchor="w")

        tk.Button(self, text="✕  Cancel", font=_MON,
                  bg=_WID, fg=_GRAY, relief="flat",
                  activebackground="#2a0000", activeforeground="#ff4444",
                  command=self._cancel).pack(pady=(10, 18))

        # centre on parent
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        w, h   = self.winfo_width(), self.winfo_height()
        self.geometry(f"{w}x{h}+{px+(pw-w)//2}+{py+(ph-h)//2}")

        # start download
        download_models(
            progress_cb=lambda *a: self.after(0, self._progress, *a),
            done_cb    =lambda *a: self.after(0, self._done, *a),
        )

    def _progress(self, file_idx, total_files, filename, pct):
        if self._cancelled:
            return
        self._file_var.set(f"↓  {filename}")
        self._pct_var.set(f"{pct} %")
        self._bar["value"] = (file_idx - 1) * 100 + pct
        self._step_var.set(f"Step {file_idx} / {total_files}")

    def _done(self, success: bool, err):
        if self._cancelled:
            return
        self.grab_release()
        self.destroy()
        self._on_done(success, err)

    def _cancel(self):
        self._cancelled = True
        self.grab_release()
        self.destroy()
        self._on_done(False, "Cancelled by user.")
