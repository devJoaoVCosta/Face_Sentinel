"""
report_html.py — Generates a self-contained index.html face detection report.

Each detected face entry contains:
  · Timestamp of detection
  · Camera index
  · Face number in that frame
  · Bounding-box dimensions (w × h px)
  · Detection confidence (DNN) or "Haar" label
  · Base64-encoded JPEG crop of the face (embedded directly in the HTML)

The output is a single .html file with no external dependencies.
"""

from __future__ import annotations

import base64
import datetime
import cv2
import numpy as np
from dataclasses import dataclass, field


# ─── Data record ─────────────────────────────────────────────────────────────

@dataclass
class FaceRecord:
    """One detected face — everything needed to build the HTML card."""
    timestamp:    str          # ISO-8601 string
    camera_index: int
    face_number:  int          # within the frame (0-based → display 1-based)
    confidence:   float        # 0.0–1.0  (1.0 for Haar)
    bbox_w:       int
    bbox_h:       int
    engine:       str          # "DNN" | "HAAR"
    b64_jpeg:     str          # base64-encoded JPEG of the cropped face


def crop_to_b64(face_bgr: np.ndarray, size: int = 120) -> str:
    """Resize a BGR face crop to *size*×*size* and encode as base64 JPEG."""
    resized = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


# ─── HTML template ────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d0d0d;
  color: #e0e0e0;
  font-family: 'Courier New', Courier, monospace;
  font-size: 13px;
  padding: 24px;
}
h1 {
  color: #00ff41;
  font-size: 22px;
  letter-spacing: 3px;
  margin-bottom: 6px;
}
.meta {
  color: #666;
  font-size: 11px;
  margin-bottom: 24px;
  border-bottom: 1px solid #222;
  padding-bottom: 12px;
}
.meta span { color: #00cc33; }

/* Summary bar */
.summary {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 28px;
}
.stat-card {
  background: #111;
  border: 1px solid #222;
  border-bottom: 2px solid #00cc33;
  padding: 10px 18px;
  min-width: 130px;
}
.stat-card .label { color: #666; font-size: 10px; letter-spacing: 1px; }
.stat-card .value { color: #00ff41; font-size: 20px; font-weight: bold; margin-top: 2px; }

/* Filter bar */
.filters {
  margin-bottom: 20px;
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}
.filters label { color: #666; font-size: 11px; }
.filters select, .filters input {
  background: #1a1a1a;
  border: 1px solid #333;
  color: #00ff41;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  padding: 4px 8px;
}
#search-box { width: 200px; }

/* Face grid */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 14px;
}
.card {
  background: #111;
  border: 1px solid #1e1e1e;
  border-top: 2px solid #00cc33;
  padding: 0;
  overflow: hidden;
  transition: border-color .15s;
}
.card:hover { border-top-color: #00ff41; }
.card img {
  width: 100%;
  display: block;
  image-rendering: pixelated;
  filter: brightness(.95) contrast(1.05);
}
.card .info {
  padding: 10px 12px;
}
.card .face-id {
  color: #00ff41;
  font-size: 13px;
  font-weight: bold;
  margin-bottom: 6px;
  letter-spacing: 1px;
}
.card .row { display: flex; justify-content: space-between; margin-top: 3px; }
.card .key { color: #555; font-size: 10px; }
.card .val { color: #aaa; font-size: 10px; }
.card .conf-bar-wrap {
  background: #1a1a1a;
  height: 3px;
  margin-top: 8px;
  border-radius: 1px;
}
.card .conf-bar {
  height: 3px;
  border-radius: 1px;
  background: #00cc33;
}
.card .engine-badge {
  display: inline-block;
  font-size: 9px;
  padding: 1px 6px;
  margin-top: 6px;
  border: 1px solid;
}
.engine-dnn  { color: #80ffcc; border-color: #80ffcc33; }
.engine-haar { color: #c8ff00; border-color: #c8ff0033; }

.empty {
  color: #333;
  text-align: center;
  padding: 60px 0;
  font-size: 16px;
  letter-spacing: 2px;
}
footer {
  margin-top: 40px;
  padding-top: 12px;
  border-top: 1px solid #1a1a1a;
  color: #333;
  font-size: 10px;
  letter-spacing: 1px;
}
"""

_JS = """
(function() {
  var allCards = [];

  function init() {
    allCards = Array.from(document.querySelectorAll('.card'));
    document.getElementById('cam-filter').addEventListener('change', applyFilters);
    document.getElementById('engine-filter').addEventListener('change', applyFilters);
    document.getElementById('search-box').addEventListener('input', applyFilters);
  }

  function applyFilters() {
    var cam    = document.getElementById('cam-filter').value;
    var eng    = document.getElementById('engine-filter').value;
    var search = document.getElementById('search-box').value.toLowerCase();

    allCards.forEach(function(c) {
      var d = c.dataset;
      var matchCam = (cam === 'all' || d.cam === cam);
      var matchEng = (eng === 'all' || d.engine === eng);
      var matchSrc = (search === '' || c.textContent.toLowerCase().includes(search));
      c.style.display = (matchCam && matchEng && matchSrc) ? '' : 'none';
    });

    var visible = allCards.filter(function(c) { return c.style.display !== 'none'; });
    document.getElementById('visible-count').textContent = visible.length;
  }

  document.addEventListener('DOMContentLoaded', init);
})();
"""


def _conf_color(conf: float, engine: str) -> str:
    if engine == "HAAR":
        return "#c8ff00"
    if conf >= 0.80:
        return "#00ff41"
    if conf >= 0.60:
        return "#80ffcc"
    return "#c8ff00"


def generate_html(records: list[FaceRecord],
                  session_start: str,
                  session_end: str,
                  engine: str) -> str:
    """
    Build and return the complete HTML string.

    Parameters
    ----------
    records       : list of FaceRecord (chronological)
    session_start : ISO datetime string
    session_end   : ISO datetime string
    engine        : "DNN" | "HAAR"
    """
    total   = len(records)
    cameras = sorted({r.camera_index for r in records})
    avg_conf = (sum(r.confidence for r in records) / total) if total else 0.0
    max_sim  = 0
    # rough max simultaneous: group by timestamp prefix (second)
    from collections import Counter
    ts_counts = Counter(r.timestamp[:19] for r in records)
    max_sim   = max(ts_counts.values(), default=0)

    # ── camera filter options ─────────────────────────────────────────────────
    cam_opts = '<option value="all">All cameras</option>\n'
    for c in cameras:
        cam_opts += f'<option value="{c}">Camera {c}</option>\n'

    # ── face cards ────────────────────────────────────────────────────────────
    if not records:
        cards_html = '<div class="empty">◈ NO FACES RECORDED ◈</div>'
    else:
        cards = []
        for i, r in enumerate(records):
            conf_pct  = f"{r.confidence:.0%}" if r.engine == "DNN" else "N/A"
            conf_w    = int(r.confidence * 100) if r.engine == "DNN" else 100
            bar_color = _conf_color(r.confidence, r.engine)
            eng_cls   = "engine-dnn" if r.engine == "DNN" else "engine-haar"
            img_src   = f"data:image/jpeg;base64,{r.b64_jpeg}" if r.b64_jpeg else ""
            img_tag   = (f'<img src="{img_src}" alt="Face crop" width="120" height="120">'
                         if img_src else
                         '<div style="height:120px;background:#0a0a0a;display:flex;'
                         'align-items:center;justify-content:center;color:#333">'
                         'NO IMAGE</div>')

            card = f"""
<div class="card" data-cam="{r.camera_index}" data-engine="{r.engine}">
  {img_tag}
  <div class="info">
    <div class="face-id">FACE #{i+1:04d}</div>
    <div class="row"><span class="key">TIMESTAMP</span>
                     <span class="val">{r.timestamp[11:19]}</span></div>
    <div class="row"><span class="key">DATE</span>
                     <span class="val">{r.timestamp[:10]}</span></div>
    <div class="row"><span class="key">CAMERA</span>
                     <span class="val">index {r.camera_index}</span></div>
    <div class="row"><span class="key">FACE IN FRAME</span>
                     <span class="val">#{r.face_number + 1}</span></div>
    <div class="row"><span class="key">BBOX</span>
                     <span class="val">{r.bbox_w} × {r.bbox_h} px</span></div>
    <div class="row"><span class="key">CONFIDENCE</span>
                     <span class="val" style="color:{bar_color}">{conf_pct}</span></div>
    <div class="conf-bar-wrap">
      <div class="conf-bar" style="width:{conf_w}%;background:{bar_color}"></div>
    </div>
    <span class="engine-badge {eng_cls}">{r.engine}</span>
  </div>
</div>"""
            cards.append(card)
        cards_html = '\n'.join(cards)

    # ── assemble ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Face Sentinel — Detection Report</title>
  <style>{_CSS}</style>
</head>
<body>

<h1>◈ FACE SENTINEL — Detection Report</h1>
<div class="meta">
  Generated: <span>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
  &nbsp;|&nbsp; Session: <span>{session_start}</span> → <span>{session_end}</span>
  &nbsp;|&nbsp; Engine: <span>{engine}</span>
</div>

<div class="summary">
  <div class="stat-card">
    <div class="label">TOTAL FACES</div>
    <div class="value">{total}</div>
  </div>
  <div class="stat-card">
    <div class="label">CAMERAS</div>
    <div class="value">{len(cameras)}</div>
  </div>
  <div class="stat-card">
    <div class="label">AVG CONFIDENCE</div>
    <div class="value">{avg_conf:.0%}</div>
  </div>
  <div class="stat-card">
    <div class="label">MAX SIMULTANEOUS</div>
    <div class="value">{max_sim}</div>
  </div>
  <div class="stat-card">
    <div class="label">ENGINE</div>
    <div class="value" style="font-size:14px">{engine}</div>
  </div>
</div>

<div class="filters">
  <label>CAMERA</label>
  <select id="cam-filter">{cam_opts}</select>
  <label>ENGINE</label>
  <select id="engine-filter">
    <option value="all">All</option>
    <option value="DNN">DNN</option>
    <option value="HAAR">HAAR</option>
  </select>
  <label>SEARCH</label>
  <input id="search-box" type="text" placeholder="timestamp, size…">
  <span style="color:#555;font-size:11px">showing <span id="visible-count">{total}</span> of {total}</span>
</div>

<div class="grid">
{cards_html}
</div>

<footer>◈ FACE SENTINEL v2.0 &nbsp;·&nbsp; OpenCV DNN / Haar Cascade
&nbsp;·&nbsp; Report generated {datetime.datetime.now().isoformat(timespec='seconds')}</footer>

<script>{_JS}</script>
</body>
</html>"""
    return html
