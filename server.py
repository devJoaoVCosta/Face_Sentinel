"""
server.py — Face Sentinel Web Server

Serves the full application at http://localhost:5000
No extra services required — just Python + Flask + OpenCV.

Routes
------
GET  /                      → redirect to /login or /dashboard
GET  /login                 → login page
POST /api/auth/login        → authenticate, set session
POST /api/auth/logout       → clear session
GET  /dashboard             → main SPA (requires login)
GET  /video_feed/<cam_id>   → MJPEG stream with detection overlay
GET  /api/stream/stats      → SSE: per-camera face count + FPS
GET  /api/cameras/scan      → scan for available cameras
POST /api/cameras/open      → open selected cameras
POST /api/cameras/close     → close one camera
GET  /api/reports/faces     → face records for current user
POST /api/reports/push      → push to GitHub Pages
GET  /api/users             → list users (master only)
POST /api/users             → create user (master only)
DELETE /api/users/<name>    → delete user (master only)
GET  /api/config/github     → get GitHub config
POST /api/config/github     → save GitHub config
POST /api/config/github/test → test GitHub connection
"""

import cv2
import threading
import time
import datetime
import os
import json
import base64
import webbrowser
from functools import wraps
from pathlib import Path

from flask import (Flask, render_template, request, jsonify,
                   Response, session, redirect, url_for, stream_with_context)

from user_manager   import UserManager
from detector       import FaceDetector
from camera_manager import CameraManager, CameraInfo, scan_cameras
from github_publisher import GitHubPublisher, GitHubConfig, load_config, save_config
from report_html    import FaceRecord, crop_to_b64

# ─── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# ─── Global state ─────────────────────────────────────────────────────────────

um          = UserManager()
cam_mgr     = CameraManager()
gh_config   = load_config()

# per-camera detectors  {cam_idx: FaceDetector}
detectors:  dict[int, FaceDetector] = {}
# last annotated BGR frame per camera  {cam_idx: bytes}  (JPEG)
last_jpeg:  dict[int, bytes]        = {}
# per-camera live stats  {cam_idx: dict}
cam_stats:  dict[int, dict]         = {}
# per-user face records  {username: [FaceRecord]}
face_records: dict[str, list[FaceRecord]] = {}
# throttle last-capture timestamp per (user, cam)
last_capture: dict[tuple, float] = {}

_FACE_INTERVAL = 2.0    # seconds between face captures
_MAX_RECORDS   = 500

# ─── Auth helpers ─────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "not_authenticated"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

def master_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "master":
            return jsonify({"error": "forbidden"}), 403
        return f(*args, **kwargs)
    return decorated

def current_user() -> str:
    return session.get("username", "unknown")

# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page"))

@app.route("/login")
def login_page():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    user = um.get_user(current_user())
    return render_template("dashboard.html",
                           username=user.username,
                           role=user.role,
                           is_master=user.is_master)

# ─── Auth API ─────────────────────────────────────────────────────────────────

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data     = request.get_json(force=True) or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    user     = um.authenticate(username, password)
    if not user:
        return jsonify({"ok": False, "error": "Credenciais inválidas."}), 401
    session["username"] = user.username
    session["role"]     = user.role
    return jsonify({"ok": True, "username": user.username, "role": user.role})

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/auth/me")
@login_required
def api_me():
    return jsonify({"username": current_user(), "role": session.get("role")})

# ─── Camera scanning ──────────────────────────────────────────────────────────

@app.route("/api/cameras/scan")
@login_required
def api_cameras_scan():
    """Scan cameras synchronously (called once from frontend)."""
    results = scan_cameras(max_index=9)
    return jsonify([{
        "index": c.index,
        "label": c.label,
        "width": c.width,
        "height": c.height,
        "fps":   c.fps,
    } for c in results])

@app.route("/api/cameras/open", methods=["POST"])
@login_required
def api_cameras_open():
    data    = request.get_json(force=True) or {}
    indices = [int(i) for i in data.get("indices", [])]
    errors  = cam_mgr.open(indices)
    opened  = []
    for idx in indices:
        if idx not in errors:
            if idx not in detectors:
                detectors[idx] = FaceDetector()
            opened.append(idx)
    return jsonify({"opened": opened, "errors": errors})

@app.route("/api/cameras/close", methods=["POST"])
@login_required
def api_cameras_close():
    data = request.get_json(force=True) or {}
    idx  = int(data.get("index", -1))
    cam_mgr.close(idx)
    detectors.pop(idx, None)
    last_jpeg.pop(idx, None)
    cam_stats.pop(idx, None)
    return jsonify({"ok": True})

@app.route("/api/cameras/active")
@login_required
def api_cameras_active():
    result = []
    for idx in cam_mgr.active_indices:
        info = cam_mgr.get_info(idx)
        result.append({
            "index": idx,
            "label": str(info) if info else f"Camera {idx}",
        })
    return jsonify(result)

# ─── MJPEG Stream ─────────────────────────────────────────────────────────────

def _detection_settings() -> dict:
    """Read detection settings from query string.
    JS sends conf as 0-99 int, scale as 105-150 int — divide by 100.
    """
    raw_conf  = request.args.get("conf",     "50")
    raw_scale = request.args.get("scale",    "115")
    # if value > 1 it was sent as integer percentage → normalise
    conf  = float(raw_conf)  / 100.0 if float(raw_conf)  > 1 else float(raw_conf)
    scale = float(raw_scale) / 100.0 if float(raw_scale) > 2 else float(raw_scale)
    return {
        "mode":          request.args.get("mode",  "face"),
        "confidence":    max(0.1, min(0.99, conf)),
        "scale":         max(1.05, min(1.50, scale)),
        "min_neighbors": int(request.args.get("neigh", "5")),
        "show_heatmap":  request.args.get("heatmap",   "0") == "1",
        "scanlines":     request.args.get("scanlines",  "1") == "1",
    }

def _generate_mjpeg(cam_idx: int, user: str = "unknown"):
    """Generator: yields MJPEG frames for one camera.
    *user* must be passed in — do NOT call current_user() inside a generator,
    it loses the Flask request context after the first yield.
    """
    det = detectors.get(cam_idx)
    if det is None:
        # wait briefly for detector to be registered
        for _ in range(20):
            time.sleep(0.1)
            det = detectors.get(cam_idx)
            if det:
                break
    if det is None:
        return

    while cam_mgr.is_open(cam_idx):
        frame = cam_mgr.get_frame(cam_idx)
        if frame is None:
            time.sleep(0.01)
            continue

        # apply latest detection settings stored on detector
        result, stats = det.process_frame(frame)

        # update live stats
        cam_stats[cam_idx] = stats

        # capture face photos (throttled, per user)
        _maybe_capture(frame, cam_idx, stats, user, det)

        # encode to JPEG
        ok, buf = cv2.imencode(".jpg", result,
                               [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            continue
        jpg = buf.tobytes()
        last_jpeg[cam_idx] = jpg

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

    # send a blank frame on close
    blank = b""
    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + blank + b"\r\n"

@app.route("/video_feed/<int:cam_idx>")
def video_feed(cam_idx: int):
    # Capture username now (inside request context) before yielding
    uname = session.get("username", "unknown")
    # apply detection settings from query params
    det = detectors.get(cam_idx)
    if det:
        s = _detection_settings()
        for k, v in s.items():
            setattr(det, k, v)
    return Response(
        stream_with_context(_generate_mjpeg(cam_idx, uname)),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# ─── SSE live stats ───────────────────────────────────────────────────────────

@app.route("/api/stream/stats")
@login_required
def api_stream_stats():
    def _generate():
        while True:
            data = {}
            for idx in list(cam_mgr.active_indices):
                s = cam_stats.get(idx, {})
                data[str(idx)] = {
                    "fps":        round(s.get("fps",        0.0), 1),
                    "face_count": s.get("face_count", 0),
                    "total_det":  s.get("total_det",  0),
                    "engine":     s.get("engine",     "—"),
                }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1.0)
    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ─── Face records API ─────────────────────────────────────────────────────────

def _maybe_capture(frame, cam_idx: int, stats: dict, username: str, det: FaceDetector):
    if stats.get("face_count", 0) == 0:
        return
    key = (username, cam_idx)
    now = time.time()
    if now - last_capture.get(key, 0) < _FACE_INTERVAL:
        return
    last_capture[key] = now

    if det.engine == "DNN":
        raw = det._detect_dnn(frame)
        boxes = [(x, y, w, h, c) for x, y, w, h, c in raw]
    else:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        raw  = det._haar["face"].detectMultiScale(
            gray, scaleFactor=det.scale,
            minNeighbors=det.min_neighbors, minSize=(28, 28))
        boxes = [(x, y, w, h, 1.0) for (x, y, w, h) in raw] if len(raw) else []

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    recs = face_records.setdefault(username, [])
    fh2, fw2 = frame.shape[:2]

    for i, (fx, fy, fw, fh, conf) in enumerate(boxes):
        fx, fy = max(0, int(fx)), max(0, int(fy))
        x2, y2 = min(fw2, fx + int(fw)), min(fh2, fy + int(fh))
        crop   = frame[fy:y2, fx:x2]
        if crop.size == 0:
            continue
        recs.append(FaceRecord(
            timestamp    = ts,
            camera_index = int(cam_idx),
            face_number  = i,
            confidence   = float(conf),
            bbox_w       = int(fw),
            bbox_h       = int(fh),
            engine       = det.engine,
            b64_jpeg     = crop_to_b64(crop, size=120),
        ))

    if len(recs) > _MAX_RECORDS:
        face_records[username] = recs[-_MAX_RECORDS:]

@app.route("/api/reports/faces")
@login_required
def api_reports_faces():
    user = current_user()
    recs = face_records.get(user, [])
    return jsonify([{
        "timestamp":    r.timestamp,
        "camera_index": r.camera_index,
        "face_number":  r.face_number,
        "confidence":   r.confidence,
        "bbox_w":       r.bbox_w,
        "bbox_h":       r.bbox_h,
        "engine":       r.engine,
        "image":        r.b64_jpeg,
    } for r in recs])

@app.route("/api/reports/push", methods=["POST"])
@login_required
def api_reports_push():
    user    = current_user()
    recs    = face_records.get(user, [])
    cfg     = load_config()

    if not cfg.is_valid():
        return jsonify({"ok": False, "error": "GitHub não configurado."})
    if not recs:
        return jsonify({"ok": False, "error": "Nenhuma face registrada."})

    pub = GitHubPublisher(cfg)

    # ensure index.html
    if not pub._get_sha("index.html"):
        pub.ensure_index_html()

    # push users manifest
    pub.push_users(um.export_for_web())

    # push per-user report
    det    = next(iter(detectors.values()), FaceDetector())
    engine = det.engine
    ok, msg = pub.push_report(
        recs,
        session_start = datetime.datetime.now().isoformat(timespec="seconds"),
        engine        = engine,
        username      = user)

    if ok and cfg.pages_url:
        return jsonify({"ok": True, "message": msg, "url": cfg.pages_url})
    return jsonify({"ok": ok, "message": msg})

@app.route("/api/reports/clear", methods=["POST"])
@login_required
def api_reports_clear():
    face_records.pop(current_user(), None)
    return jsonify({"ok": True})

# ─── User management (master only) ────────────────────────────────────────────

@app.route("/api/users", methods=["GET"])
@login_required
@master_required
def api_users_list():
    return jsonify([{
        "username":   u.username,
        "role":       u.role,
        "created_at": u.created_at,
        "created_by": u.created_by,
    } for u in um.list_users()])

@app.route("/api/users", methods=["POST"])
@login_required
@master_required
def api_users_create():
    data   = request.get_json(force=True) or {}
    ok, msg = um.create_user(
        data.get("username", ""),
        data.get("password", ""),
        created_by=current_user())
    return jsonify({"ok": ok, "message": msg})

@app.route("/api/users/<username>", methods=["DELETE"])
@login_required
@master_required
def api_users_delete(username):
    ok, msg = um.delete_user(username)
    return jsonify({"ok": ok, "message": msg})

@app.route("/api/users/<username>/password", methods=["POST"])
@login_required
@master_required
def api_users_password(username):
    data = request.get_json(force=True) or {}
    ok, msg = um.change_password(username, data.get("password", ""))
    return jsonify({"ok": ok, "message": msg})

# ─── GitHub config ────────────────────────────────────────────────────────────

@app.route("/api/config/github", methods=["GET"])
@login_required
@master_required
def api_github_get():
    cfg = load_config()
    return jsonify({
        "repo":             cfg.repo,
        "branch":           cfg.branch,
        "pages_url":        cfg.pages_url,
        "auto_sync":        cfg.auto_sync,
        "sync_interval_s":  cfg.sync_interval_s,
        "token_set":        bool(cfg.token),
    })

@app.route("/api/config/github", methods=["POST"])
@login_required
@master_required
def api_github_save():
    data = request.get_json(force=True) or {}
    cfg  = load_config()
    if data.get("token"):
        cfg.token = data["token"]
    cfg.repo            = data.get("repo",            cfg.repo)
    cfg.branch          = data.get("branch",          cfg.branch) or "main"
    cfg.pages_url       = data.get("pages_url",       cfg.pages_url).rstrip("/")
    cfg.auto_sync       = bool(data.get("auto_sync",  cfg.auto_sync))
    cfg.sync_interval_s = int(data.get("sync_interval_s", cfg.sync_interval_s))
    save_config(cfg)
    global gh_config
    gh_config = cfg
    return jsonify({"ok": True})

@app.route("/api/config/github/test", methods=["POST"])
@login_required
@master_required
def api_github_test():
    cfg = load_config()
    pub = GitHubPublisher(cfg)
    ok, msg = pub.test_connection()
    return jsonify({"ok": ok, "message": msg})

# ─── Detection settings update ────────────────────────────────────────────────

@app.route("/api/detection/settings", methods=["POST"])
@login_required
def api_detection_settings():
    data = request.get_json(force=True) or {}
    for det in detectors.values():
        det.mode          = data.get("mode",        det.mode)
        det.confidence    = float(data.get("confidence", det.confidence))
        det.scale         = float(data.get("scale",      det.scale))
        det.min_neighbors = int(data.get("min_neighbors", det.min_neighbors))
        det.show_heatmap  = bool(data.get("show_heatmap", det.show_heatmap))
        det.scanlines     = bool(data.get("scanlines",    det.scanlines))
    return jsonify({"ok": True})

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════╗
║  FACE SENTINEL — Web Server                  ║
║  http://localhost:5000                       ║
╚══════════════════════════════════════════════╝
""")
    # open browser after short delay
    def _open():
        time.sleep(1.2)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=_open, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
