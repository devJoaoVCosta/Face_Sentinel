"""
github_publisher.py — Commits face-detection data to a GitHub repo via REST API.

Per-user data layout on GitHub:
  index.html                ← dashboard SPA (uploaded once)
  data/users.json           ← {username: {hash, role}}  (for web login)
  data/{username}/report.json  ← face records for each user

No pip dependencies — uses only urllib (stdlib).
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
import urllib.error
import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path

from report_html import FaceRecord

# ─── Config file location ─────────────────────────────────────────────────────
_CONFIG_DIR  = Path.home() / ".face_sentinel"
_CONFIG_FILE = _CONFIG_DIR / "config.json"
_API_BASE    = "https://api.github.com"

MAX_RECORDS_PUSH = 200


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class GitHubConfig:
    token:           str  = ""
    repo:            str  = ""
    branch:          str  = "main"
    pages_url:       str  = ""
    auto_sync:       bool = False
    sync_interval_s: int  = 30

    @property
    def owner(self) -> str:
        return self.repo.split("/")[0] if "/" in self.repo else ""

    @property
    def reponame(self) -> str:
        return self.repo.split("/")[1] if "/" in self.repo else ""

    def is_valid(self) -> bool:
        return bool(self.token and "/" in self.repo and self.owner and self.reponame)


def load_config() -> GitHubConfig:
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        return GitHubConfig(**{k: v for k, v in data.items()
                                if k in GitHubConfig.__dataclass_fields__})
    except Exception:
        return GitHubConfig()


def save_config(cfg: GitHubConfig):
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


# ─── Publisher ────────────────────────────────────────────────────────────────

class GitHubPublisher:
    """
    Pushes per-user face-detection reports + shared user manifest to GitHub.

    Data layout
    -----------
    index.html                    ← SPA (uploaded once)
    data/users.json               ← {username: {hash, role}}
    data/{username}/report.json   ← per-user face data
    """

    def __init__(self, config: GitHubConfig):
        self.cfg = config

    # ── API helpers ───────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"token {self.cfg.token}",
            "Accept":        "application/vnd.github+json",
            "Content-Type":  "application/json",
            "User-Agent":    "FaceSentinel/2.0",
        }

    def _get(self, path: str) -> tuple[int, dict]:
        req = urllib.request.Request(
            f"{_API_BASE}{path}", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, {}
        except Exception as e:
            return 0, {"error": str(e)}

    def _put(self, path: str, body: dict) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8")
        req  = urllib.request.Request(
            f"{_API_BASE}{path}", data=data,
            headers=self._headers(), method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            rb = {}
            try:
                rb = json.loads(e.read())
            except Exception:
                pass
            return e.code, rb
        except Exception as e:
            return 0, {"error": str(e)}

    def _get_sha(self, file_path: str) -> str | None:
        code, data = self._get(
            f"/repos/{self.cfg.repo}/contents/{file_path}?ref={self.cfg.branch}")
        return data.get("sha") if code == 200 else None

    def _put_file(self, file_path: str, content_str: str,
                  commit_msg: str) -> tuple[bool, str]:
        """Encode *content_str* as base64 and PUT to GitHub."""
        b64  = base64.b64encode(content_str.encode("utf-8")).decode("ascii")
        sha  = self._get_sha(file_path)
        body = {"message": commit_msg, "content": b64, "branch": self.cfg.branch}
        if sha:
            body["sha"] = sha
        code, resp = self._put(
            f"/repos/{self.cfg.repo}/contents/{file_path}", body)
        if code in (200, 201):
            return True, "ok"
        return False, resp.get("message", f"HTTP {code}")

    # ── Public API ────────────────────────────────────────────────────────────

    def test_connection(self) -> tuple[bool, str]:
        if not self.cfg.is_valid():
            return False, "Token ou repositório inválido."
        code, data = self._get(f"/repos/{self.cfg.repo}")
        if code == 200:
            priv = "privado" if data.get("private") else "público"
            return True, f"Conectado: {data.get('full_name', self.cfg.repo)} ({priv}) ✓"
        if code == 404:
            return False, f"Repositório '{self.cfg.repo}' não encontrado."
        if code == 401:
            return False, "Token inválido ou sem permissão."
        return False, f"Erro HTTP {code}."

    def ensure_index_html(self) -> tuple[bool, str]:
        """Upload index.html once (or update if changed)."""
        html_path = Path(__file__).parent / "web" / "index.html"
        if not html_path.exists():
            return False, "web/index.html não encontrado."
        ok, msg = self._put_file(
            "index.html",
            html_path.read_text(encoding="utf-8"),
            "chore: deploy Face Sentinel dashboard")
        return ok, ("index.html enviado ✓" if ok else f"Erro: {msg}")

    def push_users(self, users_export: dict) -> tuple[bool, str]:
        """
        Push data/users.json (username → {hash, role}).
        Called whenever users change or on first sync.
        """
        content = json.dumps(users_export, indent=2, ensure_ascii=False)
        ok, msg = self._put_file(
            "data/users.json", content,
            f"data: update users manifest [{datetime.datetime.now().strftime('%H:%M:%S')}]")
        return ok, ("users.json enviado ✓" if ok else f"Erro users.json: {msg}")

    def push_report(self,
                    records: list[FaceRecord],
                    session_start: str,
                    engine: str,
                    username: str = "unknown") -> tuple[bool, str]:
        """
        Push data/{username}/report.json with face records.
        Returns (success, message).
        """
        if not self.cfg.is_valid():
            return False, "GitHub não configurado."

        recent = records[-MAX_RECORDS_PUSH:] if len(records) > MAX_RECORDS_PUSH \
                 else records

        payload = {
            "meta": {
                "username":      username,
                "session_start": session_start,
                "last_updated":  datetime.datetime.now().isoformat(timespec="seconds"),
                "engine":        engine,
                "total_records": len(records),
                "shown_records": len(recent),
            },
            "faces": [
                {
                    "timestamp":    r.timestamp,
                    "camera_index": r.camera_index,
                    "face_number":  r.face_number,
                    "confidence":   round(r.confidence, 4),
                    "bbox_w":       r.bbox_w,
                    "bbox_h":       r.bbox_h,
                    "engine":       r.engine,
                    "image":        r.b64_jpeg,
                }
                for r in recent
            ],
        }

        file_path = f"data/{username}/report.json"
        ok, msg   = self._put_file(
            file_path,
            json.dumps(payload, ensure_ascii=False),
            f"data: sync {len(recent)} faces for {username} "
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}]")

        if ok:
            return True, f"{len(recent)} face(s) sincronizadas ✓"
        return False, f"Erro ao fazer push: {msg}"
