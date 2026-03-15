"""
user_manager.py — Local user management for Face Sentinel.

Users are stored in ~/.face_sentinel/users.json
Passwords are stored as SHA-256 hashes (never plaintext).

Roles
-----
  master  — can create / delete users, access everything
  user    — can only run detection and see own reports
"""

from __future__ import annotations

import hashlib
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field


# ─── Storage ──────────────────────────────────────────────────────────────────
_CONFIG_DIR   = Path.home() / ".face_sentinel"
_USERS_FILE   = _CONFIG_DIR / "users.json"

# Hardcoded master bootstrap credentials
MASTER_USERNAME = "master"
MASTER_PASSWORD = "@m123"


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class User:
    username:    str
    role:        str          # "master" | "user"
    created_at:  str
    created_by:  str
    password_hash: str = field(repr=False)

    @property
    def is_master(self) -> bool:
        return self.role == "master"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "User":
        return User(**d)


def _sha256(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ─── Manager ──────────────────────────────────────────────────────────────────

class UserManager:
    """Thread-safe local user store."""

    def __init__(self):
        self._users: dict[str, User] = {}
        self._load()
        self._bootstrap_master()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        try:
            data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
            self._users = {k: User.from_dict(v) for k, v in data.items()}
        except Exception:
            self._users = {}

    def _save(self):
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._users.items()}
        _USERS_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _bootstrap_master(self):
        """Ensure master account always exists with the default password."""
        if MASTER_USERNAME not in self._users:
            self._users[MASTER_USERNAME] = User(
                username      = MASTER_USERNAME,
                role          = "master",
                created_at    = datetime.datetime.now().isoformat(timespec="seconds"),
                created_by    = "system",
                password_hash = _sha256(MASTER_PASSWORD),
            )
            self._save()

    # ── Public API ────────────────────────────────────────────────────────────

    def authenticate(self, username: str, password: str) -> User | None:
        """Return User if credentials are valid, else None."""
        u = self._users.get(username.lower().strip())
        if u and u.password_hash == _sha256(password):
            return u
        return None

    def list_users(self) -> list[User]:
        return sorted(self._users.values(), key=lambda u: u.created_at)

    def get_user(self, username: str) -> User | None:
        return self._users.get(username.lower().strip())

    def create_user(self, username: str, password: str,
                    created_by: str = MASTER_USERNAME) -> tuple[bool, str]:
        """Create a new 'user' role account. Returns (success, message)."""
        key = username.lower().strip()
        if not key:
            return False, "Nome de usuário não pode ser vazio."
        if len(key) < 3:
            return False, "Nome de usuário deve ter ao menos 3 caracteres."
        if key in self._users:
            return False, f"Usuário '{key}' já existe."
        if len(password) < 4:
            return False, "Senha deve ter ao menos 4 caracteres."

        self._users[key] = User(
            username      = key,
            role          = "user",
            created_at    = datetime.datetime.now().isoformat(timespec="seconds"),
            created_by    = created_by,
            password_hash = _sha256(password),
        )
        self._save()
        return True, f"Usuário '{key}' criado com sucesso."

    def delete_user(self, username: str) -> tuple[bool, str]:
        """Delete a user (master cannot be deleted)."""
        key = username.lower().strip()
        if key == MASTER_USERNAME:
            return False, "O usuário master não pode ser deletado."
        if key not in self._users:
            return False, f"Usuário '{key}' não encontrado."
        del self._users[key]
        self._save()
        return True, f"Usuário '{key}' deletado."

    def change_password(self, username: str, new_password: str) -> tuple[bool, str]:
        key = username.lower().strip()
        if key not in self._users:
            return False, "Usuário não encontrado."
        if len(new_password) < 4:
            return False, "Senha deve ter ao menos 4 caracteres."
        self._users[key].password_hash = _sha256(new_password)
        self._save()
        return True, "Senha alterada com sucesso."

    def export_for_web(self) -> dict:
        """
        Return a dict safe to publish on GitHub Pages.
        Contains usernames + hashes only (no dates, no creator).
        The web dashboard uses this for client-side auth.
        """
        return {
            u.username: {
                "hash": u.password_hash,
                "role": u.role,
            }
            for u in self._users.values()
        }
