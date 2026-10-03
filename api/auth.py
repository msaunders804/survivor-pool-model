"""One shared password for the three of you, not per-user accounts.

This league has one real player (Brent) and two collaborators helping
build the tool -- there's no one here who should see different data
from anyone else, so per-user login is complexity this doesn't need
yet. A single APP_PASSWORD env var gates the whole app; a signed,
time-limited token (itsdangerous) is the session, stateless so it
survives the API restarting between requests. Swap this for real
per-user accounts if the league ever actually needs to separate who
sees what -- the User/LeagueMembership tables already in models.py
are exactly what that would build on.
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

APP_PASSWORD = os.environ.get("APP_PASSWORD", "change-me-locally")
SECRET_KEY = os.environ.get("SECRET_KEY", "local-dev-only-not-secret")
TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24 * 14  # 14 days -- log in every couple weeks, not every visit

_signer = URLSafeTimedSerializer(SECRET_KEY)


def check_password(password: str) -> bool:
    return password == APP_PASSWORD


def issue_token() -> str:
    return _signer.dumps({"ok": True})


def verify_token(token: str) -> bool:
    try:
        _signer.loads(token, max_age=TOKEN_MAX_AGE_SECONDS)
        return True
    except (BadSignature, SignatureExpired):
        return False


def require_auth(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency -- add to any route that needs to be behind the password."""
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ")
    if not verify_token(token):
        raise HTTPException(401, "invalid or expired token -- log in again")
