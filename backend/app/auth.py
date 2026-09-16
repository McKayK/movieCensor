"""Sign in with Plex. Only accounts that can access this Plex server get in; the server owner is admin."""
from __future__ import annotations

import logging
import sqlite3

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from . import plex
from .config import settings
from .db import execute, get_conn, now, row

log = logging.getLogger("auth")
router = APIRouter(prefix="/api/auth", tags=["auth"])


def _public_base(request: Request) -> str:
    if settings.public_url:
        return settings.public_url
    return str(request.base_url).rstrip("/")


@router.get("/login")
def login(request: Request):
    if settings.auth_disabled:
        return {"url": "/", "disabled": True}
    try:
        pin = plex.create_pin()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Could not reach plex.tv: {e}")
    request.session["pin_id"] = pin["id"]
    forward = f"{_public_base(request)}/auth/callback"
    return {"url": plex.auth_url(pin["code"], forward)}


@router.get("/callback")
def callback(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    pin_id = request.session.get("pin_id")
    if not pin_id:
        raise HTTPException(400, "No sign-in in progress. Start again.")
    try:
        token = plex.check_pin(pin_id)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"plex.tv error: {e}")
    if not token:
        raise HTTPException(401, "Plex has not approved this sign-in yet.")

    try:
        user = plex.plex_user(token)
        machine_id = plex.server().machine_identifier()
        resource = plex.user_server_access(token, machine_id)
    except (httpx.HTTPError, plex.PlexError) as e:
        raise HTTPException(502, f"Could not verify Plex access: {e}")
    if not resource:
        request.session.pop("pin_id", None)
        raise HTTPException(403, "This Plex account doesn't have access to this server.")

    username = user.get("username") or user.get("title") or "plex user"
    is_admin = bool(resource.get("owned")) or username.lower() in settings.admin_usernames
    existing = row(conn, "SELECT id FROM users WHERE plex_uuid = ?", (user["uuid"],))
    if existing:
        conn.execute(
            "UPDATE users SET username=?, thumb=?, is_admin=?, last_login=? WHERE id=?",
            (username, user.get("thumb"), int(is_admin), now(), existing["id"]),
        )
        uid = existing["id"]
    else:
        uid = execute(
            conn,
            "INSERT INTO users (plex_uuid, username, thumb, is_admin, created_at, last_login) VALUES (?,?,?,?,?,?)",
            (user["uuid"], username, user.get("thumb"), int(is_admin), now(), now()),
        )
    request.session.pop("pin_id", None)
    request.session["uid"] = uid
    log.info("signed in %s (admin=%s)", username, is_admin)
    return {"ok": True}


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


def _dev_user(conn: sqlite3.Connection) -> dict:
    u = row(conn, "SELECT * FROM users WHERE plex_uuid = 'dev'")
    if not u:
        execute(
            conn,
            "INSERT INTO users (plex_uuid, username, is_admin, created_at) VALUES ('dev','dev',1,?)",
            (now(),),
        )
        u = row(conn, "SELECT * FROM users WHERE plex_uuid = 'dev'")
    return u


def current_user(request: Request, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    if settings.auth_disabled:
        return _dev_user(conn)
    uid = request.session.get("uid")
    if not uid:
        raise HTTPException(401, "Sign in required")
    u = row(conn, "SELECT * FROM users WHERE id = ?", (uid,))
    if not u:
        request.session.clear()
        raise HTTPException(401, "Sign in required")
    return u


def admin_user(user: dict = Depends(current_user)) -> dict:
    if not user["is_admin"]:
        raise HTTPException(403, "Admin only")
    return user


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return {"id": user["id"], "username": user["username"], "thumb": user["thumb"], "isAdmin": bool(user["is_admin"])}
