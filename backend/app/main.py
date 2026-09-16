"""FastAPI entry point: API + the built React app."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from . import auth, db, plex
from .api import jobs as jobs_api
from .api import movies as movies_api
from .config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Movie Censor", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.get_session_secret(),
    session_cookie="movie_censor",
    max_age=60 * 60 * 24 * 30,
    same_site="lax",
    https_only=settings.cookie_secure,
)
app.include_router(auth.router)
app.include_router(movies_api.router)
app.include_router(jobs_api.router)


@app.get("/api/health")
def health():
    status = {"ok": True, "authDisabled": settings.auth_disabled}
    try:
        status["plexServer"] = plex.server().machine_identifier()
    except Exception as e:
        status["plexError"] = str(e)
    status["censoredDirWritable"] = os.access(settings.censored_dir, os.W_OK)
    return status


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # keep JSON errors for the SPA
    logging.getLogger("api").exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": str(exc)}, status_code=500)


_static = settings.static_dir
if os.path.isdir(os.path.join(_static, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static, "assets")), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(404, "Not found")
    candidate = os.path.join(_static, full_path)
    if full_path and os.path.isfile(candidate) and os.path.abspath(candidate).startswith(os.path.abspath(_static)):
        return FileResponse(candidate)
    index = os.path.join(_static, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return JSONResponse({"detail": "Frontend not built. Run `npm run build` in frontend/ or use the Docker image."}, 404)
