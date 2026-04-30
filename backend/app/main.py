"""FastAPI application for Remedy PDF Desktop."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, QueryParams
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.app.config import settings
from backend.app.routes_api import router as api_router

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Rate limiter (in-memory, per-IP) ────────────────────
class _RateBucket:
    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: int) -> None:
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()


_rate_buckets: dict[str, _RateBucket] = defaultdict(lambda: _RateBucket(settings.upload_rate_limit))


def _check_rate_limit(client_ip: str) -> bool:
    """Token-bucket rate limiter. Returns True if the request is allowed."""
    bucket = _rate_buckets[client_ip]
    now = time.monotonic()
    elapsed = now - bucket.last_refill
    bucket.tokens = min(
        settings.upload_rate_limit,
        bucket.tokens + elapsed * (settings.upload_rate_limit / 60.0),
    )
    bucket.last_refill = now
    if bucket.tokens >= 1.0:
        bucket.tokens -= 1.0
        return True
    return False


# ── API-key auth middleware ──────────────────────────────
def _api_key_matches(provided: str | None) -> bool:
    if not settings.app_api_key:
        return True
    return secrets.compare_digest(provided or "", settings.app_api_key)


class _APIKeyMiddleware:
    """Reject API and WebSocket requests missing the configured API key."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not settings.app_api_key:
            await self.app(scope, receive, send)
            return

        scope_type = scope["type"]
        path = str(scope.get("path", ""))

        if scope_type == "http":
            if path == "/api/health" or not path.startswith("/api/"):
                await self.app(scope, receive, send)
                return

            headers = Headers(scope=scope)
            query = QueryParams(scope.get("query_string", b"").decode("latin-1"))
            provided = headers.get("x-api-key") or query.get("api_key")
            if not _api_key_matches(provided):
                response = JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing API key"},
                )
                await response(scope, receive, send)
                return

        elif scope_type == "websocket":
            if not path.startswith(("/api/ws/", "/ws/")):
                await self.app(scope, receive, send)
                return

            headers = Headers(scope=scope)
            query = QueryParams(scope.get("query_string", b"").decode("latin-1"))
            provided = headers.get("x-api-key") or query.get("api_key")
            if not _api_key_matches(provided):
                await send({
                    "type": "websocket.close",
                    "code": 1008,
                    "reason": "Invalid or missing API key",
                })
                return

        await self.app(scope, receive, send)


# ── Job cleanup background task ──────────────────────────
async def _cleanup_old_jobs() -> None:
    """Periodically remove expired jobs and their files."""
    from backend.app.remediation import purge_expired_jobs
    while True:
        await asyncio.sleep(300)  # run every 5 minutes
        try:
            removed = purge_expired_jobs(settings.job_ttl_seconds)
            if removed:
                logger.info("Purged %d expired job(s)", removed)
        except Exception:
            logger.exception("Job cleanup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_cleanup_old_jobs())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Remedy PDF Desktop", version="0.3.0", lifespan=lifespan)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


@app.middleware("http")
async def add_security_headers(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response

# Middleware (order matters — outermost runs first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(_APIKeyMiddleware)

app.include_router(api_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


_frontend_dist = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
