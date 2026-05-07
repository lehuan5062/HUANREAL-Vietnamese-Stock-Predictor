"""One-time vnstock client introduction.

Stamps every outbound vnstock HTTP call with a friendly identifier so the
broker sees a named client instead of an anonymous scraper. Idempotent —
safe to call multiple times per process.
"""
from __future__ import annotations

import logging
import threading

_INTRODUCED = False
_LOCK = threading.Lock()
_LOGGED_FIRST = False

CLIENT_ID = "HUANREAL"
CLIENT_REPO = "https://github.com/lehuan5062/HUANREAL-Vietnamese-Stock-Predictor"
USER_AGENT = "HUANREAL-Vietnamese-Stock-Predictor"

_log = logging.getLogger("stockpredict.intro")


def _intro_headers() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "X-Client-ID": CLIENT_ID,
        "X-Client-Repo": CLIENT_REPO,
    }


def introduce(verify: bool = True) -> None:
    """Identify this client to vnstock as HUANREAL + repo URL."""
    global _INTRODUCED
    with _LOCK:
        if _INTRODUCED:
            return
        _INTRODUCED = True

    # 1. Optional API-key registration. vnstock.register_user() with no
    #    argument drops into an interactive prompt — that's the wrong UX
    #    for a batch run. Only register if VNSTOCK_API_KEY is set in env;
    #    otherwise we identify purely via headers below (still recognized
    #    as HUANREAL, just on the Guest tier).
    import os
    api_key = os.environ.get("VNSTOCK_API_KEY", "").strip()
    if api_key:
        try:
            from vnstock import register_user
            register_user(api_key=api_key)
        except Exception as e:
            _log.debug("register_user skipped: %s", e)

    # 2. Monkey-patch get_headers so every request carries our identifiers.
    #    The original function is responsible for the realistic browser
    #    headers + per-source Referer/Origin; we just stamp our three keys
    #    on top via the override_headers path it already supports.
    try:
        from vnstock.core.utils import user_agent as _ua

        original_get_headers = _ua.get_headers
        intro_overrides = _intro_headers()

        def patched_get_headers(*args, **kwargs):
            user_override = kwargs.get("override_headers") or {}
            merged = {**intro_overrides, **user_override}
            kwargs["override_headers"] = merged
            headers = original_get_headers(*args, **kwargs)
            if verify:
                _maybe_log_first(headers, kwargs.get("data_source", "?"))
            return headers

        _ua.get_headers = patched_get_headers

        # Some explorer modules import get_headers by name at import time.
        # Patch those references too so already-bound callsites pick up
        # the new behavior.
        for modname in (
            "vnstock.explorer.vci.quote",
            "vnstock.explorer.vci.listing",
            "vnstock.explorer.tcbs.quote",
            "vnstock.explorer.msn.quote",
        ):
            try:
                import importlib
                m = importlib.import_module(modname)
                if hasattr(m, "get_headers"):
                    m.get_headers = patched_get_headers
            except Exception:
                pass
    except Exception as e:
        _log.warning("header patch failed (intro will not land): %s", e)


def _maybe_log_first(headers: dict, data_source: str) -> None:
    global _LOGGED_FIRST
    with _LOCK:
        if _LOGGED_FIRST:
            return
        _LOGGED_FIRST = True
    # Ensure the message reaches the console even if the root logger is
    # set to WARNING — `stockpredict.intro` is independent of vnstock's
    # own loggers (which fetcher.quiet_vnstock_logger silences).
    if not _log.handlers and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    _log.setLevel(logging.INFO)
    _log.info(
        "[stockpredict.intro] %s handshake -> %s  ua=%s  client=%s  repo=%s",
        CLIENT_ID,
        data_source,
        headers.get("User-Agent", "?"),
        headers.get("X-Client-ID", "?"),
        headers.get("X-Client-Repo", "?"),
    )
