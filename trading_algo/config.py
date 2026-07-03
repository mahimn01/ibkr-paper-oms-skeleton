"""Runtime configuration for the IBKR trading stack.

Hardened for enterprise use:
- `_get_env_bool` is STRICT — unknown values raise `EnvParseError`. A typo
  on a safety-critical flag like `TRADING_ALLOW_LIVE=tru` can never silently
  flip it off.
- `atomic_write_text` writes via tempfile → `os.replace` with owner-only
  permissions, so any cached artifact (flex CSV dumps, contract cache, etc.)
  is never observed in a partial / world-readable state.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSY = frozenset({"0", "false", "no", "off", "n", "f", ""})


class EnvParseError(ValueError):
    """Raised when an env var is set but cannot be parsed.

    Never silently defaulted — a typo on a safety-critical env var must fail
    loud at startup, not quietly flip behaviour.
    """


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise EnvParseError(
            f"Env var {name}={raw!r} is not an int: {exc}"
        ) from exc


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise EnvParseError(
            f"Env var {name}={raw!r} is not a float: {exc}"
        ) from exc


def _get_env_bool(name: str, default: bool) -> bool:
    """Strict env-bool parser.

    Unset → `default`. Set to anything in `_TRUTHY` or `_FALSY` → explicit.
    Anything else → `EnvParseError`. This matters for safety flags like
    `TRADING_ALLOW_LIVE` where a typo must never be silently interpreted.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if norm in _TRUTHY:
        return True
    if norm in _FALSY:
        return False
    raise EnvParseError(
        f"Env var {name}={raw!r} is not a recognised boolean. "
        f"Use one of: {sorted(_TRUTHY | _FALSY - {''})}. "
        f"Unset the var to get the default ({default})."
    )


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def atomic_write_text(path: Path, data: str, *, mode: int = 0o600) -> None:
    """Atomically write `data` to `path` with the given file mode.

    Writes to a temp file in the same directory, fsyncs, then renames over
    the target. Guarantees:
    - No partial-write state visible on filesystem (rename is atomic on POSIX).
    - Temp file is created with `mode` permissions from the start (no TOCTOU
      window where another process could read a world-readable version).
    - Parent directory is created with 0o700 if missing.

    On Windows rename is best-effort (replace existing); mode is a no-op.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass

    fd: int | None = None
    tmp_name: str | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        if os.name == "posix":
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None  # fdopen took ownership
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync not supported on some filesystems (network mounts).
                pass
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_name is not None and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

IBKR_PORT_TWS_LIVE = 7496
IBKR_PORT_TWS_PAPER = 7497
IBKR_PORT_GW_LIVE = 4001
IBKR_PORT_GW_PAPER = 4002


@dataclass(frozen=True)
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7


@dataclass(frozen=True)
class TradingConfig:
    broker: str = "ibkr"  # "ibkr" | "sim"
    live_enabled: bool = False
    require_paper: bool = True  # forced-on safety rail
    allow_live: bool = False  # explicit override to connect to live accounts (read + trade)
    dry_run: bool = False
    order_token: str | None = None
    confirm_token_required: bool = False
    db_path: str | None = None
    poll_seconds: int = 5
    # Pre-transmit constitution gate. OPT-IN: default off everywhere, enabled
    # only by TRADING_CONSTITUTION_REQUIRED=true. It is NOT auto-coupled to
    # live_enabled because the verdict-writer (the constitution-check command /
    # CLI gate) must be wired and paper-verified first — auto-enabling it with no
    # writer would refuse every live transmit ("no fresh clearance").
    constitution_required: bool = False
    # A cleared verdict older than this is stale -> re-check. The verdict is
    # stamped at enrichment START (data age, honest) and live enrichment takes
    # ~20-25s over the gateway, so 120s leaves ~90s to review + transmit.
    constitution_max_age_s: int = 120
    ibkr: IBKRConfig = IBKRConfig()

    @staticmethod
    def from_env() -> "TradingConfig":
        ibkr = IBKRConfig(
            host=_get_env("IBKR_HOST", "127.0.0.1"),
            port=_get_env_int("IBKR_PORT", 7497),
            client_id=_get_env_int("IBKR_CLIENT_ID", 7),
        )
        allow_live = _get_env_bool("TRADING_ALLOW_LIVE", False)
        live_enabled = _get_env_bool("TRADING_LIVE_ENABLED", False)
        return TradingConfig(
            broker=_get_env("TRADING_BROKER", "ibkr"),
            live_enabled=live_enabled,
            require_paper=not allow_live,
            allow_live=allow_live,
            dry_run=_get_env_bool("TRADING_DRY_RUN", False),
            order_token=(_get_env("TRADING_ORDER_TOKEN", "").strip() or None),
            confirm_token_required=_get_env_bool("TRADING_CONFIRM_TOKEN_REQUIRED", False),
            db_path=(_get_env("TRADING_DB_PATH", "").strip() or None),
            poll_seconds=_get_env_int("TRADING_POLL_SECONDS", 5),
            constitution_required=_get_env_bool("TRADING_CONSTITUTION_REQUIRED", False),
            constitution_max_age_s=_get_env_int("TRADING_CONSTITUTION_MAX_AGE_S", 120),
            ibkr=ibkr,
        )
