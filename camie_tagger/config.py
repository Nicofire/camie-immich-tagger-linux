"""Settings loading.

Precedence, highest first: CLI argument, process environment, .env file, built-in default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or invalid."""


@dataclass
class Settings:
    scan_dirs: list[Path] = field(default_factory=list)
    model_dir: Path = PROJECT_ROOT / "models"
    exiftool: str = "exiftool"
    data_dir: Path = PROJECT_ROOT / "data"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_level: str = "INFO"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backups: int = 5
    log_file: Path | None = None
    immich_url: str = ""
    immich_api_key: str = ""
    library_ids: list[str] = field(default_factory=list)
    saucenao_api_key: str = ""
    threshold: float = 0.5
    device: str = "auto"
    fail_on_cpu_fallback: bool = False
    tier0_min_similarity: float = 88.0
    tier0_daily_cap: int = 100
    tier0_interval: float = 18.0
    env_file: Path | None = None

    @property
    def resolved_log_file(self) -> Path:
        return self.log_file or (self.log_dir / "camie-tagger.log")

    def secrets(self) -> list[str]:
        return [s for s in (self.immich_api_key, self.saucenao_api_key) if s]

    def require_scan_dirs(self) -> list[Path]:
        existing = [d for d in self.scan_dirs if d.is_dir()]
        if not self.scan_dirs:
            raise ConfigError(
                "No scan directories configured. Set CAMIE_SCAN_DIRS in your .env file."
            )
        if not existing:
            listed = ", ".join(str(d) for d in self.scan_dirs)
            raise ConfigError(f"None of the configured scan directories exist: {listed}")
        return existing

    def require_immich(self) -> None:
        if not self.immich_url:
            raise ConfigError("IMMICH_URL is not set.")
        if not self.immich_api_key:
            raise ConfigError(
                "IMMICH_API_KEY is not set. Put it in .env or pass --immich-api-key."
            )

    def require_saucenao(self) -> None:
        if not self.saucenao_api_key:
            raise ConfigError(
                "SAUCENAO_API_KEY is not set. Put it in .env or pass --saucenao-api-key."
            )


def _get(key: str) -> str | None:
    value = os.environ.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _as_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _as_path_list(value: str | None) -> list[Path]:
    if not value:
        return []
    # Colon-separated, matching PATH conventions on Linux.
    return [Path(p).expanduser() for p in value.split(":") if p.strip()]


def _as_str_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_float(value: str | None, key: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {value!r}") from exc


def _as_int(value: str | None, key: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {value!r}") from exc


def load_settings(config_path: str | os.PathLike[str] | None = None, **overrides) -> Settings:
    """Build Settings from the .env file, the environment and explicit overrides."""
    overrides = {k: v for k, v in overrides.items() if v is not None}

    env_file = Path(config_path).expanduser() if config_path else DEFAULT_ENV_FILE
    if config_path and not env_file.is_file():
        raise ConfigError(f"Config file not found: {env_file}")
    if env_file.is_file():
        # override=False keeps real environment variables ahead of the .env file.
        load_dotenv(env_file, override=False)
    else:
        env_file = None

    settings = Settings(env_file=env_file)

    settings.scan_dirs = _as_path_list(_get("CAMIE_SCAN_DIRS"))
    settings.model_dir = _as_path(_get("CAMIE_MODEL_DIR")) or settings.model_dir
    settings.exiftool = _get("CAMIE_EXIFTOOL") or settings.exiftool
    settings.data_dir = _as_path(_get("CAMIE_DATA_DIR")) or settings.data_dir
    settings.log_dir = _as_path(_get("CAMIE_LOG_DIR")) or settings.log_dir
    settings.log_level = _get("CAMIE_LOG_LEVEL") or settings.log_level

    log_max_mb = _as_int(_get("CAMIE_LOG_MAX_MB"), "CAMIE_LOG_MAX_MB")
    if log_max_mb:
        settings.log_max_bytes = log_max_mb * 1024 * 1024
    log_backups = _as_int(_get("CAMIE_LOG_BACKUPS"), "CAMIE_LOG_BACKUPS")
    if log_backups is not None:
        settings.log_backups = log_backups

    settings.immich_url = (_get("IMMICH_URL") or settings.immich_url).rstrip("/")
    settings.immich_api_key = _get("IMMICH_API_KEY") or settings.immich_api_key
    settings.library_ids = _as_str_list(_get("IMMICH_LIBRARY_IDS"))
    settings.saucenao_api_key = _get("SAUCENAO_API_KEY") or settings.saucenao_api_key

    threshold = _as_float(_get("CAMIE_THRESHOLD"), "CAMIE_THRESHOLD")
    if threshold is not None:
        settings.threshold = threshold
    settings.device = _get("CAMIE_DEVICE") or settings.device

    min_similarity = _as_float(
        _get("CAMIE_TIER0_MIN_SIMILARITY"), "CAMIE_TIER0_MIN_SIMILARITY"
    )
    if min_similarity is not None:
        settings.tier0_min_similarity = min_similarity
    daily_cap = _as_int(_get("CAMIE_TIER0_DAILY_CAP"), "CAMIE_TIER0_DAILY_CAP")
    if daily_cap is not None:
        settings.tier0_daily_cap = daily_cap
    interval = _as_float(_get("CAMIE_TIER0_INTERVAL"), "CAMIE_TIER0_INTERVAL")
    if interval is not None:
        settings.tier0_interval = interval

    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise ConfigError(f"Unknown setting override: {key}")
        if key == "immich_url" and isinstance(value, str):
            value = value.rstrip("/")
        if key in {"model_dir", "data_dir", "log_dir", "log_file"} and value is not None:
            value = Path(value).expanduser()
        setattr(settings, key, value)

    if not 0.0 < settings.threshold <= 1.0:
        raise ConfigError(f"Threshold must be between 0 and 1, got {settings.threshold}")

    return settings
