"""
config.py
---------
Application-level configuration for the AntiDeepfake backend.

All settings are read from environment variables with sensible defaults.
Use a ``.env`` file in the project root or export variables in your shell
before starting the server.

Example:
    export HOST=0.0.0.0
    export PORT=8000
    export LOG_LEVEL=info
    export DEFAULT_EPSILON=0.02

    uvicorn src.backend.main:app --reload
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backend configuration resolved from environment variables.

    Attributes:
        host: Network interface the uvicorn server binds to.
        port: TCP port the server listens on.
        log_level: Logging level string (debug / info / warning / error).
        default_epsilon: Default PGD perturbation budget used when the
            caller does not supply an ``epsilon`` value.
        app_title: Human-readable title shown in the Swagger UI.
        app_version: SemVer version string shown in the Swagger UI.
        app_description: Short description shown in the Swagger UI.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server settings ──────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    # ── ML settings ──────────────────────────────────────────────────────────
    default_epsilon: float = 0.02

    # ── API metadata ─────────────────────────────────────────────────────────
    app_title: str = "AntiDeepfake API"
    app_version: str = "1.0.0"
    app_description: str = (
        "Adversarial Image Cloaking API — applies PGD perturbations to "
        "protect facial images from automated face-recognition systems."
    )


# Module-level singleton — import this wherever settings are needed.
settings = Settings()
