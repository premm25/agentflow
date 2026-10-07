from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROA_", env_file=str(PROJECT_ROOT / ".env"), extra="ignore"
    )

    # LLM: OpenAI-compatible gateway (e.g. LiteLLM). Secrets come from .env, never from code.
    llm_base_url: str = "http://localhost:4000/v1"  # override with ROA_LLM_BASE_URL in .env
    llm_api_key: str = ""

    # Layout. harness/ is immutable definitions, runtime/ is the only writable tree.
    harness_dir: Path = PROJECT_ROOT / "harness"
    runtime_dir: Path = PROJECT_ROOT / "runtime"
    data_dir: Path = PROJECT_ROOT / "data"

    # Orchestrator service (single process hosts every agent)
    host: str = "0.0.0.0"
    port: int = 8100

    # Observability: the orchestrator only exports OTLP; the dashboard is a separate process.
    otlp_endpoint: str = "http://localhost:8200/v1/traces"
    dashboard_url: str = "http://localhost:8200"
    otel_enabled: bool = True

    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "roa.db"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.runtime_dir.mkdir(parents=True, exist_ok=True)
