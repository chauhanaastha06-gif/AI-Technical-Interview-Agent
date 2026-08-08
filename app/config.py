import os
from pathlib import Path
from dotenv import load_dotenv

# Locate and load .env file if present
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


class Settings:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()
    MAX_TURNS: int = int(os.getenv("MAX_TURNS", "10"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Data paths
    DATA_DIR: Path = BASE_DIR / "app" / "data"
    CANDIDATES_FILE: Path = DATA_DIR / "candidates.json"
    CURRICULUM_FILE: Path = DATA_DIR / "curriculum.json"

    @property
    def is_mock_mode(self) -> bool:
        return not bool(self.ANTHROPIC_API_KEY)


settings = Settings()
