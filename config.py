import os
from dotenv import load_dotenv


class Config:
    """
    App-wide configuration loaded from environment variables.
    """

    def __init__(self):
        self.pgvector_conn_str: str = self._require("PGVECTOR_CONN_STR")
        self.env: str = os.getenv("ENV", "dev")

    def _require(self, key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise RuntimeError(f"Missing required environment variable: {key}")
        return value


load_dotenv()
config = Config()
