"""
config.py — Centralized configuration for the Igris agent.

Loads settings from environment variables (.env file) and provides
validated defaults using Pydantic Settings (Issue #3 — Pydantic AI support).
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env file if it exists alongside this script
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)


class AgentSettings(BaseSettings):
    """Validated agent configuration backed by environment variables."""

    groq_api_key: str = Field(
        default="",
        description="Groq API key for LLM access",
    )

    # Model tuning (Issue #2 — Increase capacity)
    model_name: str = Field(
        default="llama-3.1-70b-versatile",
        description="Groq model identifier. Upgraded from llama3-8b-8192 for higher capacity.",
    )
    model_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    model_max_tokens: int = Field(
        default=2048,
        description="Max tokens per response — doubled from the original 500.",
    )

    # Memory paths
    memory_file: str = Field(default="igris_chat_memory.pkl")
    memory_backup_file: str = Field(default="igris_chat_memory.pkl.bak")
    memory_index_dir: str = Field(default="igris_memory.index")
    docs_file: str = Field(default="igris_docs.pkl")

    # Document upload directory
    documents_dir: str = Field(default="documents")

    # Embedding model
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    class Config:
        env_prefix = ""
        case_sensitive = False


# Singleton settings instance — import this everywhere
settings = AgentSettings()
