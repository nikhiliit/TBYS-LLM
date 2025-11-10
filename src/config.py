"""Configuration settings for Qwen Agent."""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration."""

    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '8080'))

    # Database settings
    DB_PATH = Path(os.getenv('DB_PATH', 'qwen3_chat.db'))

    # Model settings
    TEXT_MODEL_PATH = os.getenv('TEXT_MODEL_PATH', './models/models--Qwen--Qwen3-0.6B/snapshots')
    VL_MODEL_PATH = os.getenv('VL_MODEL_PATH', './models_vl')

    # Model parameters
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', '2048'))
    DEFAULT_TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))

    # Conversation settings
    MAX_CONVERSATIONS = int(os.getenv('MAX_CONVERSATIONS', '100'))
    CLEANUP_DAYS = int(os.getenv('CLEANUP_DAYS', '5'))

    # Template and static folders
    TEMPLATE_FOLDER = Path('templates')
    STATIC_FOLDER = Path('static')

    @classmethod
    def get_db_path(cls) -> Path:
        """Get database path, creating parent directories if needed."""
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return cls.DB_PATH

    @classmethod
    def get_template_folder(cls) -> Path:
        """Get template folder path."""
        return cls.TEMPLATE_FOLDER

    @classmethod
    def get_static_folder(cls) -> Path:
        """Get static folder path."""
        return cls.STATIC_FOLDER
