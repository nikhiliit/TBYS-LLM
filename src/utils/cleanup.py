"""Cleanup utilities for maintenance tasks."""

from datetime import datetime, timedelta

from database import db_manager
from config import Config


def cleanup_old_conversations(days: int = None) -> int:
    """Clean up conversations older than specified days.

    Args:
        days: Number of days to keep conversations (default: from config)

    Returns:
        Number of conversations deleted
    """
    if days is None:
        days = Config.CLEANUP_DAYS

    return db_manager.cleanup_old_conversations(days)
