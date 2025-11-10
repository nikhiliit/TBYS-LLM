"""Database operations for conversation management."""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any

from .config import Config


class DatabaseManager:
    """Manages SQLite database operations for conversations and messages."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or Config.get_db_path()
        self._local = threading.local()
        self.init_db()

    def init_db(self) -> None:
        """Initialize database tables."""
        with self.get_connection() as conn:
            c = conn.cursor()

            # Conversations table
            c.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Messages table
            c.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            ''')

            conn.commit()

    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)

        try:
            yield self._local.connection
        finally:
            pass  # Connection stays open for thread-local reuse

    def get_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent conversations.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation dictionaries
        """
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (limit,))

            conversations = []
            for row in c.fetchall():
                conversations.append({
                    'id': row[0],
                    'title': row[1],
                    'created_at': row[2],
                    'updated_at': row[3]
                })

            return conversations

    def create_conversation(self, title: str) -> int:
        """Create a new conversation.

        Args:
            title: Conversation title

        Returns:
            New conversation ID
        """
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO conversations (title) VALUES (?)', (title,))
            conversation_id = c.lastrowid
            conn.commit()
            return conversation_id

    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get conversation with messages.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation data with messages or None if not found
        """
        with self.get_connection() as conn:
            c = conn.cursor()

            # Get conversation info
            c.execute('SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?',
                     (conversation_id,))
            conv_row = c.fetchone()

            if not conv_row:
                return None

            # Get messages
            c.execute('''
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
            ''', (conversation_id,))

            messages = []
            for msg_row in c.fetchall():
                messages.append({
                    'role': msg_row[0],
                    'content': msg_row[1],
                    'created_at': msg_row[2]
                })

            return {
                'id': conv_row[0],
                'title': conv_row[1],
                'created_at': conv_row[2],
                'updated_at': conv_row[3],
                'messages': messages
            }

    def save_message(self, conversation_id: int, role: str, content: str) -> int:
        """Save a message to conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content

        Returns:
            New message ID
        """
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, ?, ?)
            ''', (conversation_id, role, content))

            # Update conversation timestamp
            c.execute('''
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (conversation_id,))

            conn.commit()
            return c.lastrowid

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
            deleted = c.rowcount > 0
            conn.commit()
            return deleted

    def cleanup_old_conversations(self, days: int = 5) -> int:
        """Clean up conversations older than specified days.

        Args:
            days: Number of days to keep conversations

        Returns:
            Number of conversations deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM conversations WHERE updated_at < ?', (cutoff_date,))
            deleted_count = c.rowcount
            conn.commit()
            return deleted_count

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


# Global database instance
db_manager = DatabaseManager()
