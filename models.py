"""
MemoAI Database Models - Pinecone-Only Backend
All data storage is handled through PineconeManager (vector_store.py).
This module provides the DatabaseManager interface that app.py expects,
but delegates everything to Pinecone.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    """
    Manages all data operations for MemoAI via Pinecone.
    Drop-in replacement for the old SQLite-based DatabaseManager.
    """

    def __init__(self):
        from vector_store import PineconeManager

        pinecone_api_key = os.getenv('PINECONE_API_KEY', '')
        pinecone_index = os.getenv('PINECONE_INDEX_NAME', 'memo-ai-index')

        self.pinecone = PineconeManager(
            api_key=pinecone_api_key,
            index_name=pinecone_index,
        )
        print(f"DatabaseManager initialized with Pinecone (index: {pinecone_index})")

    # ───────────────────────────────────────────────
    #  USER OPERATIONS
    # ───────────────────────────────────────────────

    def create_user(self, name: str, email: str, password_hash: str) -> bool:
        """Create a new user"""
        result = self.pinecone.create_user(name, email, password_hash)
        return result is not None

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        return self.pinecone.get_user_by_email(email)

    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        return self.pinecone.get_all_users()

    # ───────────────────────────────────────────────
    #  MEMORY OPERATIONS
    # ───────────────────────────────────────────────

    def save_memory(self, user_id: int, memory_data: Dict, vector: List[float] = None) -> int:
        """
        Save a memory to Pinecone.
        If no vector is provided, a zero vector is used (text-only memory).
        """
        if vector is None:
            vector = [0.0] * self.pinecone.dimension

        memory_id = self.pinecone.save_memory(user_id, memory_data, vector)
        return memory_id if memory_id else 0

    def get_memory(self, memory_id: int, user_id: Optional[int] = None) -> Optional[Dict]:
        """Get a specific memory by ID"""
        return self.pinecone.get_memory(memory_id, user_id=user_id)

    def get_all_memories(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get all memories ordered by creation date"""
        return self.pinecone.get_all_memories(user_id=user_id)

    def search_memories(self, query: str, user_id: Optional[int] = None) -> List[Dict]:
        """Search memories by text query (in-memory filter fallback)"""
        return self.pinecone.search_memories(query, user_id=user_id)

    def delete_memory(self, memory_id: int, user_id: Optional[int] = None) -> bool:
        """Delete a memory by ID"""
        return self.pinecone.delete_memory(memory_id, user_id=user_id)

    def update_memory(self, memory_id: int, update_data: Dict, user_id: Optional[int] = None) -> bool:
        """Update a memory"""
        return self.pinecone.update_memory(memory_id, update_data, user_id=user_id)

    # ───────────────────────────────────────────────
    #  INIT (no-op, Pinecone handles initialization)
    # ───────────────────────────────────────────────

    def init_database(self):
        """No-op for Pinecone backend. Index is created at connection time."""
        pass


# Singleton instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the database manager instance"""
    return db_manager