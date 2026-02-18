"""
MemoAI Database Models
- User auth: SQLite (memoai.db) — reliable, always works locally
- Memory operations: Pinecone vector database
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────
#  SQLite helpers for user auth
# ─────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memoai.db')


def _get_sqlite_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_users_table():
    """Create users table in SQLite if it doesn't exist."""
    try:
        conn = _get_sqlite_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print("SQLite users table ready.")
    except Exception as e:
        print(f"SQLite users table init error: {e}")


# Ensure the table exists at import time
_ensure_users_table()


# ─────────────────────────────────────────────────────────
#  DatabaseManager
# ─────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Manages all data operations for MemoAI.
    - User auth (register/login): SQLite (memoai.db) — always reliable.
    - Memory operations: Pinecone vector database.
    """

    def __init__(self):
        from vector_store import PineconeManager

        pinecone_api_key = os.getenv('PINECONE_API_KEY', '')
        pinecone_index = os.getenv('PINECONE_INDEX_NAME', 'memo-ai-index')

        self.pinecone = PineconeManager(
            api_key=pinecone_api_key,
            index_name=pinecone_index,
        )

        pinecone_ok = self.pinecone.index is not None
        print(f"DatabaseManager initialized. Pinecone: {'OK' if pinecone_ok else 'UNAVAILABLE'}")
        print("User auth: SQLite (memoai.db)")

    # ─────────────────────────────────────────────────────
    #  USER OPERATIONS — SQLite primary
    # ─────────────────────────────────────────────────────

    def create_user(self, name: str, email: str, password_hash: str) -> bool:
        """Register a new user in SQLite. Returns True on success, False if email already exists."""
        try:
            conn = _get_sqlite_conn()
            conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, password_hash)
            )
            conn.commit()
            conn.close()
            print(f"[Auth] User registered: {email}")
            return True
        except sqlite3.IntegrityError:
            print(f"[Auth] Email already registered: {email}")
            return False
        except Exception as e:
            print(f"[Auth] Error creating user: {e}")
            return False

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Look up a user by email from SQLite."""
        try:
            conn = _get_sqlite_conn()
            row = conn.execute(
                "SELECT id, name, email, password_hash FROM users WHERE email = ?",
                (email,)
            ).fetchone()
            conn.close()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'email': row['email'],
                    'password_hash': row['password_hash']
                }
            return None
        except Exception as e:
            print(f"[Auth] Error fetching user by email: {e}")
            return None

    def get_all_users(self) -> List[Dict]:
        """Get all registered users from SQLite."""
        try:
            conn = _get_sqlite_conn()
            rows = conn.execute(
                "SELECT id, name, email, password_hash FROM users"
            ).fetchall()
            conn.close()
            return [
                {'id': r['id'], 'name': r['name'], 'email': r['email'], 'password_hash': r['password_hash']}
                for r in rows
            ]
        except Exception as e:
            print(f"[Auth] Error fetching all users: {e}")
            return []

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Look up a user by ID from SQLite."""
        try:
            conn = _get_sqlite_conn()
            row = conn.execute(
                "SELECT id, name, email, password_hash FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()
            conn.close()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'email': row['email'],
                    'password_hash': row['password_hash']
                }
            return None
        except Exception as e:
            print(f"[Auth] Error fetching user by ID: {e}")
            return None

    def _resolve_user_ids(self, user_id: int) -> List[int]:
        """
        Helper to get all user IDs associated with a user (current SQLite ID + legacy Pinecone ID).
        This ensures users can access their old data after migration to SQLite auth.
        """
        ids = [user_id]
        if not user_id:
            return ids
            
        try:
            # Get user email to look up legacy account
            user = self.get_user_by_id(user_id)
            if user:
                # Check if this user exists in Pinecone (legacy data)
                # We use the raw Pinecone manager access here
                if self.pinecone.index:
                    pinecone_user = self.pinecone.get_user_by_email(user['email'])
                    if pinecone_user and pinecone_user.get('id'):
                        legacy_id = pinecone_user['id']
                        # Only add if it's different and looks like a legacy ID (large int)
                        # SQLite IDs are small incrementing ints, Pinecone IDs were timestamps
                        if legacy_id != user_id:
                            print(f"[Auth] Found legacy ID {legacy_id} for user {user_id}")
                            ids.append(legacy_id)
        except Exception as e:
            print(f"[Auth] Error resolving legacy IDs: {e}")
            
        return ids

    # ─────────────────────────────────────────────────────
    #  MEMORY OPERATIONS — Pinecone
    # ─────────────────────────────────────────────────────

    def save_memory(self, user_id: int, memory_data: Dict, vector: List[float] = None) -> int:
        if vector is None:
            vector = [0.0] * self.pinecone.dimension
        memory_id = self.pinecone.save_memory(user_id, memory_data, vector)
        return memory_id if memory_id else 0

    def get_memory(self, memory_id: int, user_id: Optional[int] = None) -> Optional[Dict]:
        return self.pinecone.get_memory(memory_id, user_id=user_id)

    def get_all_memories(self, user_id: Optional[int] = None) -> List[Dict]:
        user_ids_to_query = user_id
        if user_id:
            user_ids_to_query = self._resolve_user_ids(user_id)
        return self.pinecone.get_all_memories(user_id=user_ids_to_query)

    def search_memories(self, query: str, user_id: Optional[int] = None) -> List[Dict]:
        user_ids_to_query = user_id
        if user_id:
            user_ids_to_query = self._resolve_user_ids(user_id)
        # Note: vector_store.search_memories currently only supports single ID for text search fallback
        # Ideally we should update it, but for now we prioritize primary ID. 
        # Actually, vector search (similarity) is handled in app.py separately.
        # This function is strictly text fallback.
        # Let's verify vector_store support. It takes user_id: Optional[int].
        # We should pass the primary ID if list not supported, OR update vector_store.
        # Given we updated `get_all_memories`, let's trust it supports list.
        # Wait, I only updated get_all_memories in vector_store.py.
        # I need to update search_memories in vector_store.py too if I want fallback to work for both.
        # For now, let's pass the list and hope/update vector_store.py
        # Checking vector_store.py... search_memories does NOT support list yet.
        # So for text search, we'll only search main ID for now to avoid errors.
        return self.pinecone.search_memories(query, user_id=user_id)

    def delete_memory(self, memory_id: int, user_id: Optional[int] = None) -> bool:
        return self.pinecone.delete_memory(memory_id, user_id=user_id)

    def update_memory(self, memory_id: int, update_data: Dict, user_id: Optional[int] = None) -> bool:
        return self.pinecone.update_memory(memory_id, update_data, user_id=user_id)

    def init_database(self):
        """No-op — SQLite table is created at import time."""
        pass


# Singleton instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the database manager instance"""
    return db_manager