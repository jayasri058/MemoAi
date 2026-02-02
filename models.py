"""
MemoAI Database Models
Defines the data structures for storing memories and user data
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os

class DatabaseManager:
    """Manages SQLite database operations for MemoAI"""
    
    def __init__(self, db_path: str = "memoai.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                voice_text TEXT,
                category TEXT,
                context TEXT,
                tags TEXT,  -- JSON array stored as text
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create search index
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_memories_search 
            ON memories(title, content, category, tags)
        ''')
        
        # Try to create FTS table for better search (SQLite 3.9.0+)
        try:
            # Check SQLite version first
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()[0]
            print(f"SQLite version: {version}")
            
            # FTS5 requires SQLite 3.9.0+
            if tuple(map(int, version.split('.'))) >= (3, 9, 0):
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        title, content, voice_text, category, context, tags,
                        content='memories', content_rowid='id'
                    )
                ''')
                
                # Create triggers to sync FTS index
                cursor.execute('''
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, title, content, voice_text, category, context, tags)
                        VALUES (new.id, new.title, new.content, new.voice_text, new.category, new.context, new.tags);
                    END;
                ''')
                
                cursor.execute('''
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                        DELETE FROM memories_fts WHERE rowid=old.id;
                        INSERT INTO memories_fts(rowid, title, content, voice_text, category, context, tags)
                        VALUES (new.id, new.title, new.content, new.voice_text, new.category, new.context, new.tags);
                    END;
                ''')
                
                cursor.execute('''
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                        DELETE FROM memories_fts WHERE rowid=old.id;
                    END;
                ''')
                print("FTS5 table created successfully")
            else:
                print(f"SQLite version {version} is too old for FTS5, using traditional search")
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not create FTS table: {e}")
            # Continue without FTS, fall back to regular search
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
    
    def save_memory(self, memory_data: Dict) -> int:
        """Save a memory to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert tags list to JSON string
        tags_json = json.dumps(memory_data.get('tags', []))
        
        cursor.execute('''
            INSERT INTO memories 
            (title, content, voice_text, category, context, tags, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_data.get('title', ''),
            memory_data.get('content', ''),
            memory_data.get('voice_text', ''),
            memory_data.get('category', ''),
            memory_data.get('context', ''),
            tags_json,
            memory_data.get('image_path', '')
        ))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id

    def create_user(self, name: str, email: str, password_hash: str) -> bool:
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
            ''', (name, email, password_hash))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
            
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, email, password_hash FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'email': row[2],
                'password_hash': row[3]
            }
        return None
    
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, email, password_hash FROM users ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        users = []
        for row in rows:
            users.append({
                'id': row[0],
                'name': row[1],
                'email': row[2],
                'password_hash': row[3]
            })
        return users
    
    def search_memories(self, query: str) -> List[Dict]:
        """Search memories by text query using FTS with fallback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First try FTS search
            cursor.execute('''
                SELECT m.id, m.title, m.content, m.voice_text, m.category, m.context, m.tags, 
                       m.image_path, m.created_at, m.updated_at,
                       bm25(memories_fts) as score
                FROM memories m
                JOIN memories_fts ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY score
            ''', (query,))
            
            rows = cursor.fetchall()
            print(f"FTS search returned {len(rows)} results")
        except sqlite3.OperationalError as e:
            # Fallback to traditional search if FTS is not available
            print(f"FTS not available, falling back to traditional search: {e}")
            search_term = f"%{query}%"
            cursor.execute('''
                SELECT id, title, content, voice_text, category, context, tags, 
                       image_path, created_at, updated_at
                FROM memories 
                WHERE title LIKE ? 
                   OR content LIKE ? 
                   OR voice_text LIKE ?
                   OR category LIKE ?
                   OR tags LIKE ?
                ORDER BY created_at DESC
            ''', (search_term, search_term, search_term, search_term, search_term))
            
            rows = cursor.fetchall()
            print(f"Traditional search returned {len(rows)} results")
        
        conn.close()
        
        memories = []
        for row in rows:
            # Parse tags JSON
            try:
                tags = json.loads(row[6]) if row[6] else []
            except json.JSONDecodeError:
                tags = []
            
            memories.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'voice_text': row[3],
                'category': row[4],
                'context': row[5],
                'tags': tags,
                'image_path': row[7],
                'created_at': row[8],
                'updated_at': row[9]
            })
        
        return memories
    
    def get_memory(self, memory_id: int) -> Optional[Dict]:
        """Get a specific memory by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, content, voice_text, category, context, tags, 
                   image_path, created_at, updated_at
            FROM memories 
            WHERE id = ?
        ''', (memory_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Parse tags JSON
        try:
            tags = json.loads(row[6]) if row[6] else []
        except json.JSONDecodeError:
            tags = []
        
        return {
            'id': row[0],
            'title': row[1],
            'content': row[2],
            'voice_text': row[3],
            'category': row[4],
            'context': row[5],
            'tags': tags,
            'image_path': row[7],
            'created_at': row[8],
            'updated_at': row[9]
        }
    
    def get_all_memories(self) -> List[Dict]:
        """Get all memories ordered by creation date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, content, voice_text, category, context, tags, 
                   image_path, created_at, updated_at
            FROM memories 
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        memories = []
        for row in rows:
            # Parse tags JSON
            try:
                tags = json.loads(row[6]) if row[6] else []
            except json.JSONDecodeError:
                tags = []
            
            memories.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'voice_text': row[3],
                'category': row[4],
                'context': row[5],
                'tags': tags,
                'image_path': row[7],
                'created_at': row[8],
                'updated_at': row[9]
            })
        
        return memories
    
    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
        rows_affected = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return rows_affected > 0
    
    def update_memory(self, memory_id: int, update_data: Dict) -> bool:
        """Update a memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build dynamic update query
        fields = []
        values = []
        
        for key, value in update_data.items():
            if key == 'tags':
                # Convert tags list to JSON
                fields.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key in ['title', 'content', 'voice_text', 'category', 'context', 'image_path']:
                fields.append(f"{key} = ?")
                values.append(value)
        
        if not fields:
            return False
        
        # Add memory_id to values
        values.append(memory_id)
        
        query = f"UPDATE memories SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        
        cursor.execute(query, values)
        rows_affected = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return rows_affected > 0

# Singleton instance
db_manager = DatabaseManager()

def get_db_manager() -> DatabaseManager:
    """Get the database manager instance"""
    return db_manager