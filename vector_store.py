"""
MemoAI Unified Pinecone Storage Layer
All data (users + memories) stored in Pinecone.
Uses namespaces to separate user data from memory vectors.
"""

import os
import time
import uuid
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime


class PineconeManager:
    """
    Unified storage manager using Pinecone for everything:
    - 'users' namespace: stores user accounts (email, name, password_hash)
    - 'memories' namespace: stores memory vectors with full metadata
    """

    USERS_NAMESPACE = "users"
    MEMORIES_NAMESPACE = "memories"

    def __init__(self, api_key: str, index_name: str = "memo-ai-index", dimension: int = 384):
        self.index = None
        self.dimension = dimension

        if not api_key:
            print("Warning: No Pinecone API Key provided")
            return

        try:
            print("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=api_key)

            # Check / create index
            active_indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in active_indexes]

            if index_name not in existing_names:
                print(f"Creating new Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                while not self.pc.describe_index(index_name).status['ready']:
                    print("Waiting for index to be ready...")
                    time.sleep(1)

            self.index = self.pc.Index(index_name)
            print(f"Pinecone Index '{index_name}' connection established")

        except Exception as e:
            print(f"FAILED to initialize Pinecone: {e}")
            import traceback
            traceback.print_exc()
            self.index = None

    # ───────────────────────────────────────────────
    #  USER OPERATIONS (namespace: users)
    # ───────────────────────────────────────────────

    def create_user(self, name: str, email: str, password_hash: str) -> Optional[int]:
        """
        Create a new user in Pinecone.
        Returns a generated user_id (int) or None on failure.
        Uses email as the vector ID to enforce uniqueness.
        """
        if not self.index:
            return None

        # Check if user already exists
        existing = self._fetch_user_by_email(email)
        if existing:
            return None  # User already exists

        # Generate a numeric user_id
        user_id = int(datetime.now().timestamp() * 1000) % 2_000_000_000

        # Store user - use a dummy vector (users don't need real embeddings)
        dummy_vector = [0.0] * self.dimension

        metadata = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.now().isoformat(),
            "type": "user"
        }

        try:
            self.index.upsert(
                vectors=[(f"user_{email}", dummy_vector, metadata)],
                namespace=self.USERS_NAMESPACE
            )
            print(f"User created: {email} (ID: {user_id})")
            return user_id
        except Exception as e:
            print(f"Error creating user in Pinecone: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email address"""
        return self._fetch_user_by_email(email)

    def _fetch_user_by_email(self, email: str) -> Optional[Dict]:
        """Internal: fetch user record from Pinecone by email"""
        if not self.index:
            return None

        try:
            result = self.index.fetch(
                ids=[f"user_{email}"],
                namespace=self.USERS_NAMESPACE
            )

            vectors = result.get('vectors', {})
            if f"user_{email}" in vectors:
                meta = vectors[f"user_{email}"]["metadata"]
                return {
                    'id': meta.get('user_id'),
                    'name': meta.get('name'),
                    'email': meta.get('email'),
                    'password_hash': meta.get('password_hash')
                }
            return None
        except Exception as e:
            print(f"Error fetching user by email: {e}")
            return None

    def get_all_users(self) -> List[Dict]:
        """
        Get all users. Uses a metadata filter query on the users namespace.
        Since Pinecone requires a vector for queries, we use a zero vector.
        """
        if not self.index:
            return []

        try:
            dummy_vector = [0.0] * self.dimension
            results = self.index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True,
                filter={"type": {"$eq": "user"}},
                namespace=self.USERS_NAMESPACE
            )

            users = []
            for match in results.get('matches', []):
                meta = match.get('metadata', {})
                users.append({
                    'id': meta.get('user_id'),
                    'name': meta.get('name'),
                    'email': meta.get('email'),
                    'password_hash': meta.get('password_hash')
                })
            return users
        except Exception as e:
            print(f"Error fetching all users: {e}")
            return []

    # ───────────────────────────────────────────────
    #  MEMORY OPERATIONS (namespace: memories)
    # ───────────────────────────────────────────────

    def save_memory(self, user_id: int, memory_data: Dict, vector: List[float]) -> Optional[int]:
        """
        Save a memory to Pinecone with its embedding vector.
        Returns a generated memory_id (int).
        """
        if not self.index:
            return None

        memory_id = int(datetime.now().timestamp() * 1000) % 2_000_000_000

        # Build metadata (Pinecone metadata values must be str, int, float, bool, or list[str])
        import json
        tags = memory_data.get('tags', [])
        tags_str = json.dumps(tags) if isinstance(tags, list) else str(tags)

        metadata = {
            "memory_id": memory_id,
            "user_id": user_id,
            "title": str(memory_data.get('title', '')),
            "content": str(memory_data.get('content', ''))[:1000],  # Pinecone metadata limit
            "voice_text": str(memory_data.get('voice_text', ''))[:1000],
            "category": str(memory_data.get('category', '')),
            "context": str(memory_data.get('context', ''))[:1000],
            "tags": tags_str,
            "image_path": str(memory_data.get('image_path', '') or ''),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "type": memory_data.get('type', 'memory'),
            "has_image": bool(memory_data.get('has_image', False)),
            "date": datetime.now().strftime('%Y-%m-%d'),
        }

        try:
            self.index.upsert(
                vectors=[(str(memory_id), vector, metadata)],
                namespace=self.MEMORIES_NAMESPACE
            )
            return memory_id
        except Exception as e:
            print(f"Error saving memory to Pinecone: {e}")
            return None

    def save_memory_chunks(self, memory_id: int, user_id: int, chunks: List[Dict]):
        """
        Save multiple chunks for a single memory (e.g., PDF chunks).
        Each chunk dict should have 'vector', 'metadata'.
        """
        if not self.index:
            return False

        try:
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{memory_id}_{i}"
                meta = chunk.get('metadata', {})
                meta['memory_id'] = memory_id
                meta['user_id'] = user_id
                meta['chunk_index'] = i
                meta['type'] = 'pdf_chunk'
                meta['created_at'] = datetime.now().isoformat()
                meta['date'] = datetime.now().strftime('%Y-%m-%d')

                # Clean metadata values
                clean_meta = self._clean_metadata(meta)
                vectors_to_upsert.append((chunk_id, chunk['vector'], clean_meta))

            # Batch upsert
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=self.MEMORIES_NAMESPACE)

            return True
        except Exception as e:
            print(f"Error saving memory chunks to Pinecone: {e}")
            return False

    def get_memory(self, memory_id: int, user_id: Optional[int] = None) -> Optional[Dict]:
        """Get a specific memory by ID"""
        if not self.index:
            return None

        try:
            result = self.index.fetch(
                ids=[str(memory_id)],
                namespace=self.MEMORIES_NAMESPACE
            )

            vectors = result.get('vectors', {})
            if str(memory_id) in vectors:
                meta = vectors[str(memory_id)].get('metadata', {})

                # Check user_id scoping
                if user_id and meta.get('user_id') != user_id:
                    return None

                return self._metadata_to_memory(meta)
            return None
        except Exception as e:
            print(f"Error fetching memory: {e}")
            return None

    def get_all_memories(self, user_id: Optional[int] = None) -> List[Dict]:
        """
        Get all memories for a user. 
        Uses a query with zero vector and user_id filter.
        """
        if not self.index:
            return []

        try:
            dummy_vector = [0.0] * self.dimension
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = {"$eq": user_id}
            # Exclude chunks - only get main memories
            filter_dict["type"] = {"$ne": "pdf_chunk"}

            results = self.index.query(
                vector=dummy_vector,
                top_k=1000,
                include_metadata=True,
                filter=filter_dict,
                namespace=self.MEMORIES_NAMESPACE
            )

            memories = []
            for match in results.get('matches', []):
                meta = match.get('metadata', {})
                memories.append(self._metadata_to_memory(meta))

            # Sort by created_at descending
            memories.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return memories
        except Exception as e:
            print(f"Error fetching all memories: {e}")
            return []

    def search_memories(self, query_text: str, user_id: Optional[int] = None) -> List[Dict]:
        """
        Text-based search fallback. In Pinecone-only mode, this does a 
        metadata scan. For proper semantic search, use query_similarity().
        """
        # For text search fallback, get all memories and do in-memory filter
        all_memories = self.get_all_memories(user_id=user_id)
        query_lower = query_text.lower()

        results = []
        for mem in all_memories:
            searchable = f"{mem.get('title', '')} {mem.get('content', '')} {mem.get('voice_text', '')} {mem.get('category', '')} {mem.get('tags', '')}".lower()
            if query_lower in searchable:
                results.append(mem)

        return results

    def query_similarity(self, vector: List[float], user_id: int, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Query Pinecone for similar vectors, scoped by user_id.
        """
        if not self.index:
            return []

        try:
            filter_dict = {"user_id": {"$eq": user_id}}

            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
                namespace=self.MEMORIES_NAMESPACE
            )

            matches = []
            for match in results.get('matches', []):
                score = match.get('score', 0)
                if score >= threshold:
                    matches.append({
                        'id': match['id'],
                        'score': score,
                        'metadata': match.get('metadata', {})
                    })

            return matches
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

    def delete_memory(self, memory_id: int, user_id: Optional[int] = None) -> bool:
        """Delete a memory by ID (and its chunks)"""
        if not self.index:
            return False

        try:
            # First verify ownership if user_id provided
            if user_id:
                memory = self.get_memory(memory_id, user_id=user_id)
                if not memory:
                    return False

            # Delete the main memory vector
            ids_to_delete = [str(memory_id)]

            # Also delete any associated chunks (e.g., PDF chunks: memoryId_0, memoryId_1, ...)
            for i in range(100):  # Max 100 chunks per memory
                ids_to_delete.append(f"{memory_id}_{i}")

            self.index.delete(
                ids=ids_to_delete,
                namespace=self.MEMORIES_NAMESPACE
            )
            return True
        except Exception as e:
            print(f"Error deleting memory from Pinecone: {e}")
            return False

    def update_memory(self, memory_id: int, update_data: Dict, user_id: Optional[int] = None) -> bool:
        """
        Update a memory's metadata in Pinecone.
        Re-upserts with the same ID and updated metadata.
        """
        if not self.index:
            return False

        try:
            # Fetch existing
            result = self.index.fetch(
                ids=[str(memory_id)],
                namespace=self.MEMORIES_NAMESPACE
            )

            vectors = result.get('vectors', {})
            if str(memory_id) not in vectors:
                return False

            existing = vectors[str(memory_id)]
            meta = existing.get('metadata', {})

            # Check user ownership
            if user_id and meta.get('user_id') != user_id:
                return False

            # Update allowed fields
            import json
            for key, value in update_data.items():
                if key == 'tags':
                    meta['tags'] = json.dumps(value) if isinstance(value, list) else str(value)
                elif key in ['title', 'content', 'voice_text', 'category', 'context', 'image_path']:
                    meta[key] = str(value)[:1000]  # Respect metadata size limits

            meta['updated_at'] = datetime.now().isoformat()

            # Re-upsert with same vector but updated metadata
            existing_vector = existing.get('values', [0.0] * self.dimension)
            self.index.upsert(
                vectors=[(str(memory_id), existing_vector, meta)],
                namespace=self.MEMORIES_NAMESPACE
            )
            return True
        except Exception as e:
            print(f"Error updating memory in Pinecone: {e}")
            return False

    def upsert_vector(self, id: str, vector: List[float], metadata: Dict[str, Any], user_id: int):
        """
        Direct vector upsert for backward compatibility.
        """
        if not self.index:
            return False

        metadata['user_id'] = user_id
        clean_meta = self._clean_metadata(metadata)

        try:
            self.index.upsert(
                vectors=[(str(id), vector, clean_meta)],
                namespace=self.MEMORIES_NAMESPACE
            )
            return True
        except Exception as e:
            print(f"Error upserting vector to Pinecone: {e}")
            return False

    def delete_vector(self, id: str):
        """Delete vector by ID"""
        if not self.index:
            return
        try:
            self.index.delete(ids=[str(id)], namespace=self.MEMORIES_NAMESPACE)
        except Exception as e:
            print(f"Error deleting from Pinecone: {e}")

    # ───────────────────────────────────────────────
    #  HELPERS
    # ───────────────────────────────────────────────

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata to only contain Pinecone-compatible types"""
        clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = [str(item) for item in v]
            else:
                clean[k] = str(v)
        return clean

    def _metadata_to_memory(self, meta: Dict) -> Dict:
        """Convert Pinecone metadata to memory dict format"""
        import json

        # Parse tags
        tags_raw = meta.get('tags', '[]')
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        except (json.JSONDecodeError, TypeError):
            tags = []

        return {
            'id': meta.get('memory_id'),
            'title': meta.get('title', ''),
            'content': meta.get('content', ''),
            'voice_text': meta.get('voice_text', ''),
            'category': meta.get('category', ''),
            'context': meta.get('context', ''),
            'tags': tags,
            'image_path': meta.get('image_path', ''),
            'created_at': meta.get('created_at', ''),
            'updated_at': meta.get('updated_at', ''),
        }
