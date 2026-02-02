"""
Migration Script: SQLite to Vector Database
Migrates all existing memories from SQLite to FAISS vector index
"""

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

def migrate_to_vector_db():
    """Migrate all memories from SQLite to vector database"""
    
    print("=" * 60)
    print("Starting Migration: SQLite → Vector Database")
    print("=" * 60)
    
    # Initialize embedder
    print("\n[1/5] Loading sentence transformer model...")
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Initialize FAISS index
    print("\n[2/5] Initializing FAISS vector index...")
    vector_index = faiss.IndexFlatIP(dimension)
    vector_memory = []
    print("✓ FAISS index initialized")
    
    # Connect to SQLite database
    print("\n[3/5] Connecting to SQLite database...")
    db_path = "memoai.db"
    if not os.path.exists(db_path):
        print(f"✗ Database file '{db_path}' not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print("✓ Connected to database")
    
    # Fetch all memories
    print("\n[4/5] Fetching all memories from SQLite...")
    cursor.execute('''
        SELECT id, title, content, voice_text, category, context, tags
        FROM memories
        ORDER BY id
    ''')
    
    memories = cursor.fetchall()
    total_memories = len(memories)
    print(f"✓ Found {total_memories} memories to migrate")
    
    if total_memories == 0:
        print("\n⚠ No memories found in database. Nothing to migrate.")
        conn.close()
        return
    
    # Process and add to vector index
    print("\n[5/5] Processing memories and creating embeddings...")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    for idx, memory in enumerate(memories, 1):
        memory_id, title, content, voice_text, category, context, tags = memory
        
        try:
            # Combine all text fields for embedding
            text_parts = []
            if title:
                text_parts.append(f"Title: {title}")
            if content:
                text_parts.append(f"Content: {content}")
            if voice_text:
                text_parts.append(f"Voice: {voice_text}")
            if category:
                text_parts.append(f"Category: {category}")
            if context:
                text_parts.append(f"Context: {context}")
            
            combined_text = " | ".join(text_parts)
            
            if not combined_text.strip():
                print(f"  [{idx}/{total_memories}] Memory ID {memory_id}: Skipped (empty)")
                failed += 1
                continue
            
            # Generate embedding
            embedding = embedder.encode(combined_text, convert_to_numpy=True)
            
            # Normalize for cosine similarity (required for IndexFlatIP)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Add to FAISS index
            vector_index.add(np.array([embedding], dtype=np.float32))
            vector_memory.append(memory_id)
            
            successful += 1
            
            # Progress indicator
            if idx % 10 == 0 or idx == total_memories:
                print(f"  Progress: {idx}/{total_memories} memories processed...")
            
        except Exception as e:
            print(f"  [{idx}/{total_memories}] Memory ID {memory_id}: Failed - {e}")
            failed += 1
    
    conn.close()
    
    # Save vector index to disk
    print("\n[6/6] Saving vector index to disk...")
    try:
        # Save FAISS index
        faiss.write_index(vector_index, "vector_index.faiss")
        
        # Save memory ID mapping
        with open("vector_memory.pkl", "wb") as f:
            pickle.dump(vector_memory, f)
        
        print("✓ Vector index saved to 'vector_index.faiss'")
        print("✓ Memory mapping saved to 'vector_memory.pkl'")
    except Exception as e:
        print(f"✗ Error saving index: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total memories in SQLite: {total_memories}")
    print(f"Successfully migrated:     {successful}")
    print(f"Failed/Skipped:            {failed}")
    print(f"Vector index size:         {vector_index.ntotal} vectors")
    print("=" * 60)
    
    if successful > 0:
        print("\n✓ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Restart your Flask app (python app.py)")
        print("2. The app will automatically load the vector index")
        print("3. Test vector search with /api/search-memories endpoint")
    else:
        print("\n⚠ No memories were migrated. Please check your database.")

if __name__ == "__main__":
    migrate_to_vector_db()
