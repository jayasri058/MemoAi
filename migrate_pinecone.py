import os
import time
from pinecone import Pinecone
from ai_services import GeminiService
from dotenv import load_dotenv

load_dotenv()

def migrate():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('Gemini_api_key')

    if not pinecone_api_key or not gemini_api_key:
        print("Missing API keys")
        return

    print("Initializing Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)
    old_index = pc.Index("memo-ai-index")
    new_index = pc.Index("memo-ai-index-gemini")

    print("Initializing Gemini...")
    gemini_service = GeminiService(api_key=gemini_api_key)

    print("Querying memories from old index...")
    try:
        mem_results = old_index.query(
            vector=[0.0001] * 384,
            top_k=10000,
            include_metadata=True,
            namespace='memories'
        )
        memories = getattr(mem_results, 'matches', [])

        print(f"Found {len(memories)} memories in old index.")

        for match in memories:
            meta = getattr(match, 'metadata', match.get('metadata', {}))
            mem_id = getattr(match, 'id', match.get('id'))
            
            # Recreate embedding
            search_text = f"{meta.get('content', '')} {meta.get('context', '')} {meta.get('tags', '[]')}"
            embedding = gemini_service.get_embedding(search_text)
            
            # Save to new index
            new_index.upsert(
                vectors=[(mem_id, embedding, meta)],
                namespace='memories'
            )
            print(f"Migrated memory: {mem_id}")
            time.sleep(0.5) # Avoid rate limits
            
    except Exception as e:
        print(f"Error migrating memories: {e}")

    print("Querying __default__ namespace for users...")
    try:
        user_results = old_index.query(
            vector=[0.0001] * 384,
            top_k=10000,
            include_metadata=True,
            namespace=''
        )
        users = getattr(user_results, 'matches', [])
        print(f"Found {len(users)} users in old index.")

        for match in users:
            meta = getattr(match, 'metadata', match.get('metadata', {}))
            user_id = getattr(match, 'id', match.get('id'))
            
            # Save to new index with 3072-D dummy vector
            new_index.upsert(
                vectors=[(user_id, [0.0001] * 3072, meta)],
                namespace='users'
            )
            print(f"Migrated user: {user_id}")
            
    except Exception as e:
        print(f"Error migrating users: {e}")

    print("Migration finished!")

if __name__ == "__main__":
    migrate()
