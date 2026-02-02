
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')
print(f"Testing with key: {api_key[:10]}...")

try:
    print("Attempting to init Pinecone object...")
    pc = Pinecone(api_key=api_key)
    print("Pinecone object created.")
    
    print("Attempting to list indexes...")
    indexes = pc.list_indexes()
    print(f"Indexes found: {indexes}")
    
    print("Attempting to connect to index...")
    index = pc.Index("memo-ai-index")
    print("Index connection successful.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
