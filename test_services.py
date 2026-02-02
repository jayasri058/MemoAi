
import os
import traceback
import sys
from dotenv import load_dotenv

# Ensure we are in the right directory
os.chdir(r'e:\DataScience\memo\memo')
load_dotenv()

def test():
    print(f"Python Version: {sys.version}")
    
    print("\n--- Testing Pinecone ---")
    try:
        from vector_store import PineconeManager
        p_key = os.getenv('PINECONE_API_KEY')
        if not p_key:
            print("ERROR: PINECONE_API_KEY is missing from .env")
        else:
            print(f"Key found: {p_key[:10]}...")
            pm = PineconeManager(p_key)
            print("Pinecone Initialization successful")
    except Exception:
        print("Pinecone Initialization FAILED")
        traceback.print_exc()

    print("\n--- Testing Gemini ---")
    try:
        from ai_services import GeminiService
        g_key = os.getenv('GEMINI_API_KEY') or os.getenv('Gemini_api_key')
        if not g_key:
            print("ERROR: GEMINI_API_KEY is missing from .env")
        else:
            print(f"Key found: {g_key[:10]}...")
            gs = GeminiService(g_key)
            print("Gemini Initialization successful")
    except Exception:
        print("Gemini Initialization FAILED")
        traceback.print_exc()

if __name__ == "__main__":
    test()
