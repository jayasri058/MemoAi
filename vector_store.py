
import os
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

class PineconeManager:
    def __init__(self, api_key: str, index_name: str = "memo-ai-index"):
        if not api_key:
            print("Warning: No Pinecone API Key provided")
            self.index = None
            return

        try:
            print(f"Initializing Pinecone client...")
            self.pc = Pinecone(api_key=api_key)
            
            print(f"Checking for Pinecone index: {index_name}")
            # list_indexes returns an IndexList which is iterable
            active_indexes = self.pc.list_indexes()
            existing_names = [idx.name for idx in active_indexes]
            
            if index_name not in existing_names:
                print(f"Creating new Pinecone index: {index_name} because it was not in {existing_names}")
                self.pc.create_index(
                    name=index_name,
                    dimension=384,
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

    def upsert_vector(self, id: str, vector: List[float], metadata: Dict[str, Any]):
        """
        Upsert a single vector to Pinecone
        """
        return self.upsert_vectors([(id, vector, metadata)])

    def upsert_vectors(self, items: List[tuple]):
        """
        Upsert multiple vectors to Pinecone
        items: List of (id, vector, metadata)
        """
        if not self.index:
            return False
            
        try:
            vectors_to_upsert = []
            for item_id, vector, metadata in items:
                clean_metadata = {}
                for k, v in metadata.items():
                    if v is None: continue
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    elif isinstance(v, list):
                        clean_metadata[k] = [str(item) for item in v]
                    else:
                        clean_metadata[k] = str(v)
                
                vectors_to_upsert.append((str(item_id), vector, clean_metadata))

            # Batch upsert (Pinecone recommends batches of ~100)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            return True
        except Exception as e:
            print(f"Error batch upserting to Pinecone: {e}")
            return False

    def query_similarity(self, vector: List[float], top_k: int = 5, threshold: float = 0.0, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Query Pinecone for similar vectors
        """
        if not self.index:
            return []
            
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            matches = []
            for match in results['matches']:
                score = match['score']
                # Apply filtered threshold logic here to fix "irrelevant results"
                if score >= threshold:
                    matches.append({
                        'id': match['id'],
                        'score': score,
                        'metadata': match['metadata']
                    })
            
            return matches
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

    def delete_vector(self, id: str):
        """Delete vector by ID"""
        if not self.index:
            return
        try:
            self.index.delete(ids=[str(id)])
        except Exception as e:
            print(f"Error deleting from Pinecone: {e}")
