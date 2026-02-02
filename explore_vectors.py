"""
FAISS Vector Explorer
Interactive tool to explore and visualize your vector database
"""

import os
import pickle
import numpy as np
from typing import Optional

def load_vector_data():
    """Load FAISS index and memory mapping"""
    try:
        import faiss
    except ImportError:
        print("❌ Error: faiss-cpu not installed")
        print("Install with: pip install faiss-cpu")
        return None, None
    
    # Check if files exist
    if not os.path.exists("vector_index.faiss"):
        print("❌ Error: vector_index.faiss not found")
        print("Run migrate_to_vector.py first to create the index")
        return None, None
    
    if not os.path.exists("vector_memory.pkl"):
        print("❌ Error: vector_memory.pkl not found")
        return None, None
    
    # Load FAISS index
    print("Loading FAISS index...")
    vector_index = faiss.read_index("vector_index.faiss")
    
    # Load memory mapping
    with open("vector_memory.pkl", "rb") as f:
        vector_memory = pickle.load(f)
    
    print(f"✓ Loaded {vector_index.ntotal} vectors")
    return vector_index, vector_memory

def get_vector_stats(vector_index):
    """Get statistics about the vector index"""
    if vector_index is None:
        return
    
    print("\n" + "=" * 60)
    print("VECTOR DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total vectors:        {vector_index.ntotal}")
    print(f"Vector dimension:     {vector_index.d}")
    print(f"Index type:           {type(vector_index).__name__}")
    print(f"Is trained:           {vector_index.is_trained}")
    print("=" * 60)

def view_vector_by_index(vector_index, vector_memory, idx: int):
    """View a specific vector by its index"""
    if vector_index is None or idx >= vector_index.ntotal:
        print(f"❌ Invalid index. Must be between 0 and {vector_index.ntotal - 1}")
        return
    
    # Reconstruct the vector
    vector = vector_index.reconstruct(int(idx))
    memory_id = vector_memory[idx]
    
    print("\n" + "-" * 60)
    print(f"Vector Index: {idx}")
    print(f"Memory ID:    {memory_id}")
    print(f"Dimensions:   {len(vector)}")
    print(f"\nVector preview (first 10 dimensions):")
    print(vector[:10])
    print(f"\nVector statistics:")
    print(f"  Min:  {vector.min():.6f}")
    print(f"  Max:  {vector.max():.6f}")
    print(f"  Mean: {vector.mean():.6f}")
    print(f"  Std:  {vector.std():.6f}")
    print("-" * 60)

def search_similar_vectors(vector_index, vector_memory, idx: int, k: int = 5):
    """Find similar vectors to a given vector"""
    if vector_index is None or idx >= vector_index.ntotal:
        print(f"❌ Invalid index")
        return
    
    # Get the query vector
    query_vector = vector_index.reconstruct(int(idx))
    query_vector = np.array([query_vector], dtype=np.float32)
    
    # Search for similar vectors
    scores, indices = vector_index.search(query_vector, k)
    
    print("\n" + "=" * 60)
    print(f"TOP {k} SIMILAR VECTORS to Vector #{idx} (Memory ID: {vector_memory[idx]})")
    print("=" * 60)
    
    for i, (score, similar_idx) in enumerate(zip(scores[0], indices[0]), 1):
        memory_id = vector_memory[similar_idx]
        print(f"{i}. Vector #{similar_idx} (Memory ID: {memory_id}) - Score: {score:.4f}")
    
    print("=" * 60)

def export_vectors_to_csv(vector_index, vector_memory, output_file: str = "vectors_export.csv"):
    """Export all vectors to CSV for external analysis"""
    if vector_index is None:
        return
    
    print(f"\nExporting {vector_index.ntotal} vectors to {output_file}...")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("vector_index,memory_id," + ",".join([f"dim_{i}" for i in range(vector_index.d)]) + "\n")
        
        # Write vectors
        for idx in range(vector_index.ntotal):
            vector = vector_index.reconstruct(int(idx))
            memory_id = vector_memory[idx]
            f.write(f"{idx},{memory_id}," + ",".join([f"{v:.6f}" for v in vector]) + "\n")
    
    print(f"✓ Exported to {output_file}")
    print(f"  You can open this file in Excel, Google Sheets, or any data analysis tool")

def visualize_vectors_2d(vector_index, vector_memory):
    """Create a 2D visualization of vectors using t-SNE"""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ Visualization requires scikit-learn and matplotlib")
        print("Install with: pip install scikit-learn matplotlib")
        return
    
    if vector_index is None or vector_index.ntotal == 0:
        print("❌ No vectors to visualize")
        return
    
    print("\nGenerating 2D visualization using t-SNE...")
    print("(This may take a moment for large datasets)")
    
    # Get all vectors
    vectors = np.array([vector_index.reconstruct(int(i)) for i in range(vector_index.ntotal)])
    
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, vector_index.ntotal - 1))
    vectors_2d = tsne.fit_transform(vectors)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, s=50)
    
    # Add labels for first 20 vectors
    for i in range(min(20, len(vectors_2d))):
        plt.annotate(f"M{vector_memory[i]}", 
                    (vectors_2d[i, 0], vectors_2d[i, 1]),
                    fontsize=8, alpha=0.7)
    
    plt.title(f"Vector Database Visualization ({vector_index.ntotal} vectors)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    
    output_file = "vector_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_file}")
    
    try:
        plt.show()
    except:
        print("  (Display not available, but image saved)")

def interactive_explorer():
    """Interactive menu for exploring vectors"""
    vector_index, vector_memory = load_vector_data()
    
    if vector_index is None:
        return
    
    get_vector_stats(vector_index)
    
    while True:
        print("\n" + "=" * 60)
        print("FAISS VECTOR EXPLORER - MENU")
        print("=" * 60)
        print("1. View vector by index")
        print("2. Find similar vectors")
        print("3. Export vectors to CSV")
        print("4. Visualize vectors (2D)")
        print("5. Show statistics")
        print("6. Exit")
        print("=" * 60)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            idx = input(f"Enter vector index (0-{vector_index.ntotal - 1}): ").strip()
            try:
                view_vector_by_index(vector_index, vector_memory, int(idx))
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == '2':
            idx = input(f"Enter vector index (0-{vector_index.ntotal - 1}): ").strip()
            k = input("How many similar vectors to find? (default: 5): ").strip() or "5"
            try:
                search_similar_vectors(vector_index, vector_memory, int(idx), int(k))
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == '3':
            filename = input("Enter output filename (default: vectors_export.csv): ").strip() or "vectors_export.csv"
            export_vectors_to_csv(vector_index, vector_memory, filename)
        
        elif choice == '4':
            visualize_vectors_2d(vector_index, vector_memory)
        
        elif choice == '5':
            get_vector_stats(vector_index)
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          FAISS VECTOR DATABASE EXPLORER                    ║
    ║          Explore and visualize your vector data            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    interactive_explorer()
