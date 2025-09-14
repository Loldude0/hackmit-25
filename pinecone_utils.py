import os
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

def search_songs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for songs using RAG query against the embedded song database.
    
    Args:
        query (str): RAG query describing the desired music (e.g., "high energy rap workout music")
        top_k (int): Number of top results to return (default: 5)
    
    Returns:
        List[Dict]: List of song results with metadata (title, artist, text)
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Connect to the same index used in chunk.py
    index_name = "songs-embeddings"
    index = pc.Index(index_name)
    
    # For Pinecone 7.x with serverless managed embeddings
    # We need to use the inference API to generate embeddings first, then query
    
    # Step 1: Generate embedding using Pinecone's inference API
    pc_inference = pc.inference  # Use the inference client
    
    try:
        # Generate embedding using the same model as the index
        embedding_response = pc_inference.embed(
            model="llama-text-embed-v2",
            inputs=[{"text": query}],
            parameters={"input_type": "query", "truncate": "END"}
        )
        
        # Extract the embedding vector
        query_vector = embedding_response.data[0].values
        
        # Step 2: Query using the generated embedding
        results = index.query(
            namespace="song-embeddings",
            include_metadata=True,
            top_k=top_k,
            vector=query_vector
        )
        
    except Exception as e:
        print(f"Inference embedding failed: {e}")
        # Fallback: try direct query (this might work in some configurations)
        results = index.query(
            namespace="song-embeddings", 
            include_metadata=True,
            top_k=top_k,
            vector=[]  # Empty vector as placeholder
        )
    
    # Extract and format results
    songs = []
    for match in results.matches:
        song = {
            'title': match.metadata.get('title', 'Unknown'),
            'artist': match.metadata.get('artist', 'Unknown'),
            'text': match.metadata.get('text', ''),
            'score': match.score
        }
        songs.append(song)
    
    return songs

def check_index_status():
    """Check if the index has data and what's in it"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # List all indexes
    indexes = pc.list_indexes()
    print(f"Available indexes: {indexes}")
    
    index = pc.Index("songs-embeddings")
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    
    # Check if there are records in the song-embeddings namespace
    namespace_stats = stats.get('namespaces', {}).get('song-embeddings', {})
    vector_count = namespace_stats.get('vector_count', 0)
    
    print(f"Vectors in 'song-embeddings' namespace: {vector_count}")
    
    # Check all namespaces
    all_namespaces = stats.get('namespaces', {})
    print(f"All namespaces: {list(all_namespaces.keys())}")
    
    if vector_count == 0:
        print("❌ No vectors found in the namespace! You need to run 'python songs/chunk.py' first to embed the songs.")
        print("\nDebugging steps:")
        print("1. cd songs")
        print("2. python chunk.py")
        print("3. Wait for it to complete")
        print("4. Then try the search again")
    else:
        print(f"✅ Found {vector_count} songs in the index")
    
    return vector_count > 0

if __name__ == "__main__":
    # For testing
    print("🔍 Checking index status...")
    check_index_status()
    print("\n🔍 Testing search...")
    try:
        songs = search_songs("workout music", top_k=2)
        print(f"Found {len(songs)} songs")
        for song in songs:
            print(f"- {song['title']} by {song['artist']}")
    except Exception as e:
        print(f"Search failed: {e}")
