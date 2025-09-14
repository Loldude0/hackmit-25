import json
import os
from pinecone import Pinecone
from dotenv import load_dotenv
import uuid
# Load environment variables
load_dotenv()

def init_pinecone():
    """Initialize Pinecone client"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Connect to existing index or create if needed
    index_name = "songs-embeddings"
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model":"llama-text-embed-v2",
                "field_map":{"text": "chunk_text"}
            }
        )
        
    return pc.Index(index_name)

def create_text_for_embedding(song):
    """Create a neat string from description, genre, and tags"""
    description = song.get('description', '')
    genre = song.get('genre', '')
    tags = ', '.join(song.get('tags', []))
    
    text = f"{description} Genre: {genre} Tags: {tags}"
    return text.strip()

def embed_all_songs():
    """Embed all songs from data.json"""
    
    # Initialize Pinecone
    dense_index = init_pinecone()
    
    # Load data
    with open('data.json', 'r') as f:
        songs = json.load(f)
    
    print(f"Processing {len(songs)} songs...")
    
    # Prepare all records for batch upsert
    records = []
    
    for i, song in enumerate(songs):
        print(f"Processing song {i+1}/{len(songs)}: {song['title']} by {song['artist']}")
        
        # Create embedding text
        embedding_text = create_text_for_embedding(song)
        
        # Create record
        record = {
            '_id': str(uuid.uuid4()),
            'chunk_text': embedding_text,   # Pinecone will embed this
            'artist': song['artist'],
            'title': song['title'],
            'text': embedding_text.lower()
        }
        
        records.append(record)
    
    # Batch upsert all records
    print(f"Upserting {len(records)} records to Pinecone...")
    dense_index.upsert_records(
        namespace="song-embeddings", 
        records=records
    )

    stats = dense_index.describe_index_stats()
    print(f"Index stats: {stats}")
    print(f"Successfully embedded {len(songs)} songs using llama-text-embed-v2 model!")

if __name__ == "__main__":
    embed_all_songs()
