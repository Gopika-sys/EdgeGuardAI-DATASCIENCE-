"""
EdgeGuard AI - Build RAG Index

Reads the synthetic SOP data, builds the TF-IDF vector index,
and syncs the documents to the Supabase database.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from rag import RAGRetriever

def main():
    print("=== EdgeGuard AI — Building RAG Layer ===")
    
    # 1. Build the local RAG index
    json_path = Path(__file__).parent / "sop_data.json"
    retriever = RAGRetriever()
    retriever.build_index(str(json_path))
    
    # 2. Sync to Supabase (optional — fails gracefully if no credentials)
    print("\nSyncing SOPs to Supabase `sop_documents` table...")
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key or "PASTE" in (url + key):
        print("⚠️ Warning: SUPABASE_URL or SUPABASE_KEY not configured. Skipping DB sync.")
        print("   (The local RAG index was built successfully — backend search will still work.)")
        return
    
    try:
        supabase: Client = create_client(url, key)
        
        # Clear existing to avoid duplicates during hackathon testing
        supabase.table("sop_documents").delete().neq("id", 0).execute()
        
        # Insert new
        for doc in retriever.documents:
            supabase.table("sop_documents").insert({
                "title": doc["title"],
                "component": doc["component"],
                "content_chunk": doc["content_chunk"],
                "embedding_ref": "tf-idf-local"
            }).execute()
            
        print("✅ Successfully synced SOP documents to Supabase.")
    except Exception as e:
        print(f"⚠️ Supabase sync failed (non-critical): {e}")
        print("   The local RAG index is still valid — backend will use local search.")

if __name__ == "__main__":
    main()
