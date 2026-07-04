"""
EdgeGuard AI - RAG Layer (TF-IDF Vector Search)

A lightweight vector search engine for the hackathon using scikit-learn.
It acts exactly like Chroma/FAISS by embedding documents and querying them
using cosine similarity, but requires zero C++ build tools or external daemons.

Supports enriched SOP documents with severity, tools_required, and
estimated_downtime fields (v2 — June 2026).
"""

import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from features import get_feature_columns, SENSOR_COLS
import pickle

INDEX_PATH = Path(__file__).parent / "models" / "rag_index.pkl"

class RAGRetriever:
    def __init__(self):
        self.vectorizer = None
        self.doc_vectors = None
        self.documents = []
        
    def build_index(self, json_path: str):
        """Reads SOP JSON, builds the TF-IDF vector index, and saves it."""
        print(f"Building RAG index from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
            
        # Embed: component + severity + title + content for richer matching
        texts_to_embed = [
            f"{doc['component']} {doc.get('severity', '')} {doc['title']} {doc['content_chunk']}" 
            for doc in self.documents
        ]
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(texts_to_embed)
        
        # Save to disk
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'doc_vectors': self.doc_vectors,
                'documents': self.documents
            }, f)
        print(f"✅ RAG index built and saved to {INDEX_PATH}")
        print(f"   {len(self.documents)} documents indexed.")
        
    def load_index(self):
        """Loads the pre-built index."""
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"RAG index not found at {INDEX_PATH}. Run build_rag_index.py first.")
            
        with open(INDEX_PATH, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.doc_vectors = data['doc_vectors']
            self.documents = data['documents']
            
    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0):
        """
        Queries the index and returns the most relevant SOP documents.
        
        Args:
            query:     free-text search query
            top_k:     number of results to return
            min_score: minimum cosine similarity to include (0.0 = any match)
            
        Returns:
            list of dicts, each containing the SOP document fields PLUS
            a 'similarity_score' float.
        """
        if self.vectorizer is None:
            self.load_index()
            
        # Embed the query
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarity between query and all documents
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        
        # Get top_k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > min_score:
                doc = dict(self.documents[idx])  # copy to avoid mutating original
                doc['similarity_score'] = round(score, 4)
                doc['doc_index'] = int(idx)
                results.append(doc)
                
        return results

    def retrieve_by_component(self, component: str, severity: str = None):
        """
        Returns all SOPs for a given component, optionally filtered by severity.
        No vector search — straightforward filter.
        """
        if not self.documents:
            self.load_index()
            
        results = [doc for doc in self.documents if doc['component'] == component]
        if severity:
            results = [doc for doc in results if doc.get('severity') == severity]
        return results

    def get_all_documents(self):
        """Returns all SOP documents."""
        if not self.documents:
            self.load_index()
        return self.documents

    def get_components(self):
        """Returns unique component names."""
        if not self.documents:
            self.load_index()
        return list(set(doc['component'] for doc in self.documents))
