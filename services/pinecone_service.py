from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Dict, Any, Optional
import os

class PineconeService:
    """
    Service for managing Pinecone vector database operations.
    Handles indexing and querying product embeddings.
    """
    
    def __init__(self, api_key: str, index_name: str, dimension: int = 512):
        """
        Initialize Pinecone service.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Dimension of vectors (512 for CLIP ViT-B/32)
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        
        # Get or create index
        self._ensure_index_exists()
        
        # Connect to index
        self.index = self.pc.Index(index_name)
        
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"Index {self.index_name} created successfully!")
        else:
            print(f"Using existing index: {self.index_name}")
    
    def upsert_products(self, products: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Insert or update product embeddings in Pinecone.
        
        Args:
            products: List of product dictionaries with id, name, image, etc.
            embeddings: Corresponding embeddings array (products x dimension)
        """
        vectors = []
        
        for i, product in enumerate(products):
            product_id = str(product.get('id'))
            embedding = embeddings[i].tolist()
            
            # Metadata to store with the vector (optimized for search & display)
            metadata = {
                'product_id': product_id,
                'name': product.get('name', ''),
                'price': product.get('price', ''),
                'image_url': product.get('image_url', ''),
                'permalink': product.get('permalink', ''),
                'description': product.get('description', ''),  # Already trimmed to 100-200 words
                'categories': ', '.join(product.get('categories', [])),  # Category names as comma-separated string
                'affiliate_url': product.get('affiliate_url', '')
            }
            
            vectors.append({
                'id': product_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"Upserted batch {i // batch_size + 1} ({len(batch)} products)")
    
    def query_similar_products(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        min_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Query for similar products using an embedding.
        
        Args:
            query_embedding: Query vector (512 dimensions)
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of matching products with scores
        """
        # Convert to list if numpy array
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        products = []
        for match in results.matches:
            # Filter by minimum score
            if match.score >= min_score:
                product = {
                    'id': match.metadata.get('product_id'),
                    'name': match.metadata.get('name'),
                    'price': match.metadata.get('price'),
                    'image_url': match.metadata.get('image_url'),
                    'permalink': match.metadata.get('permalink'),
                    'description': match.metadata.get('description'),
                    'categories': match.metadata.get('categories', ''),
                    'affiliate_url': match.metadata.get('affiliate_url', ''),
                    'similarity_score': float(match.score)
                }
                products.append(product)
        
        return products
    
    def delete_all_vectors(self):
        """Delete all vectors from the index."""
        self.index.delete(delete_all=True)
        print(f"Deleted all vectors from index: {self.index_name}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'index_fullness': stats.index_fullness
        }


# Global instance
_pinecone_service_instance = None

def get_pinecone_service() -> PineconeService:
    """
    Get or create the global Pinecone service instance.
    """
    global _pinecone_service_instance
    
    if _pinecone_service_instance is None:
        api_key = os.getenv('PINECONE_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'hippiekit-products')
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        _pinecone_service_instance = PineconeService(
            api_key=api_key,
            index_name=index_name,
            dimension=512
        )
    
    return _pinecone_service_instance
