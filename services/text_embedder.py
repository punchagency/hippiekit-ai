"""
Text Embedding Service using OpenAI
Generates semantic embeddings for product names and descriptions to enable similarity search
"""

from openai import OpenAI
import os
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Service for generating text embeddings using OpenAI's embedding models.
    Used for semantic product search.
    """
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small", dimension: int = 512):
        """
        Initialize the text embedder.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: OpenAI embedding model to use
            dimension: Output dimension (512 to match Pinecone index, or 1536 for full)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.dimension = dimension
        
        logger.info(f"Text embedder initialized with model: {model}, dimension: {dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Clean and prepare text
            cleaned_text = text.strip()
            if not cleaned_text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension
            
            # Generate embedding
            response = self.client.embeddings.create(
                input=cleaned_text,
                model=self.model,
                dimensions=self.dimension
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(cleaned_text)})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch (more efficient).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Clean texts
            cleaned_texts = [text.strip() for text in texts]
            
            # Remove empty strings but track their positions
            non_empty_texts = []
            empty_indices = []
            for i, text in enumerate(cleaned_texts):
                if text:
                    non_empty_texts.append(text)
                else:
                    empty_indices.append(i)
            
            if not non_empty_texts:
                logger.warning("All texts are empty")
                return [[0.0] * self.dimension] * len(texts)
            
            # Generate embeddings for non-empty texts
            response = self.client.embeddings.create(
                input=non_empty_texts,
                model=self.model,
                dimensions=self.dimension
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            # Insert zero vectors for empty strings
            for idx in empty_indices:
                embeddings.insert(idx, [0.0] * self.dimension)
            
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def create_product_text(self, product: dict) -> str:
        """
        Create a searchable text representation of a product.
        Combines name, description, and categories for better semantic matching.
        Repeats product name multiple times to increase its weight in embeddings.
        
        Args:
            product: Product dictionary with name, description, categories
            
        Returns:
            Combined text string optimized for search
        """
        parts = []
        
        # Product name (most important) - repeat 3x for better matching
        if product.get('name'):
            name = product['name']
            parts.append(name)
            parts.append(name)  # Repeat for emphasis
            parts.append(name)  # Repeat again
        
        # Categories (important for context)
        categories = product.get('categories', [])
        if categories:
            category_names = [cat.get('name', '') for cat in categories if isinstance(cat, dict)]
            if category_names:
                parts.append("Categories: " + ", ".join(category_names))
        
        # Description (provides context) - shortened to 200 chars
        description = product.get('description') or product.get('short_description', '')
        if description:
            # Limit description length to avoid token limits and reduce noise
            description_clean = description[:200]
            parts.append(description_clean)
        
        # SKU for exact matching
        if product.get('sku'):
            parts.append(f"SKU: {product['sku']}")
        
        combined_text = " | ".join(parts)
        return combined_text
    
    def embed_product(self, product: dict) -> List[float]:
        """
        Generate embedding for a product.
        
        Args:
            product: Product dictionary
            
        Returns:
            Embedding vector
        """
        product_text = self.create_product_text(product)
        return self.embed_text(product_text)
    
    def embed_products(self, products: List[dict]) -> List[List[float]]:
        """
        Generate embeddings for multiple products in batch.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of embedding vectors
        """
        product_texts = [self.create_product_text(product) for product in products]
        return self.embed_batch(product_texts)


# Singleton instance
_text_embedder_instance = None


def get_text_embedder() -> TextEmbedder:
    """Get or create the singleton text embedder instance"""
    global _text_embedder_instance
    
    if _text_embedder_instance is None:
        dimension = int(os.getenv("TEXT_EMBEDDING_DIM", "512"))
        model = os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-small")
        _text_embedder_instance = TextEmbedder(dimension=dimension, model=model)
    
    return _text_embedder_instance
