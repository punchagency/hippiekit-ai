from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import requests
from PIL import Image
import io

from models import get_clip_embedder
from services import get_pinecone_service, get_wordpress_service

router = APIRouter()

@router.post("/index/products")
async def index_products(max_products: Optional[int] = Query(None, description="Maximum number of products to index")) -> Dict[str, Any]:
    """
    Index products from WordPress into Pinecone vector database.
    
    Args:
        max_products: Maximum number of products to index (None for all)
        
    Returns:
        Indexing status and statistics
    """
    try:
        # Fetch products from WordPress
        print("Fetching products from WordPress...")
        wordpress_service = get_wordpress_service()
        products = wordpress_service.fetch_products(max_products=max_products)
        
        if not products:
            return {
                'success': False,
                'message': 'No products found to index',
                'indexed_count': 0
            }
        
        # Download and embed product images
        print(f"Processing {len(products)} products...")
        valid_products = []
        embeddings = []
        
        clip_embedder = get_clip_embedder()
        
        for i, product in enumerate(products):
            try:
                # Download image
                image_url = product.get('image_url')
                if not image_url:
                    print(f"Skipping product {product.get('id')} - no image URL")
                    continue
                
                print(f"Processing product {i+1}/{len(products)}: {product.get('name')}")
                
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                
                # Load image
                image = Image.open(io.BytesIO(response.content))
                
                # Generate embedding
                embedding = clip_embedder.embed_image(image)
                
                valid_products.append(product)
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error processing product {product.get('id')}: {e}")
                continue
        
        if not valid_products:
            return {
                'success': False,
                'message': 'No valid products with images to index',
                'indexed_count': 0
            }
        
        # Convert embeddings list to numpy array
        import numpy as np
        embeddings_array = np.array(embeddings)
        
        # Upsert to Pinecone
        print(f"Upserting {len(valid_products)} products to Pinecone...")
        pinecone_service = get_pinecone_service()
        pinecone_service.upsert_products(valid_products, embeddings_array)
        
        # Get index stats
        stats = pinecone_service.get_index_stats()
        
        return {
            'success': True,
            'message': f'Successfully indexed {len(valid_products)} products',
            'indexed_count': len(valid_products),
            'skipped_count': len(products) - len(valid_products),
            'index_stats': stats
        }
        
    except Exception as e:
        print(f"Error indexing products: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing products: {str(e)}"
        )

@router.get("/index/stats")
async def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the Pinecone index."""
    try:
        pinecone_service = get_pinecone_service()
        stats = pinecone_service.get_index_stats()
        
        return {
            'success': True,
            'stats': stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting index stats: {str(e)}"
        )
