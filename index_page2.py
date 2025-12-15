#!/usr/bin/env python3
"""
Script to index 20 products from page 2 of WordPress into Pinecone.
Run this script to populate the Pinecone index with products.
"""

import asyncio
import requests
from PIL import Image
import io
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import services
from services import get_pinecone_service, get_wordpress_service
from models import get_clip_embedder


async def index_page2_products():
    """Index 20 products from page 2 of WordPress into Pinecone."""
    
    print("\n" + "="*60)
    print("Indexing 20 Products from Page 2 of WordPress")
    print("="*60 + "\n")
    
    try:
        # Initialize services
        wordpress_service = get_wordpress_service()
        pinecone_service = get_pinecone_service()
        clip_embedder = get_clip_embedder()
        
        # Fetch products from page 2 (20 products max)
        print("📥 Fetching 20 products from WordPress page 2...")
        products = []
        page = 2
        per_page = 20
        
        params = {
            'page': page,
            'per_page': per_page
        }
        
        try:
            response = requests.get(
                os.getenv('WORDPRESS_API_URL', 
                         'https://dodgerblue-otter-660921.hostingersite.com/wp-json/wp/v2/products/'),
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            batch = response.json()
            
            # Process each product
            for product in batch:
                processed = wordpress_service._process_product(product)
                if processed and processed.get('image_url'):
                    products.append(processed)
            
            print(f"✓ Found {len(products)} products with images on page 2\n")
            
        except Exception as e:
            print(f"❌ Error fetching products: {e}")
            return
        
        if not products:
            print("❌ No products with images found on page 2")
            return
        
        # Download images and generate embeddings
        print(f"🤖 Processing {len(products)} products with CLIP model...")
        print("-" * 60)
        
        valid_products = []
        embeddings = []
        
        for i, product in enumerate(products, 1):
            try:
                image_url = product.get('image_url')
                product_name = product.get('name', 'Unknown')
                
                print(f"[{i:2d}/{len(products)}] Processing: {product_name[:50]}")
                
                # Download image
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                
                # Load and process image
                image = Image.open(io.BytesIO(response.content))
                
                # Generate CLIP embedding
                embedding = clip_embedder.embed_image(image)
                
                valid_products.append(product)
                embeddings.append(embedding)
                
                print(f"       ✓ Embedded successfully (embedding size: {len(embedding)})")
                
            except Exception as e:
                print(f"       ❌ Error: {str(e)[:60]}")
                continue
        
        print("-" * 60)
        print(f"\n✓ Successfully processed {len(valid_products)}/{len(products)} products\n")
        
        if not valid_products:
            print("❌ No valid products to index")
            return
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Upsert to Pinecone
        print(f"📤 Upserting {len(valid_products)} products to Pinecone...")
        print("-" * 60)
        
        pinecone_service.upsert_products(valid_products, embeddings_array)
        
        print("-" * 60)
        print("✓ Products successfully upserted to Pinecone!\n")
        
        # Get and display index stats
        stats = pinecone_service.get_index_stats()
        
        print("📊 Pinecone Index Statistics:")
        print("-" * 60)
        print(f"Total vectors in index: {stats.get('total_vector_count', 'N/A')}")
        print(f"Index dimension: {stats.get('dimension', 'N/A')}")
        print(f"Index fullness: {stats.get('index_fullness', 'N/A')}")
        print("-" * 60)
        
        print("\n" + "="*60)
        print("✅ Indexing Complete!")
        print("="*60 + "\n")
        
        # Summary
        print("Summary:")
        print(f"  • Page indexed: 2")
        print(f"  • Products indexed: {len(valid_products)}")
        print(f"  • Embedding model: CLIP ViT-B/32")
        print(f"  • Total vectors in Pinecone: {stats.get('total_vector_count', 'N/A')}\n")
        
    except Exception as e:
        print(f"\n❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(index_page2_products())
