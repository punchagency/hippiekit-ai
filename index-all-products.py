"""
Script to index ALL WordPress products into Pinecone vector database.
This will fetch all ~350 products and create CLIP embeddings for them.

Run with: python index-all-products.py
"""

import os
import sys
import requests
from PIL import Image
import io
import numpy as np
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from models import get_clip_embedder
from services import get_pinecone_service, get_wordpress_service


def index_all_products():
    """Index all WordPress products into Pinecone."""
    try:
        print("\n" + "="*60)
        print("🚀 INDEXING ALL WORDPRESS PRODUCTS TO PINECONE")
        print("="*60 + "\n")
        
        # Step 1: Fetch all products from WordPress
        print("📥 Step 1: Fetching products from WordPress...")
        wordpress_service = get_wordpress_service()
        products = wordpress_service.fetch_products(max_products=None)  # None = all products
        
        if not products:
            print("❌ No products found in WordPress!")
            return
        
        print(f"✅ Fetched {len(products)} products from WordPress\n")
        
        # Step 2: Process images and generate embeddings
        print("🖼️  Step 2: Processing images and generating CLIP embeddings...")
        valid_products = []
        embeddings = []
        
        clip_embedder = get_clip_embedder()
        
        for i, product in enumerate(products, 1):
            try:
                image_url = product.get('image_url')
                product_name = product.get('name', 'Unknown')
                product_id = product.get('id')
                
                if not image_url:
                    print(f"⚠️  [{i}/{len(products)}] Skipping '{product_name}' - no image URL")
                    continue
                
                print(f"⏳ [{i}/{len(products)}] Processing: {product_name[:50]}...")
                
                # Download image
                response = requests.get(image_url, timeout=15)
                response.raise_for_status()
                
                # Load and convert image
                image = Image.open(io.BytesIO(response.content))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Generate CLIP embedding (512-dim vector)
                embedding = clip_embedder.embed_image(image)
                
                valid_products.append(product)
                embeddings.append(embedding)
                
                # Progress indicator every 10 products
                if i % 10 == 0:
                    print(f"   📊 Progress: {len(valid_products)}/{i} products processed successfully")
                
            except Exception as e:
                print(f"❌ [{i}/{len(products)}] Error processing '{product_name}': {str(e)}")
                continue
        
        print(f"\n✅ Successfully processed {len(valid_products)} products")
        print(f"⚠️  Skipped {len(products) - len(valid_products)} products (no image or error)\n")
        
        if not valid_products:
            print("❌ No valid products to index!")
            return
        
        # Step 3: Convert to numpy array
        print("🔢 Step 3: Converting embeddings to numpy array...")
        embeddings_array = np.array(embeddings)
        print(f"✅ Embeddings shape: {embeddings_array.shape}\n")
        
        # Step 4: Upsert to Pinecone
        print("📤 Step 4: Upserting products to Pinecone...")
        pinecone_service = get_pinecone_service()
        pinecone_service.upsert_products(valid_products, embeddings_array)
        print(f"✅ Successfully upserted {len(valid_products)} products to Pinecone\n")
        
        # Step 5: Get index statistics
        print("📊 Step 5: Fetching index statistics...")
        stats = pinecone_service.get_index_stats()
        
        print("\n" + "="*60)
        print("✨ INDEXING COMPLETE!")
        print("="*60)
        print(f"Total products fetched: {len(products)}")
        print(f"Successfully indexed:   {len(valid_products)}")
        print(f"Skipped/Failed:         {len(products) - len(valid_products)}")
        print(f"\nPinecone Index Stats:")
        print(f"  Total vectors:        {stats.get('total_vector_count', 'N/A')}")
        print(f"  Dimension:            {stats.get('dimension', 'N/A')}")
        print("="*60 + "\n")
        
        print("🎉 You can now use the barcode recommendation system!")
        print("   WordPress products will be suggested as alternatives.\n")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n⚡ Starting product indexing process...\n")
    index_all_products()
    print("✅ Done!\n")
