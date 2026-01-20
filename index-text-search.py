"""
Script to index ALL WordPress products with TEXT EMBEDDINGS into Pinecone.
This creates semantic search capabilities for product names and descriptions.

Run with: python index-text-search.py
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from services.text_embedder import get_text_embedder
from services import get_pinecone_service, get_wordpress_service


def index_text_embeddings():
    """Index all WordPress products with text embeddings for semantic search."""
    try:
        print("\n" + "="*70)
        print("🔍 INDEXING PRODUCTS WITH TEXT EMBEDDINGS FOR SEMANTIC SEARCH")
        print("="*70 + "\n")
        
        # Step 1: Fetch all products from WordPress
        print("📥 Step 1: Fetching products from WordPress...")
        wordpress_service = get_wordpress_service()
        products = wordpress_service.fetch_products(max_products=None)
        
        if not products:
            print("❌ No products found in WordPress!")
            return
        
        print(f"✅ Fetched {len(products)} products from WordPress\n")
        
        # Step 2: Prepare products and generate text embeddings
        print("📝 Step 2: Generating text embeddings with OpenAI...")
        print("   (This may take a few minutes for 630 products)\n")
        
        text_embedder = get_text_embedder()
        
        # Prepare product texts for batch embedding
        product_texts = []
        valid_products = []
        
        for i, product in enumerate(products, 1):
            try:
                product_name = product.get('name', 'Unknown')
                
                # Create searchable text
                product_text = text_embedder.create_product_text(product)
                
                if not product_text.strip():
                    print(f"⚠️  [{i}/{len(products)}] Skipping '{product_name}' - no text content")
                    continue
                
                product_texts.append(product_text)
                valid_products.append(product)
                
                if i % 50 == 0:
                    print(f"   📊 Prepared {i}/{len(products)} products...")
                
            except Exception as e:
                print(f"❌ [{i}/{len(products)}] Error preparing '{product_name}': {str(e)}")
                continue
        
        print(f"\n✅ Prepared {len(valid_products)} products for embedding")
        
        if not valid_products:
            print("❌ No valid products to index!")
            return
        
        # Step 3: Generate embeddings in batches (OpenAI limit: 2048 texts per request)
        print("\n🧠 Step 3: Generating OpenAI embeddings in batches...")
        
        batch_size = 100  # Process 100 products at a time
        all_embeddings = []
        
        for i in range(0, len(product_texts), batch_size):
            batch_texts = product_texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(product_texts) + batch_size - 1) // batch_size
            
            print(f"   ⏳ Batch {batch_num}/{total_batches}: Embedding {len(batch_texts)} products...")
            
            try:
                batch_embeddings = text_embedder.embed_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)
                print(f"   ✅ Batch {batch_num}/{total_batches} complete")
            except Exception as e:
                print(f"   ❌ Batch {batch_num} failed: {str(e)}")
                # Add zero vectors for failed batch
                all_embeddings.extend([[0.0] * text_embedder.dimension] * len(batch_texts))
        
        print(f"\n✅ Generated {len(all_embeddings)} text embeddings\n")
        
        # Step 4: Upsert to Pinecone with TEXT metadata
        print("📤 Step 4: Upserting text embeddings to Pinecone...")
        
        pinecone_service = get_pinecone_service()
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        
        for product, embedding in zip(valid_products, all_embeddings):
            # Create metadata with all searchable fields
            metadata = {
                "id": str(product.get('id', '')),
                "name": product.get('name', ''),
                "slug": product.get('slug', ''),
                "description": product.get('description', '')[:500],  # Limit to 500 chars
                "short_description": product.get('short_description', '')[:200],
                "sku": product.get('sku', ''),
                "price": str(product.get('price', '')),
                "stock_status": product.get('stock_status', 'instock'),
                "image": product.get('image_url', ''),
                "categories": product.get('categories', []),
                "category": product.get('categories', [''])[0] if product.get('categories') else '',
                "search_type": "text"  # Mark as text embedding vs image embedding
            }
            
            # Vector ID format: "text-{product_id}"
            vector_id = f"text-{product.get('id')}"
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert in batches (Pinecone limit: 100 vectors per request)
        upsert_batch_size = 100
        
        for i in range(0, len(vectors_to_upsert), upsert_batch_size):
            batch = vectors_to_upsert[i:i + upsert_batch_size]
            batch_num = (i // upsert_batch_size) + 1
            total_batches = (len(vectors_to_upsert) + upsert_batch_size - 1) // upsert_batch_size
            
            print(f"   ⏳ Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)...")
            
            try:
                pinecone_service.index.upsert(vectors=batch)
                print(f"   ✅ Batch {batch_num}/{total_batches} upserted successfully")
            except Exception as e:
                print(f"   ❌ Batch {batch_num} failed: {str(e)}")
        
        print(f"\n✅ Successfully upserted {len(vectors_to_upsert)} text embeddings\n")
        
        # Step 5: Get index statistics
        print("📊 Step 5: Fetching Pinecone index statistics...")
        stats = pinecone_service.get_index_stats()
        
        print("\n" + "="*70)
        print("✨ TEXT EMBEDDING INDEXING COMPLETE!")
        print("="*70)
        print(f"Total products fetched:     {len(products)}")
        print(f"Successfully indexed:       {len(valid_products)}")
        print(f"Skipped/Failed:             {len(products) - len(valid_products)}")
        print(f"\nPinecone Index Stats:")
        print(f"  Total vectors:            {stats.get('total_vector_count', 'N/A')}")
        print(f"  Dimension:                {stats.get('dimension', 'N/A')}")
        print(f"  Embedding model:          {text_embedder.model}")
        print("="*70 + "\n")
        
        print("🎉 Semantic search is now ready!")
        print("   Try searching: 'glassware', 'chemicals', 'protective equipment'\n")
        
        return {
            "total_products": len(products),
            "indexed": len(valid_products),
            "failed": len(products) - len(valid_products),
            "index_stats": stats
        }
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n⚡ Starting text embedding indexing process...\n")
    index_text_embeddings()
    print("✅ Done!\n")
