"""
Script to fetch and cache all product categories from WordPress.
This creates a JSON file with category data for fast searching.

Run with: python index-categories.py
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests


def fetch_all_categories():
    """Fetch all product categories from WordPress API."""
    try:
        base_url = os.getenv("WORDPRESS_API_URL", "https://dodgerblue-otter-660921.hostingersite.com")
        # Remove any trailing slashes and /wp-json prefix from base_url if present
        base_url = base_url.rstrip('/')
        if base_url.endswith('/wp-json/wp/v2/products'):
            base_url = base_url.replace('/wp-json/wp/v2/products', '')
        
        categories_url = f"{base_url}/wp-json/wp/v2/product-categories"
        
        print("\n" + "="*70)
        print("📁 INDEXING PRODUCT CATEGORIES FROM WORDPRESS")
        print("="*70 + "\n")
        
        print(f"📥 Fetching categories from: {categories_url}")
        
        all_categories = []
        page = 1
        per_page = 100
        
        while True:
            response = requests.get(
                categories_url,
                params={'page': page, 'per_page': per_page},
                timeout=30
            )
            response.raise_for_status()
            batch = response.json()
            
            if not batch:
                break
            
            # Extract relevant fields and resolve image URLs
            for idx, category in enumerate(batch, 1):
                featured_image_id = category.get("meta", {}).get("featured_image", "")
                image_url = ""
                
                # Fetch actual image URL if we have an ID
                if featured_image_id and str(featured_image_id).strip():
                    try:
                        media_url = f"{base_url}/wp-json/wp/v2/media/{featured_image_id}"
                        media_response = requests.get(media_url, timeout=10)
                        if media_response.status_code == 200:
                            media_data = media_response.json()
                            # Try different possible image URL fields
                            image_url = (
                                media_data.get("source_url") or 
                                media_data.get("guid", {}).get("rendered", "") or
                                media_data.get("media_details", {}).get("sizes", {}).get("medium", {}).get("source_url", "")
                            )
                            if image_url:
                                print(f"   ✅ Fetched image for '{category.get('name')}': {image_url[:60]}...")
                        else:
                            print(f"   ⚠️  Media ID {featured_image_id} not found for '{category.get('name')}' (HTTP {media_response.status_code})")
                    except Exception as e:
                        print(f"   ⚠️  Could not fetch image for '{category.get('name')}': {str(e)}")
                else:
                    if idx <= 5:  # Only show for first 5 in batch
                        print(f"   ℹ️  No featured_image ID for '{category.get('name')}'")
                
                category_data = {
                    "id": category.get("id"),
                    "name": category.get("name", ""),
                    "slug": category.get("slug", ""),
                    "image": image_url,
                    "parent": category.get("parent", 0),
                    "count": category.get("count", 0)
                }
                all_categories.append(category_data)
            
            # Check if there are more pages
            total_pages = int(response.headers.get('X-WP-TotalPages', page))
            print(f"   📄 Fetched page {page}/{total_pages} ({len(batch)} categories)")
            
            if page >= total_pages:
                break
            
            page += 1
        
        print(f"\n✅ Fetched {len(all_categories)} total categories")
        
        # Save to JSON file
        output_file = "categories_cache.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_categories, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved categories to: {output_file}")
        
        # Print sample categories
        print("\n📋 Sample categories:")
        for cat in all_categories[:5]:
            print(f"   - {cat['name']} (slug: {cat['slug']}, products: {cat['count']})")
        
        print("\n" + "="*70)
        print("✅ CATEGORY INDEXING COMPLETE!")
        print("="*70 + "\n")
        
        return all_categories
        
    except Exception as e:
        print(f"❌ Error fetching categories: {str(e)}")
        return []


if __name__ == "__main__":
    fetch_all_categories()
