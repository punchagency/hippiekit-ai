import requests
from typing import List, Dict, Any, Optional
import os
import time
import html
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class WordPressService:
    """
    Service for fetching product data from WordPress REST API.
    """

    def __init__(self, api_url: str = None):
        """
        Initialize WordPress service.

        Args:
            api_url: WordPress REST API URL for products
        """
        self.api_url = api_url or os.getenv(
            'WORDPRESS_API_URL',
            'https://dodgerblue-otter-660921.hostingersite.com/wp-json/wp/v2/products/'
        )
        # Get base URL for media requests
        self.base_url = self.api_url.split('/wp-json')[0]
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,  # Total retries
            backoff_factor=1,  # Wait 1, 2, 4, 8, 16 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Cache for product categories (ID -> name mapping)
        self.categories_cache = {}
        self._fetch_all_categories()

    def _fetch_all_categories(self):
        """
        Fetch all product categories from WordPress API and cache them.
        This is called once during initialization to avoid repeated API calls.
        """
        try:
            categories_url = f"{self.base_url}/wp-json/wp/v2/product-categories"
            print(f"Fetching product categories...")
            
            page = 1
            per_page = 100
            
            while True:
                response = self.session.get(
                    categories_url,
                    params={'page': page, 'per_page': per_page},
                    timeout=30
                )
                response.raise_for_status()
                batch = response.json()
                
                if not batch:
                    break
                
                # Add categories to cache
                for category in batch:
                    self.categories_cache[category['id']] = category['name']
                
                # Check if there are more pages
                total_pages = int(response.headers.get('X-WP-TotalPages', page))
                if page >= total_pages:
                    break
                
                page += 1
            
            print(f"✅ Cached {len(self.categories_cache)} product categories")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch categories: {e}")
            print(f"   Continuing without category data...")

    def fetch_products(self, max_products: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch products from WordPress API.

        Args:
            max_products: Maximum number of products to fetch (None for all)

        Returns:
            List of product dictionaries
        """
        products = []
        page = 1
        per_page = 100  # WordPress API max per page

        print(f"Fetching products from WordPress API...")
        print(f"API URL: {self.api_url}")
        
        # Test connection first
        try:
            print("Testing connection to WordPress...")
            test_response = self.session.get(self.api_url, params={'page': 1, 'per_page': 1}, timeout=30)
            test_response.raise_for_status()
            print(f"✅ Connection successful! WordPress is accessible.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to WordPress API: {e}")
            print(f"   URL: {self.api_url}")
            print(f"\n💡 Troubleshooting:")
            print(f"   1. Check if the WordPress site is online")
            print(f"   2. Verify the URL is correct in .env file (WORDPRESS_API_URL)")
            print(f"   3. Check your internet connection")
            print(f"   4. Try accessing {self.base_url} in your browser")
            return []

        while True:
            # Build request URL
            params = {
                'page': page,
                'per_page': per_page
            }

            try:
                response = self.session.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()

                batch = response.json()

                # No more products
                if not batch:
                    break

                # Process products in this batch
                for product in batch:
                    processed_product = self._process_product(product)

                    # Only include products with images
                    if processed_product and processed_product.get('image_url'):
                        products.append(processed_product)

                        # Check if we've reached max_products
                        if max_products and len(products) >= max_products:
                            print(f"Reached max_products limit: {max_products}")
                            return products

                print(f"Fetched page {page}: {len(batch)} products (total with images: {len(products)})")

                # Check if there are more pages
                total_pages = int(response.headers.get('X-WP-TotalPages', page))
                if page >= total_pages:
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                print(f"Error fetching products on page {page}: {e}")
                break

        print(f"Total products fetched: {len(products)}")
        return products

    def _get_media_url(self, media_id: int) -> Optional[str]:
        """
        Fetch the media URL for a given media ID.

        Args:
            media_id: WordPress media ID

        Returns:
            Media URL or None
        """
        if not media_id:
            return None

        try:
            media_url = f'{self.base_url}/wp-json/wp/v2/media/{media_id}'
            response = requests.get(media_url, timeout=10)
            response.raise_for_status()
            media = response.json()
            return media.get('source_url')
        except Exception as e:
            print(f"Error fetching media {media_id}: {e}")
            return None

    def _process_product(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a raw product from WordPress API.

        Args:
            product: Raw product data from API

        Returns:
            Processed product dictionary or None if invalid
        """
        try:
            # Extract featured image
            image_url = None
            
            # Try featured_media_url first
            if 'featured_media_url' in product:
                image_url = product['featured_media_url']
            # Try better_featured_image
            elif 'better_featured_image' in product:
                image_url = product['better_featured_image'].get('source_url')
            # Try yoast og_image
            elif 'yoast_head_json' in product and 'og_image' in product['yoast_head_json']:
                og_images = product['yoast_head_json']['og_image']
                if og_images and len(og_images) > 0:
                    image_url = og_images[0].get('url')
            # Try to fetch from featured_media ID
            elif product.get('featured_media'):
                image_url = self._get_media_url(product['featured_media'])

            # Extract title
            title = product.get('title', {})
            if isinstance(title, dict):
                title = title.get('rendered', '')
            # Decode HTML entities in title
            title = html.unescape(title)

            # Extract FULL description from content (not excerpt)
            content = product.get('content', {})
            if isinstance(content, dict):
                description_html = content.get('rendered', '')
            else:
                description_html = ''

            # Clean HTML using BeautifulSoup and decode entities
            if description_html:
                soup = BeautifulSoup(description_html, 'html.parser')
                description = soup.get_text(separator=' ', strip=True)
                # Decode HTML entities (like &#038; -> &)
                description = html.unescape(description)
            else:
                description = ''
            
            # Trim description to 100-200 words for embedding
            words = description.split()
            if len(words) > 200:
                description = ' '.join(words[:200]) + '...'
            
            # Extract category names from category IDs
            category_ids = product.get('product-categories', [])
            categories = [
                self.categories_cache.get(cat_id, f"Category-{cat_id}")
                for cat_id in category_ids
            ]

            # Extract affiliate URL from meta
            meta = product.get('meta', {})
            affiliate_url = meta.get('cta_button_url', '')

            # Get price (if available)
            price = product.get('price', product.get('regular_price', ''))

            return {
                'id': product.get('id'),
                'name': title,
                'price': price,
                'image_url': image_url,
                'permalink': product.get('link', ''),
                'description': description,
                'categories': categories,
                'affiliate_url': affiliate_url
            }

        except Exception as e:
            print(f"Error processing product {product.get('id')}: {e}")
            return None


# Global instance
_wordpress_service_instance = None

def get_wordpress_service() -> WordPressService:
    """Get or create the global WordPress service instance."""
    global _wordpress_service_instance

    if _wordpress_service_instance is None:
        _wordpress_service_instance = WordPressService()

    return _wordpress_service_instance
