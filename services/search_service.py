"""
Semantic Search Service
Combines Pinecone vector search with Redis caching for fast, intelligent product search
"""

import logging
from typing import List, Dict, Any, Optional
from services.text_embedder import get_text_embedder
from services.pinecone_service import PineconeService
from services.cache_service import CacheService
from services.wordpress_service import WordPressService
import json

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for semantic product search using text embeddings and vector similarity.
    """
    
    def __init__(
        self,
        pinecone_service: PineconeService,
        cache_service: CacheService,
        wordpress_service: WordPressService
    ):
        """
        Initialize search service.
        
        Args:
            pinecone_service: Pinecone vector database service
            cache_service: Redis cache service
            wordpress_service: WordPress API service
        """
        self.pinecone = pinecone_service
        self.cache = cache_service
        self.wordpress = wordpress_service
        self.text_embedder = get_text_embedder()
        self.cached_categories = self._load_cached_categories()
        
        logger.info("Search service initialized")
    
    async def search(
        self,
        query: str,
        limit: int = 20,
        min_score: float = 0.2,
        category_filter: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Perform semantic search for products.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            category_filter: Optional category name to filter by
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with results, metadata, and performance info
        """
        try:
            # Check cache first
            if use_cache:
                cached_result = await self._get_cached_search(query, limit, category_filter)
                if cached_result:
                    logger.info(f"Cache HIT for search query: '{query}'")
                    return cached_result
            
            logger.info(f"Performing semantic search for: '{query}' (limit: {limit}, min_score: {min_score})")
            
            # Generate embedding for search query
            query_embedding = self.text_embedder.embed_text(query)
            
            # Build filter for Pinecone query
            pinecone_filter = {}
            if category_filter:
                pinecone_filter["category"] = category_filter
            
            # Query Pinecone for similar products
            pinecone_results = self.pinecone.index.query(
                vector=query_embedding,
                top_k=limit * 2,  # Get more results to filter
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None
            )
            
            # For substring matching - boost any query that appears in product name
            query_lower = query.lower()
            
            # Process and enrich results
            products = []
            for match in pinecone_results.matches:
                # Extract product info from metadata
                metadata = match.metadata
                product_name = metadata.get("name", "").lower()
                
                # Boost score for substring matches in product name
                score = float(match.score)
                if query_lower in product_name:
                    # Boost the score significantly for substring matches
                    score = max(score, 0.7)
                
                # Filter by minimum score (after potential boost)
                if score < min_score:
                    continue
                
                # Validate product ID exists
                product_id = metadata.get("id", "")
                if not product_id:
                    logger.warning(f"Skipping match with no product ID: {match.id}")
                    continue
                
                # DEBUG: Log categories for first product
                if len(products) == 0:
                    logger.info(f"DEBUG: First product '{metadata.get('name')}' categories from Pinecone: {metadata.get('categories', [])} (type: {type(metadata.get('categories', []))})")
                
                categories_data = metadata.get("categories", [])
                
                product = {
                    "id": product_id,
                    "score": score,  # Use boosted score
                    "name": metadata.get("name", ""),
                    "slug": metadata.get("slug", ""),
                    "price": metadata.get("price", ""),
                    "image": metadata.get("image", ""),
                    "category": metadata.get("category", ""),
                    "categories": categories_data,
                    "description": metadata.get("description", ""),
                    "sku": metadata.get("sku", ""),
                    "stock_status": metadata.get("stock_status", "instock")
                }
                
                # DEBUG: Log product categories
                if len(products) < 2:  # Log first 2 products
                    logger.info(f"DEBUG: Product '{product['name']}' has categories: {product['categories']}")
                
                products.append(product)
                
                # Stop if we have enough results
                if len(products) >= limit:
                    break
            
            # Find matching categories
            matching_categories = await self._find_matching_categories(query, products)
            
            # DEBUG: Log what's being returned
            if matching_categories:
                logger.info(f"DEBUG: Returning {len(matching_categories)} categories to frontend:")
                for cat in matching_categories:
                    logger.info(f"  - {cat.get('name')} (id: {cat.get('id')}, slug: {cat.get('slug')}, image: '{cat.get('image')}', count: {cat.get('product_count')})")
            
            # Create result object
            result = {
                "query": query,
                "results": products,
                "matching_categories": matching_categories,
                "count": len(products),
                "limit": limit,
                "min_score": min_score,
                "category_filter": category_filter,
                "search_type": "semantic"
            }
            
            # Cache the results
            if use_cache:
                await self._cache_search_result(query, limit, category_filter, result)
            
            logger.info(f"Found {len(products)} products and {len(matching_categories)} categories for query: '{query}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            
            # Fallback to WordPress search if Pinecone fails
            return await self._fallback_wordpress_search(query, limit, category_filter)
    
    async def _get_cached_search(
        self,
        query: str,
        limit: int,
        category_filter: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get cached search results"""
        if not self.cache.redis_client:
            return None
            
        cache_key = self._generate_cache_key(query, limit, category_filter)
        
        try:
            cached_data = await self.cache.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {str(e)}")
        
        return None
    
    async def _cache_search_result(
        self,
        query: str,
        limit: int,
        category_filter: Optional[str],
        result: Dict[str, Any]
    ):
        """Cache search results"""
        if not self.cache.redis_client:
            return
            
        cache_key = self._generate_cache_key(query, limit, category_filter)
        
        try:
            # Cache for 5 minutes (300 seconds)
            ttl = 300
            await self.cache.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )
            logger.debug(f"Cached search results for: '{query}'")
        except Exception as e:
            logger.warning(f"Cache storage error: {str(e)}")
    
    def _generate_cache_key(
        self,
        query: str,
        limit: int,
        category_filter: Optional[str]
    ) -> str:
        """Generate cache key for search query"""
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        category_part = f":{category_filter}" if category_filter else ""
        
        return f"search:{normalized_query}:{limit}{category_part}"
    
    async def _fallback_wordpress_search(
        self,
        query: str,
        limit: int,
        category_filter: Optional[str]
    ) -> Dict[str, Any]:
        """
        Fallback to WordPress native search if Pinecone fails.
        """
        logger.warning(f"Using WordPress fallback search for: '{query}'")
        
        try:
            # Use WordPress service to search
            params = {
                "search": query,
                "per_page": limit
            }
            
            if category_filter:
                params["category"] = category_filter
            
            products = await self.wordpress.fetch_products(params)
            
            # Format results to match semantic search structure
            formatted_products = [
                {
                    "id": str(product.get("id")),
                    "score": 0.7,  # Default score for WordPress results
                    "name": product.get("name", ""),
                    "slug": product.get("slug", ""),
                    "price": product.get("price", ""),
                    "image": product.get("images", [{}])[0].get("src", "") if product.get("images") else "",
                    "category": category_filter or "",
                    "categories": [cat.get("name", "") for cat in product.get("categories", [])],
                    "description": product.get("description", ""),
                    "sku": product.get("sku", ""),
                    "stock_status": product.get("stock_status", "instock")
                }
                for product in products
            ]
            
            return {
                "query": query,
                "results": formatted_products,
                "matching_categories": [],
                "count": len(formatted_products),
                "limit": limit,
                "category_filter": category_filter,
                "search_type": "wordpress_fallback"
            }
            
        except Exception as e:
            logger.error(f"WordPress fallback search failed: {str(e)}")
            
            # Return empty results
            return {
                "query": query,
                "results": [],
                "matching_categories": [],
                "count": 0,
                "limit": limit,
                "category_filter": category_filter,
                "search_type": "failed",
                "error": str(e)
            }
    
    def _load_cached_categories(self) -> List[Dict[str, Any]]:
        """
        Load cached categories from JSON file.
        
        Returns:
            List of category dictionaries with id, name, slug, image, parent, count
        """
        try:
            import os
            cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'categories_cache.json')
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    categories = json.load(f)
                    logger.info(f"Loaded {len(categories)} categories from cache")
                    return categories
            else:
                logger.warning(f"Categories cache file not found: {cache_file}. Run 'python index-categories.py' to create it.")
                return []
        except Exception as e:
            logger.error(f"Error loading cached categories: {str(e)}")
            return []
    
    async def _find_matching_categories(
        self,
        query: str,
        matched_products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find categories that match the search query.
        Uses cached category data and counts matched products per category.
        
        Args:
            query: Search query
            matched_products: List of products that matched the search
            
        Returns:
            List of matching categories with id, name, slug, image, and product count
        """
        try:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            # Get category NAMES from matched products (they're stored as strings, not dicts)
            product_category_names = set()
            for product in matched_products:
                categories = product.get("categories", [])
                for cat in categories:
                    # Categories are stored as strings in Pinecone metadata
                    if isinstance(cat, str):
                        product_category_names.add(cat.lower())
                    elif isinstance(cat, dict):
                        # Fallback if somehow they're dicts
                        cat_name = cat.get("name", "")
                        if cat_name:
                            product_category_names.add(cat_name.lower())
            
            logger.info(f"Product categories from {len(matched_products)} products: {product_category_names}")
            logger.info(f"Total cached categories: {len(self.cached_categories)}")
            
            # Find matching categories from cache
            matching_categories = []
            
            for category in self.cached_categories:
                cat_name = category.get("name", "")
                cat_name_lower = cat_name.lower()
                
                # Skip if category not in matched products
                if cat_name_lower not in product_category_names:
                    continue
                
                # Match if any query word is in category name OR category name is in query
                match_found = False
                for word in query_words:
                    if len(word) >= 3 and word in cat_name_lower:
                        match_found = True
                        break
                
                # Also match if category name is in the query
                if not match_found and cat_name_lower in query_lower:
                    match_found = True
                
                if match_found:
                    matching_categories.append({
                        "id": category.get("id"),
                        "name": category.get("name"),
                        "slug": category.get("slug"),
                        "image": category.get("image"),
                        "product_count": category.get("count", 0)  # Total products in category
                    })
                    logger.info(f"Matched category: {cat_name} (slug: {category.get('slug')})")
            
            # Sort by product count (categories with more products first)
            matching_categories.sort(key=lambda x: x["product_count"], reverse=True)
            
            # Limit to top 5
            return matching_categories[:5]
            
        except Exception as e:
            logger.error(f"Error finding matching categories: {str(e)}")
            return []
    
    async def suggest(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search text
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested search terms
        """
        # This could be enhanced with a dedicated suggestions index
        # For now, perform a quick search and return product names
        
        if len(partial_query) < 2:
            return []
        
        try:
            results = await self.search(partial_query, limit=limit, min_score=0.6)
            suggestions = [product["name"] for product in results["results"]]
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Suggestion error: {str(e)}")
            return []
