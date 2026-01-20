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
            
            # For short queries, boost scores if query is substring of product name
            query_lower = query.lower()
            is_short_query = len(query.strip()) <= 4
            
            # Process and enrich results
            products = []
            for match in pinecone_results.matches:
                # Extract product info from metadata
                metadata = match.metadata
                product_name = metadata.get("name", "").lower()
                
                # Boost score for substring matches (especially for short queries)
                score = float(match.score)
                if is_short_query and query_lower in product_name:
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
                
                product = {
                    "id": product_id,
                    "score": score,  # Use boosted score
                    "name": metadata.get("name", ""),
                    "slug": metadata.get("slug", ""),
                    "price": metadata.get("price", ""),
                    "image": metadata.get("image", ""),
                    "category": metadata.get("category", ""),
                    "categories": metadata.get("categories", []),
                    "description": metadata.get("description", ""),
                    "sku": metadata.get("sku", ""),
                    "stock_status": metadata.get("stock_status", "instock")
                }
                
                products.append(product)
                
                # Stop if we have enough results
                if len(products) >= limit:
                    break
            
            # Create result object
            result = {
                "query": query,
                "results": products,
                "count": len(products),
                "limit": limit,
                "min_score": min_score,
                "category_filter": category_filter,
                "search_type": "semantic"
            }
            
            # Cache the results
            if use_cache:
                await self._cache_search_result(query, limit, category_filter, result)
            
            logger.info(f"Found {len(products)} products for query: '{query}'")
            
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
                "count": 0,
                "limit": limit,
                "category_filter": category_filter,
                "search_type": "failed",
                "error": str(e)
            }
    
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
