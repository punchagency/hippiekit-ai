"""
Redis Cache Service for Product Data
Provides caching layer for barcode lookups to reduce API calls and processing time
"""

import redis.asyncio as redis
import json
import logging
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)


class CacheService:
    """Async Redis cache service for product data"""
    
    def __init__(self):
        # Get Redis connection details from environment variables
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        redis_db = int(os.getenv("REDIS_DB", "0"))
        
        # Connection pool for better performance
        self.redis_client: Optional[redis.Redis] = None
        self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        
        # Cache TTL (7 days for barcode data - products rarely change)
        self.barcode_ttl = 604800  # 7 days in seconds
        self.product_ttl = 86400    # 1 day for product identification results
        self.ingredients_ttl = 604800  # 7 days for ingredient lookups
        
        logger.info(f"Cache service initialized (Redis: {redis_host}:{redis_port})")
    
    async def connect(self):
        """Establish Redis connection"""
        try:
            if not self.redis_client:
                self.redis_client = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=10
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
            self.redis_client = None
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    async def get_barcode(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Get cached barcode lookup result
        
        Args:
            barcode: Product barcode
            
        Returns:
            Cached product data or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"barcode:{barcode}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"Cache HIT for barcode: {barcode}")
                return json.loads(cached_data)
            else:
                logger.info(f"Cache MISS for barcode: {barcode}")
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set_barcode(self, barcode: str, data: Dict[str, Any]) -> bool:
        """
        Cache barcode lookup result
        
        Args:
            barcode: Product barcode
            data: Product data to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"barcode:{barcode}"
            serialized_data = json.dumps(data)
            
            await self.redis_client.setex(
                cache_key,
                self.barcode_ttl,
                serialized_data
            )
            
            logger.info(f"Cached barcode: {barcode} (TTL: {self.barcode_ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
    
    async def invalidate_barcode(self, barcode: str) -> bool:
        """
        Invalidate cached barcode data
        
        Args:
            barcode: Product barcode
            
        Returns:
            True if invalidated, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"barcode:{barcode}"
            deleted = await self.redis_client.delete(cache_key)
            logger.info(f"Invalidated cache for barcode: {barcode}")
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")
            return False
    
    def _make_ingredients_key(self, product_name: str, brand: str, category: str) -> str:
        """Generate cache key for ingredient lookups"""
        # Normalize the key components - use only product name and brand (category can vary between scans)
        normalized = f"{product_name.lower().strip()}:{brand.lower().strip()}"
        return f"ingredients:{normalized}"
    
    async def get_ingredients(self, product_name: str, brand: str, category: str) -> Optional[Dict[str, Any]]:
        """
        Get cached ingredient separation result
        
        Args:
            product_name: Product name
            brand: Brand name
            category: Product category
            
        Returns:
            Cached ingredient data or None if not found
        """
        if not self.redis_client:
            logger.warning(f"Redis not connected - skipping ingredient cache lookup")
            return None
        
        try:
            cache_key = self._make_ingredients_key(product_name, brand, category)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info(f"Cache HIT for ingredients: {brand} {product_name}")
                return json.loads(cached_data)
            else:
                logger.info(f"Cache MISS for ingredients: {brand} {product_name}")
                return None
                
        except Exception as e:
            logger.error(f"Ingredient cache get error: {str(e)}")
            return None
    
    async def set_ingredients(self, product_name: str, brand: str, category: str, data: Dict[str, Any]) -> bool:
        """
        Cache ingredient separation result
        
        Args:
            product_name: Product name
            brand: Brand name  
            category: Product category
            data: Ingredient data to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.redis_client:
            logger.warning(f"Redis not connected - cannot cache ingredients")
            return False
        
        try:
            cache_key = self._make_ingredients_key(product_name, brand, category)
            serialized_data = json.dumps(data)
            
            await self.redis_client.setex(
                cache_key,
                self.ingredients_ttl,
                serialized_data
            )
            
            logger.info(f"Cached ingredients for: {brand} {product_name} (TTL: {self.ingredients_ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Ingredient cache set error: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disabled"}
        
        try:
            info = await self.redis_client.info("stats")
            return {
                "status": "connected",
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Stats error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)


# Global cache service instance
cache_service = CacheService()
