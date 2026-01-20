"""
Search Router
Handles semantic search API endpoints
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
from services.search_service import SearchService
from services import get_pinecone_service, get_wordpress_service
from services.cache_service import cache_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


# Request/Response Models
class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
    min_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score")
    category: Optional[str] = Field(None, description="Filter by category name")
    use_cache: bool = Field(True, description="Use cached results if available")


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[dict]
    count: int
    limit: int
    min_score: float
    category_filter: Optional[str]
    search_type: str


class SuggestResponse(BaseModel):
    """Autocomplete suggestions response"""
    query: str
    suggestions: List[str]
    count: int


# Initialize search service
search_service = None


def get_search_service() -> SearchService:
    """Get or create search service instance"""
    global search_service
    
    if search_service is None:
        pinecone_service = get_pinecone_service()
        wordpress_service = get_wordpress_service()
        search_service = SearchService(
            pinecone_service=pinecone_service,
            cache_service=cache_service,
            wordpress_service=wordpress_service
        )
    
    return search_service


@router.post("/", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Semantic search for products using text embeddings and vector similarity.
    
    **Features:**
    - Finds products similar to the search query (e.g., "glassware" matches "glass", "beaker")
    - Relevance scoring (0-1) based on semantic similarity
    - Optional category filtering
    - Redis caching for fast repeated searches
    - WordPress fallback if vector search fails
    
    **Example:**
    ```json
    {
        "query": "glassware",
        "limit": 20,
        "min_score": 0.5,
        "category": "Laboratory Equipment"
    }
    ```
    """
    try:
        logger.info(f"Search request: query='{request.query}', limit={request.limit}")
        
        service = get_search_service()
        
        result = await service.search(
            query=request.query,
            limit=request.limit,
            min_score=request.min_score,
            category_filter=request.category,
            use_cache=request.use_cache
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/", response_model=SearchResponse)
async def search_products_get(
    q: str = Query(..., description="Search query text", min_length=1),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    min_score: float = Query(0.5, ge=0.0, le=1.0, description="Minimum score"),
    category: Optional[str] = Query(None, description="Category filter"),
    cache: bool = Query(True, description="Use cache")
):
    """
    GET endpoint for search (alternative to POST for simple queries).
    
    **Example:** `/api/search?q=glassware&limit=10&category=Laboratory`
    """
    request = SearchRequest(
        query=q,
        limit=limit,
        min_score=min_score,
        category=category,
        use_cache=cache
    )
    
    return await search_products(request)


@router.get("/suggest", response_model=SuggestResponse)
async def search_suggestions(
    q: str = Query(..., description="Partial search query", min_length=2),
    limit: int = Query(5, ge=1, le=10, description="Maximum suggestions")
):
    """
    Get autocomplete suggestions based on partial query.
    
    **Example:** `/api/search/suggest?q=gla&limit=5`
    
    Returns product names that match the partial query.
    """
    try:
        logger.info(f"Suggestion request: query='{q}', limit={limit}")
        
        service = get_search_service()
        suggestions = await service.suggest(q, limit=limit)
        
        return {
            "query": q,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Suggestion endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")


@router.get("/health")
async def search_health():
    """
    Health check for search service.
    Verifies Pinecone, Redis, and text embedder are available.
    """
    try:
        service = get_search_service()
        
        # Check Pinecone
        pinecone_stats = service.pinecone.get_index_stats()
        
        # Check Redis
        redis_available = service.cache.redis_client is not None
        
        # Check text embedder
        embedder_available = service.text_embedder is not None
        
        return {
            "status": "healthy",
            "pinecone": {
                "available": True,
                "total_vectors": pinecone_stats.get("total_vector_count", 0),
                "dimension": pinecone_stats.get("dimension", 512)
            },
            "redis": {
                "available": redis_available
            },
            "embedder": {
                "available": embedder_available,
                "model": service.text_embedder.model if embedder_available else None
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
