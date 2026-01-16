"""
Web Search Service for Product Ingredient Lookup
Uses OpenAI's native web search to find product ingredients
"""

import os
import logging
import time
from typing import Optional, Dict, Any
from urllib.parse import quote
import httpx

logger = logging.getLogger(__name__)


class WebSearchService:
    """Service for searching the web for product ingredients using OpenAI's native web search"""
    
    def __init__(self):
        self.timeout = 60  # Increased timeout for web search (can take longer)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Reuse a single httpx client for connection pooling (avoid parallel connection timeouts)
        self._client = None

        # if not self.openai_api_key:
        #     raise RuntimeError("OPENAI_API_KEY is not set")
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared httpx client for connection pooling"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=30.0),  # Increased connect timeout for parallel requests
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self._client
        
    async def search_product_ingredients(
        self, 
        product_name: str, 
        brand: str,
        category: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search for product ingredients using OpenAI's native web search.
        Uses the Responses API via raw HTTP (Python SDK doesn't support it yet).
        
        Args:
            product_name: Name of the product
            brand: Brand name
            category: Optional product category for context
            
        Returns:
            {
                'ingredients': str (comma-separated ingredient list),
                'source': 'openai_web_search',
                'confidence': 'high' | 'medium' | 'low',
                'sources': list of source URLs,
                'note': str
            }
        """
        start_time = time.time()
        logger.info(f"[OPENAI WEB SEARCH] Searching for ingredients: {brand} {product_name}")
        
        # Build search query - improved prompt for better web search results
        query = f"""Find the official ingredient list for the product below.

Brand: {brand}
Product: {product_name}
{f"Category: {category}" if category else ""}

Instructions:
- Search the manufacturer website or trusted retailers
- Extract the FULL ingredient list in order
- Respond with a comma-separated list ONLY
"""
        
        logger.info(f"[OPENAI WEB SEARCH] Query: {query}")
        
        try:
            # Use shared client for connection pooling (prevents parallel request timeouts)
            client = self._get_client()
            
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4.1-mini",
                    "input": query,
                    "tools": [{"type": "web_search"}],
                },
            )
            response.raise_for_status()
            data = response.json()
            
            # Debug: Log the full response structure to understand how sources are provided
            logger.debug(f"[OPENAI WEB SEARCH] Full response structure: {data}")
            
            # Extract output text from nested structure (tool responses nest differently)
            output_text = None
            sources = []
            
            for item in data.get("output", []):
                # Extract text and sources from message content
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            output_text = content.get("text")
                            
                            # Extract sources from annotations (url_citation type)
                            for annotation in content.get("annotations", []):
                                if annotation.get("type") == "url_citation":
                                    url = annotation.get("url")
                                    if url:
                                        sources.append(url)
            
            # Deduplicate sources
            sources = list(dict.fromkeys(sources))
            
            logger.debug(f"[OPENAI WEB SEARCH] Extracted sources: {sources}")
            
            if not output_text or len(output_text.strip()) < 20:
                logger.warning(f"[OPENAI WEB SEARCH] No ingredients found for {brand} {product_name}")
                logger.debug(f"[OPENAI WEB SEARCH] Raw response: {data}")
                return None
            
            # Check if the response is actually saying "couldn't find" instead of providing ingredients
            negative_indicators = [
                "couldn't locate", "couldn't find", "could not find", "not available",
                "unable to locate", "unable to find", "no ingredient list", 
                "doesn't provide", "does not provide", "not listed", "not provided",
                "I don't have", "I cannot find", "not accessible"
            ]
            
            output_lower = output_text.lower()
            if any(indicator in output_lower for indicator in negative_indicators):
                logger.warning(f"[OPENAI WEB SEARCH] Response indicates no ingredients found: {output_text[:100]}...")
                return None
            
            # Determine confidence based on sources
            confidence = (
                "high" if len(sources) >= 2 else
                "medium" if len(sources) == 1 else
                "low"
            )
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"[OPENAI WEB SEARCH] ✅ Found ingredients in {duration_ms:.0f}ms")
            print(f"   ⏱️  [WEB SEARCH] Ingredient search: {duration_ms:.0f}ms")
            
            return {
                "ingredients": output_text,
                "source": "openai_web_search",
                "confidence": confidence,
                "sources": sources,
                "note": f"Ingredients retrieved via OpenAI web search ({len(sources)} sources). Verify with packaging."
            }
            
        except httpx.ConnectTimeout:
            logger.error(f"[OPENAI WEB SEARCH] Connection timeout - network issue or API unreachable")
            return None
        except httpx.ReadTimeout:
            logger.error(f"[OPENAI WEB SEARCH] Read timeout - web search took too long")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"[OPENAI WEB SEARCH] HTTP error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            logger.exception(f"[OPENAI WEB SEARCH] Search failed: {e}")
            return None
    
    async def fetch_brand_logo(self, brand_name: str, product_type: str = "") -> Optional[str]:
        """
        Fetch brand logo image URL using Brandfetch Logo API CDN.
        
        Args:
            brand_name: The brand name to search for
            product_type: Optional product type context (not used)
            
        Returns:
            Direct CDN URL to the brand logo or None if brand not found
        """
        try:
            logger.info(f"[BRANDFETCH] Searching for logo: {brand_name}")
            
            # Step 1: Search for brand to get the correct domain
            # URL-encode the brand name to handle special characters (e.g., LÄRABAR)
            encoded_brand_name = quote(brand_name, safe='')
            search_url = f"https://api.brandfetch.io/v2/search/{encoded_brand_name}"
            
            async with httpx.AsyncClient(timeout=5.0) as client:  # 5s is best practice for simple lookups
                # Search for brand (no API key needed for search)
                search_response = await client.get(search_url)
                
                if search_response.status_code != 200:
                    logger.warning(f"[BRANDFETCH] Search failed for {brand_name}: {search_response.status_code}")
                    return None
                
                search_results = search_response.json()
                
                if not search_results or len(search_results) == 0:
                    logger.warning(f"[BRANDFETCH] No brands found for {brand_name}")
                    return None
                
                # Step 2: Find the best matching brand
                brand_domain = self._find_best_brand_match(brand_name, search_results)
                
                if not brand_domain:
                    logger.warning(f"[BRANDFETCH] No domain found for {brand_name}")
                    return None
                
                logger.info(f"[BRANDFETCH] Found domain: {brand_domain}")
                
                # Step 3: Construct Logo API CDN URL
                # Format: https://cdn.brandfetch.io/:domain/icon.png
                # Using icon type (square logo) in PNG format for best compatibility
                # Note: For production, get a free Client ID from https://developers.brandfetch.com/register
                # and add it as ?c=YOUR_CLIENT_ID
                
                # Try different logo formats
                logo_urls = [
                    f"https://cdn.brandfetch.io/{brand_domain}/icon.png",  # Square icon PNG
                    f"https://cdn.brandfetch.io/{brand_domain}/logo.png",  # Horizontal logo PNG
                    f"https://cdn.brandfetch.io/{brand_domain}",           # Default (WebP)
                ]
                
                # Verify which logo URL is accessible
                for logo_url in logo_urls:
                    try:
                        # Quick HEAD request to check if logo exists
                        head_response = await client.head(logo_url, timeout=3.0)
                        
                        if head_response.status_code == 200:
                            logger.info(f"[BRANDFETCH] ✓ Found logo for {brand_name}: {logo_url}")
                            return logo_url
                    except:
                        continue
                
                # If no logo accessible, return the default icon URL anyway
                # (Brandfetch will show a fallback if logo doesn't exist)
                default_url = f"https://cdn.brandfetch.io/{brand_domain}/icon.png"
                logger.info(f"[BRANDFETCH] Using default logo URL for {brand_name}: {default_url}")
                return default_url
            
        except httpx.TimeoutException:
            logger.error(f"[BRANDFETCH] Timeout searching for {brand_name} logo")
            return None
        except Exception as e:
            logger.error(f"[BRANDFETCH] Error fetching logo for {brand_name}: {str(e)}")
            return None
    
    def _find_best_brand_match(self, search_term: str, results: list[Dict]) -> Optional[str]:
        """
        Find the best matching brand from search results.
        
        Prioritizes:
        1. Exact domain match (rxbar → rxbar.com)
        2. Verified/claimed brands
        3. Name similarity
        4. Quality score
        
        Args:
            search_term: The original brand name searched
            results: List of brand search results from Brandfetch
            
        Returns:
            Best matching domain or None
        """
        if not results:
            return None
        
        # Normalize search term for comparison
        search_normalized = search_term.lower().replace(' ', '').replace("'", '').replace('&', '')
        
        # Score each result
        best_match = None
        best_score = -1
        
        for result in results:
            domain = result.get('domain', '')
            name = result.get('name') or ''
            verified = result.get('verified', False)
            claimed = result.get('claimed', False)
            quality_score = result.get('qualityScore', 0)
            
            # Extract base domain (remove .com, .co, etc.)
            domain_base = domain.split('.')[0].lower()
            name_normalized = name.lower().replace(' ', '').replace("'", '').replace('&', '')
            
            # Calculate match score
            score = 0
            
            # 1. Exact domain match (highest priority)
            if domain_base == search_normalized:
                score += 100
                logger.debug(f"[BRANDFETCH] Exact domain match: {domain}")
            
            # 2. Exact name match
            if name_normalized == search_normalized:
                score += 90
                logger.debug(f"[BRANDFETCH] Exact name match: {name}")
            
            # 3. Domain contains search term
            if search_normalized in domain_base:
                score += 50
                logger.debug(f"[BRANDFETCH] Domain contains term: {domain}")
            
            # 4. Search term contains domain (for acronyms)
            if domain_base in search_normalized:
                score += 40
            
            # 5. Name contains search term
            if search_normalized in name_normalized:
                score += 30
            
            # 6. Verified/claimed bonus
            if verified:
                score += 20
            if claimed:
                score += 15
            
            # 7. Quality score bonus (0-10 points)
            score += quality_score * 10
            
            logger.debug(f"[BRANDFETCH] {domain} ({name}) - Score: {score}")
            
            if score > best_score:
                best_score = score
                best_match = domain
        
        if best_match:
            logger.info(f"[BRANDFETCH] Best match: {best_match} (score: {best_score})")
        
        return best_match


    async def search_product_packaging(
        self, 
        product_name: str, 
        brand: str,
        category: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search for product packaging information using OpenAI's native web search.
        Uses the Responses API via raw HTTP (Python SDK doesn't support it yet).
        
        Args:
            product_name: Name of the product
            brand: Brand name
            category: Optional product category for context
            
        Returns:
            {
                'packaging': str (packaging materials description),
                'materials': list of material names,
                'source': 'openai_web_search',
                'confidence': 'high' | 'medium' | 'low',
                'sources': list of source URLs,
                'note': str
            }
        """
        logger.info(f"[OPENAI WEB SEARCH] Searching for packaging: {brand} {product_name}")
        
        # Build search query for packaging
        query = f"""Find the packaging materials for the product below.

Brand: {brand}
Product: {product_name}
{f"Category: {category}" if category else ""}

Instructions:
- Search the manufacturer website or trusted retailers
- Find specific packaging materials (plastic pouch, cardboard box, glass jar, etc.)
- Include plastic types (PET, HDPE, PP, etc.) if available
- Include recyclability claims and environmental features
- Respond with a comma-separated list of materials ONLY
- If no information found, respond with "Not found"

Example: "Plastic pouch (PP), Cardboard sleeve (recycled)"
"""
        
        logger.info(f"[OPENAI WEB SEARCH] Query: {query}")
        
        try:
            # Use shared client for connection pooling
            client = self._get_client()
            
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4.1-mini",
                    "input": query,
                    "tools": [{"type": "web_search"}],
                },
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract output text from nested structure
            output_text = None
            sources = []
            
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            output_text = content.get("text")
                            
                            # Extract sources from annotations
                            for annotation in content.get("annotations", []):
                                if annotation.get("type") == "url_citation":
                                    url = annotation.get("url")
                                    if url:
                                        sources.append(url)
            
            # Deduplicate sources
            sources = list(dict.fromkeys(sources))
            
            if not output_text or len(output_text.strip()) < 5:
                logger.warning(f"[OPENAI WEB SEARCH] No packaging found for {brand} {product_name}")
                return None
            
            # Filter out non-results (exact matches)
            if output_text.lower() in ['not found', 'unknown', 'none', 'n/a', 'no information']:
                logger.warning(f"[OPENAI WEB SEARCH] No packaging information available")
                return None
            
            # Check if the response is actually saying "couldn't find" instead of providing packaging info
            negative_indicators = [
                "couldn't locate", "couldn't find", "could not find", "not available",
                "unable to locate", "unable to find", "no packaging", 
                "doesn't provide", "does not provide", "not listed", "not provided",
                "I don't have", "I cannot find", "not accessible"
            ]
            
            output_lower = output_text.lower()
            if any(indicator in output_lower for indicator in negative_indicators):
                logger.warning(f"[OPENAI WEB SEARCH] Response indicates no packaging found: {output_text[:100]}...")
                return None
            
            # Extract material names using common materials list
            materials = []
            common_materials = [
                'plastic', 'glass', 'metal', 'aluminum', 'paper', 'cardboard',
                'pet', 'hdpe', 'pvc', 'ldpe', 'pp', 'ps', 'steel', 'tin'
            ]
            
            text_lower = output_text.lower()
            for material in common_materials:
                if material in text_lower:
                    materials.append(material)
            
            # Remove duplicates
            materials = list(set(materials))
            
            # Determine confidence based on sources
            confidence = (
                "high" if len(sources) >= 2 else
                "medium" if len(sources) == 1 else
                "low"
            )
            
            logger.info(f"[OPENAI WEB SEARCH] Successfully found packaging: {materials}")
            logger.info(f"[OPENAI WEB SEARCH] Sources: {len(sources)}")
            
            return {
                "packaging": output_text,
                "materials": materials,
                "source": "openai_web_search",
                "confidence": confidence,
                "sources": sources,
                "note": f"Packaging retrieved via OpenAI web search ({len(sources)} sources). Verify with packaging."
            }
            
        except httpx.ConnectTimeout:
            logger.error(f"[OPENAI WEB SEARCH] Connection timeout - network issue or API unreachable")
            return None
        except httpx.ReadTimeout:
            logger.error(f"[OPENAI WEB SEARCH] Read timeout - web search took too long")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"[OPENAI WEB SEARCH] HTTP error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            logger.exception(f"[OPENAI WEB SEARCH] Packaging search failed: {e}")
            return None


# Global instance
web_search_service = WebSearchService()
