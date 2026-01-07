"""
Web Search Service for Product Ingredient Lookup
Uses multiple sources to find product ingredients when database is empty
"""

import os
import re
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import quote
import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class WebSearchService:
    """Service for searching the web for product ingredients"""
    
    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.timeout = 30
        
    async def search_product_ingredients(
        self, 
        product_name: str, 
        brand: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Multi-tier search for product ingredients
        
        Tries in order:
        1. SerpAPI Google search
        2. AI knowledge base (GPT-4o)
        3. Category-based generic analysis
        
        Returns:
            {
                'ingredients': str,
                'source': 'web_search' | 'ai_knowledge' | 'category_generic',
                'confidence': 'high' | 'medium' | 'low',
                'source_url': str (if from web),
                'note': str
            }
        """
        logger.info(f"Searching for ingredients: {brand} {product_name}")
        
        # Tier 1: AI Knowledge Base (FAST & FREE - try first for well-known products)
        ai_result = await self._search_with_ai_knowledge(product_name, brand, category)
        if ai_result:
            return ai_result
        
        # Tier 2: Web Search (slower, uses API quota)
        if self.serpapi_key:
            web_result = await self._search_with_serpapi(product_name, brand, category)
            if web_result:
                return web_result
        else:
            logger.warning("SERPAPI_KEY not configured, skipping web search")
        
        # Tier 3: Category-based generic
        generic_result = self._get_category_generic_info(category)
        return generic_result
    
    async def _search_with_serpapi(
        self, 
        product_name: str, 
        brand: str,
        category: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Search using SerpAPI for ingredient information"""
        try:
            # Build search query
            query = f'{brand} {product_name} ingredients list'
            
            params = {
                'q': query,
                'api_key': self.serpapi_key,
                'engine': 'google',
                'num': 5  # Get top 5 results
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    'https://serpapi.com/search',
                    params=params
                )
                response.raise_for_status()
                data = response.json()
            
            # Extract ingredients from search results
            ingredients_text = await self._extract_ingredients_from_results(
                data, 
                product_name, 
                brand
            )
            
            if ingredients_text:
                return {
                    'ingredients': ingredients_text,
                    'source': 'web_search',
                    'confidence': 'high',
                    'source_url': data.get('organic_results', [{}])[0].get('link', ''),
                    'note': 'Ingredients sourced from web search. For most accurate analysis, please verify with product label.'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return None
    
    async def _extract_ingredients_from_results(
        self, 
        search_data: Dict[str, Any],
        product_name: str,
        brand: str
    ) -> Optional[str]:
        """Use AI to extract ingredients from search results"""
        try:
            # Get organic results
            results = search_data.get('organic_results', [])
            if not results:
                return None
            
            # Compile snippets from top results
            snippets = []
            for result in results[:3]:  # Top 3 results
                snippet = result.get('snippet', '')
                if snippet:
                    snippets.append(f"Source: {result.get('title', 'Unknown')}\n{snippet}")
            
            if not snippets:
                return None
            
            combined_text = "\n\n".join(snippets)
            
            # Use AI to extract ingredient list
            prompt = f"""
You are analyzing search results to find the ingredient list for a specific product.

Product: {brand} {product_name}

Search Results:
{combined_text}

Task: Extract the complete ingredient list for this specific product.

Rules:
1. Only return ingredients if you find a clear, complete ingredient list
2. Return ingredients in order if specified
3. If multiple sources conflict, use the most authoritative (manufacturer website preferred)
4. If no clear ingredient list is found, return "NOT_FOUND"
5. Format as comma-separated list

Ingredient List:
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            ingredients = response.choices[0].message.content.strip()
            
            if ingredients and ingredients != "NOT_FOUND" and len(ingredients) > 20:
                return ingredients
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract ingredients from search results: {e}")
            return None
    
    async def _search_with_ai_knowledge(
        self,
        product_name: str,
        brand: str,
        category: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Ask AI about known product formulations - uses GPT's training data knowledge"""
        try:
            # Simple, direct prompt like asking ChatGPT
            prompt = f"""What are the ingredients in {brand} {product_name}?

Please list all ingredients you know for this product in order from highest to lowest concentration.

If you don't have information about this specific product, respond with only: UNKNOWN"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that provides ingredient lists for commercial products. You have knowledge of many popular consumer products from your training data. When asked about a product you know, provide a complete comma-separated ingredient list."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Higher for better recall
                max_tokens=1000  # Allow longer responses
            )
            
            ingredients = response.choices[0].message.content.strip()
            
            # DETAILED DEBUG LOGGING - Show exact AI response
            logger.info(f"=" * 80)
            logger.info(f"AI KNOWLEDGE RESPONSE for: {brand} {product_name}")
            logger.info(f"Full Response: {ingredients}")
            logger.info(f"Response Length: {len(ingredients)} characters")
            logger.info(f"Is UNKNOWN: {ingredients.upper() == 'UNKNOWN'}")
            logger.info(f"=" * 80)
            
            # Check if AI has knowledge - be less strict
            if not ingredients or (ingredients.upper() == "UNKNOWN") or (len(ingredients) < 20):
                logger.info(f"AI has no useful knowledge of product: {brand} {product_name}")
                logger.info(f"Reason: empty={not ingredients}, is_unknown={ingredients.upper() == 'UNKNOWN' if ingredients else False}, too_short={len(ingredients) < 20 if ingredients else False}")
                return None
            
            # Determine confidence based on response detail
            ingredient_count = len(ingredients.split(','))
            confidence = 'high' if ingredient_count >= 10 else 'medium'
            
            logger.info(f"AI knowledge found {ingredient_count} ingredients for {brand} {product_name}")
            
            return {
                'ingredients': ingredients,
                'source': 'ai_knowledge',
                'confidence': confidence,
                'note': f'Ingredients from AI training data knowledge. Contains {ingredient_count} ingredients. This may not reflect the most current formulation - always verify with product packaging.'
            }
            
            return None
            
        except Exception as e:
            logger.error(f"AI knowledge search failed: {e}")
            return None
    
    def _get_category_generic_info(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Provide category-based generic information"""
        
        category_templates = {
            'body wash': 'Water, Sodium Laureth Sulfate, Cocamidopropyl Betaine, Glycerin, Fragrance',
            'shampoo': 'Water, Sodium Laureth Sulfate, Cocamidopropyl Betaine, Sodium Chloride, Fragrance',
            'lotion': 'Water, Glycerin, Mineral Oil, Petrolatum, Dimethicone, Fragrance',
            'soap': 'Sodium Palmate, Sodium Palm Kernelate, Water, Glycerin, Fragrance',
            'toothpaste': 'Water, Sorbitol, Hydrated Silica, Sodium Lauryl Sulfate, Flavor',
            'deodorant': 'Aluminum Zirconium, Cyclopentasiloxane, Stearyl Alcohol, Fragrance',
        }
        
        generic_ingredients = None
        if category:
            category_lower = category.lower()
            for key, ingredients in category_templates.items():
                if key in category_lower:
                    generic_ingredients = ingredients
                    break
        
        if not generic_ingredients:
            generic_ingredients = "Unable to determine ingredients without product label or database information"
        
        return {
            'ingredients': generic_ingredients,
            'source': 'category_generic',
            'confidence': 'low',
            'note': f'Generic {category or "product"} ingredients provided. For accurate analysis, please take a photo of the ingredient label or search for this specific product online.'
        }
    
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
            
            async with httpx.AsyncClient(timeout=10.0) as client:
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


# Global instance
web_search_service = WebSearchService()
