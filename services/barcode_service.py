"""
Barcode Product Lookup Service
Queries Open Food Facts, Open Beauty Facts, and Open Product Facts APIs
Includes web search fallback for products with empty ingredient data
Includes vision fallback for products with "Unknown Product" name
"""

import requests
import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any, List
import logging
from services.chemical_checker import check_ingredients, calculate_safety_score, generate_recommendations
from services.web_search_service import web_search_service
from services.cache_service import cache_service
from services.timing_logger import async_time_operation, log_timing_summary

logger = logging.getLogger(__name__)


async def download_image_from_url(url: str) -> Optional[bytes]:
    """Download image from URL and return bytes"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.read()
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
    return None


class BarcodeService:
    """Service for looking up products by barcode across multiple Open*Facts databases"""
    
    # API endpoints for different product databases
    OPENFOODFACTS_API = "https://world.openfoodfacts.org/api/v2/product/{barcode}"
    OPENBEAUTYFACTS_API = "https://world.openbeautyfacts.org/api/v2/product/{barcode}"
    OPENPRODUCTSFACTS_API = "https://world.openproductsfacts.org/api/v2/product/{barcode}"  # Note: productSfacts with 's'
    UPCITEMDB_API = "https://api.upcitemdb.com/prod/trial/lookup?upc={barcode}"
    
    def __init__(self):
        self.timeout = 10  # seconds
        self.web_search = web_search_service
        self.cache = cache_service
        
    async def lookup_barcode(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Look up a product by barcode across all Open*Facts databases.
        Checks cache first, then queries all databases in parallel if not cached.
        If found but ingredients are empty, attempts web search fallback.
        
        Args:
            barcode: Product barcode (UPC, EAN, etc.)
            
        Returns:
            Normalized product data or None if not found
        """
        total_start = time.time()
        timings = {}
        
        # Check cache first
        cache_start = time.time()
        cached_result = await self.cache.get_barcode(barcode)
        timings["cache_check"] = (time.time() - cache_start) * 1000
        
        if cached_result:
            print(f"   ⚡ [CACHE HIT] Barcode {barcode} found in cache")
            logger.info(f"Returning cached result for barcode: {barcode}")
            return cached_result
        
        print(f"   📭 [CACHE MISS] Barcode {barcode} not in cache, querying databases...")
        
        # Cache miss - query databases
        db_start = time.time()
        result = await self._fetch_from_databases(barcode)
        timings["database_queries"] = (time.time() - db_start) * 1000
        
        # Cache the result if found
        if result:
            cache_set_start = time.time()
            await self.cache.set_barcode(barcode, result)
            timings["cache_set"] = (time.time() - cache_set_start) * 1000
        
        # Log timing summary
        total_ms = (time.time() - total_start) * 1000
        log_timing_summary(timings, total_ms, f"lookup_barcode({barcode})")
        
        return result
    
    async def _fetch_from_databases(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Fetch product data from Open*Facts databases in parallel.
        OPTIMIZED: Waits for all results and picks the BEST quality data.
        Prioritizes Open*Facts databases over UPCItemDB.
        
        Args:
            barcode: Product barcode
            
        Returns:
            Normalized product data or None if not found
        """
        # Database configurations - Open*Facts first (better data quality)
        databases = [
            ("OpenFoodFacts", self.OPENFOODFACTS_API.format(barcode=barcode)),
            ("OpenBeautyFacts", self.OPENBEAUTYFACTS_API.format(barcode=barcode)),
            ("OpenProductsFacts", self.OPENPRODUCTSFACTS_API.format(barcode=barcode)),
            ("UPCItemDB", self.UPCITEMDB_API.format(barcode=barcode)),
        ]
        
        # Create parallel tasks for all databases
        tasks = [
            self._query_api_async(url, db_name) 
            for db_name, url in databases
        ]
        
        # Wait for ALL results (with timeout) and pick the best one
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Score each result and pick the best
        best_result = None
        best_score = -1
        best_db = None
        
        for result in results:
            if isinstance(result, Exception):
                continue
            
            db_name, product_data = result
            if not product_data:
                continue
            
            # Score the result based on data quality
            score = self._score_product_data(product_data, db_name)
            logger.info(f"Product found in {db_name} with quality score: {score}")
            
            if score > best_score:
                best_score = score
                best_result = product_data
                best_db = db_name
        
        if best_result:
            logger.info(f"Selected best result from {best_db} (score: {best_score})")
            return await self._normalize_product_data(best_result, best_db)
        
        logger.info(f"Product not found in any database for barcode: {barcode}")
        return None
    
    def _score_product_data(self, product_data: Dict[str, Any], db_name: str) -> int:
        """
        Score product data quality. Higher score = better data.
        Open*Facts databases get bonus points.
        
        Args:
            product_data: Raw product data from API
            db_name: Source database name
            
        Returns:
            Quality score (higher is better)
        """
        score = 0
        
        # Database priority bonus (Open*Facts > UPCItemDB)
        db_priority = {
            "OpenFoodFacts": 50,
            "OpenBeautyFacts": 45,
            "OpenProductsFacts": 40,
            "UPCItemDB": 10,  # Much lower priority - often has bad data
        }
        score += db_priority.get(db_name, 0)
        
        # Has product name (+20)
        if db_name == "UPCItemDB":
            name = product_data.get("title", "")
        else:
            name = product_data.get("product_name", "") or product_data.get("product_name_en", "")
        if name and len(name) > 3:
            score += 20
        
        # Has brand (+15)
        brand = product_data.get("brand", "") or product_data.get("brands", "")
        if brand and len(brand) > 1:
            score += 15
        
        # Has ingredients (+30 - most important!)
        if db_name == "UPCItemDB":
            has_ingredients = False  # UPCItemDB doesn't have ingredients
        else:
            ingredients_text = product_data.get("ingredients_text", "") or product_data.get("ingredients_text_en", "")
            has_ingredients = ingredients_text and len(ingredients_text) > 10
        if has_ingredients:
            score += 30
        
        # Has image (+10)
        if db_name == "UPCItemDB":
            has_image = product_data.get("images") and len(product_data.get("images", [])) > 0
        else:
            has_image = product_data.get("image_url") or product_data.get("image_front_url")
        if has_image:
            score += 10
        
        # Has valid category (+10) - UPCItemDB often has wrong categories
        if db_name != "UPCItemDB":
            categories = product_data.get("categories", "")
            if categories and "Media" not in categories and "DVDs" not in categories:
                score += 10
        
        return score
    
    async def _query_api_async(self, url: str, db_name: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Async query to an Open*Facts API endpoint
        
        Args:
            url: Full API URL with barcode
            db_name: Name of the database being queried
            
        Returns:
            Tuple of (db_name, product_data) or (db_name, None) if not found
        """
        try:
            logger.info(f"Querying {db_name}")
            headers = {
                "User-Agent": "Hippiekit - Product Scanner App - Contact: support@hippiekit.com"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle UPCItemDB API response format
                        if db_name == "UPCItemDB":
                            if data.get("code") == "OK" and data.get("items") and len(data["items"]) > 0:
                                return (db_name, data["items"][0])
                        # Handle Open*Facts API response format
                        elif data.get("status") == 1 and data.get("product"):
                            return (db_name, data["product"])
            
            return (db_name, None)
            
        except Exception as e:
            logger.error(f"{db_name} API request failed: {str(e)}")
            return (db_name, None)
    
    def _query_api(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Query an Open*Facts API endpoint
        
        Args:
            url: Full API URL with barcode
            
        Returns:
            Product data from API or None if not found
        """
        try:
            headers = {
                "User-Agent": "Hippiekit - Product Scanner App - Contact: support@hippiekit.com"
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                # Check if product was found
                if data.get("status") == 1 and data.get("product"):
                    return data["product"]
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
    
    async def _normalize_product_data(self, product: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Normalize product data from different Open*Facts APIs into a consistent format
        Includes vision fallback if product name is "Unknown Product"
        Includes web search fallback if ingredients are missing
        
        Args:
            product: Raw product data from API
            source: Name of the source database
            
        Returns:
            Normalized product dictionary with chemical analysis
        """
        # Handle UPCItemDB API format
        if source == "UPCItemDB":
            normalized = {
                "barcode": product.get("ean", ""),
                "name": product.get("title", "Unknown Product"),
                "brands": product.get("brand", ""),
                "categories": product.get("category", ""),
                "source": source,
                "image_url": product.get("images", [""])[0] if product.get("images") else "",
                "ingredients": [],
                "materials": {},
                "nutrition": {},
                "labels": "",
                "packaging": "",
                "quantity": product.get("size", ""),
                "countries": "",
                "url": "",
                "description": product.get("description", ""),
                "asin": product.get("asin", ""),
                "model": product.get("model", ""),
                "color": product.get("color", ""),
                "dimension": product.get("dimension", ""),
                "weight": product.get("weight", ""),
                "lowest_price": product.get("lowest_recorded_price"),
                "highest_price": product.get("highest_recorded_price"),
            }
            ingredients_text = ""
        else:
            # Extract common fields for Open*Facts APIs
            normalized = {
                "barcode": product.get("code", ""),
                "name": product.get("product_name", "") or product.get("product_name_en", "Unknown Product"),
                "brands": product.get("brands", ""),
                "categories": product.get("categories", ""),
                "source": source,
                "image_url": self._get_best_image(product),
                "ingredients": self._extract_ingredients(product),
                "materials": self._extract_materials(product),
                "nutrition": self._extract_nutrition(product),
                "labels": product.get("labels", ""),
                "packaging": product.get("packaging", ""),
                "quantity": product.get("quantity", ""),
                "countries": product.get("countries", ""),
                "url": product.get("url", ""),
            }
            
            # === Get ingredients text for chemical checking ===
            ingredients_text = (
                product.get("ingredients_text", "") or 
                product.get("ingredients_text_en", "")
            )
        
        # === VISION FALLBACK: If product name is "Unknown Product" and we have an image ===
        product_name = normalized.get("name", "")
        is_unknown = (
            not product_name or 
            product_name.lower() in ["unknown product", "unknown", ""] or
            len(product_name) < 3
        )
        
        if is_unknown and normalized.get("image_url"):
            logger.info(f"Product name is '{product_name}' - attempting vision identification from image")
            try:
                # Import vision service here to avoid circular imports
                from services.vision_service import get_vision_service
                
                # Download the image
                image_bytes = await download_image_from_url(normalized["image_url"])
                
                if image_bytes:
                    logger.info(f"Downloaded image ({len(image_bytes)} bytes), running vision identification...")
                    vision_service = get_vision_service()
                    vision_result = await vision_service.identify_product_from_photo(image_bytes)
                    
                    if vision_result and vision_result.get("product_name"):
                        # Update normalized data with vision results
                        logger.info(f"✅ Vision identified: {vision_result.get('brand', '')} {vision_result.get('product_name')}")
                        normalized["name"] = vision_result.get("product_name", normalized["name"])
                        normalized["brands"] = vision_result.get("brand", "") or normalized["brands"]
                        normalized["categories"] = vision_result.get("category", "") or normalized["categories"]
                        normalized["source"] = f"{source} + vision"
                        
                        # Store vision data for later use
                        normalized["vision_data"] = {
                            "product_type": vision_result.get("product_type", ""),
                            "marketing_claims": vision_result.get("marketing_claims", []),
                            "certifications_visible": vision_result.get("certifications_visible", []),
                            "container_info": vision_result.get("container_info", {}),
                        }
                    else:
                        logger.warning("Vision identification returned no product name")
                else:
                    logger.warning("Failed to download image for vision identification")
            except Exception as e:
                logger.error(f"Vision fallback failed: {e}")
        
        # === NEW: Web Search Fallback if ingredients empty ===
        data_source = "database"
        confidence = "high"
        ingredients_note = None
        
        if not ingredients_text or len(ingredients_text.strip()) < 10:
            logger.info(f"Ingredients empty in database for {normalized['name']}, attempting web search...")
            
            # Try web search for ingredients
            web_result = await self.web_search.search_product_ingredients(
                product_name=normalized['name'],
                brand=normalized['brands'],
                category=normalized['categories'].split(',')[0] if normalized['categories'] else None
            )
            
            if web_result and web_result.get('ingredients'):
                ingredients_text = web_result['ingredients']
                data_source = web_result['source']
                confidence = web_result['confidence']
                ingredients_note = web_result['note']
                logger.info(f"Ingredients found via {data_source} with {confidence} confidence")
        
        # Check for harmful chemicals
        chemical_flags = check_ingredients(ingredients_text) if ingredients_text else []
        safety_score = calculate_safety_score(chemical_flags)
        
        # Get product category for targeted recommendations
        category = product.get("categories", "").split(",")[0].strip() if product.get("categories") else None
        recommendations = generate_recommendations(chemical_flags, category)
        
        # Add chemical analysis to response
        normalized["chemical_analysis"] = {
            "flags": [
                {
                    "chemical": flag["chemical"],
                    "category": flag["category"],
                    "severity": flag["severity"],
                    "why_flagged": self._get_chemical_explanation(flag["chemical"], flag["category"])
                }
                for flag in chemical_flags
            ],
            "safety_score": safety_score,
            "recommendations": recommendations,
            "data_source": data_source,
            "confidence": confidence
        }
        
        # Add note if ingredients from web search
        if ingredients_note:
            normalized["ingredients_note"] = ingredients_note
        
        return normalized
    
    def _get_chemical_explanation(self, chemical: str, category: str) -> str:
        """Get brief explanation for why a chemical is flagged"""
        explanations = {
            "Preservatives": "Preservative that may cause irritation or hormonal disruption",
            "Surfactants": "Harsh cleaning agent that can strip natural oils and cause irritation",
            "Fragrance": "Undisclosed chemical mixture that may contain allergens and toxins",
            "Dyes": "Synthetic colorant linked to allergies and potential health risks",
            "Sweeteners": "Artificial sweetener with potential metabolic and neurological effects",
            "Seed Oils": "Highly processed oil prone to oxidation and inflammation",
            "Plastics": "Plastic-derived chemical that may leach into products",
            "PFAS": "Forever chemical that accumulates in body and environment",
            "Pesticides": "Toxic agricultural chemical with potential carcinogenic effects",
            "Heavy Metals": "Toxic metal that accumulates in body and causes organ damage",
            "Sunscreen": "Chemical UV filter that may disrupt hormones and harm coral reefs",
        }
        return explanations.get(category, f"{category} chemical with potential health concerns")
    
    def _get_best_image(self, product: Dict[str, Any]) -> str:
        """
        Get the best available product image URL
        
        Args:
            product: Product data
            
        Returns:
            Image URL or empty string
        """
        # Try different image fields in order of preference
        image_fields = [
            "image_url",
            "image_front_url",
            "image_small_url",
            "image_thumb_url",
        ]
        
        for field in image_fields:
            if product.get(field):
                return product[field]
        
        return ""
    
    def _extract_ingredients(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and structure ingredient information
        
        Args:
            product: Product data
            
        Returns:
            List of ingredient dictionaries
        """
        ingredients_list = []
        
        # Get ingredients text
        ingredients_text = (
            product.get("ingredients_text", "") or 
            product.get("ingredients_text_en", "")
        )
        
        if ingredients_text:
            ingredients_list.append({
                "text": ingredients_text,
                "type": "full_text"
            })
        
        # Get structured ingredients if available
        if product.get("ingredients"):
            for ing in product["ingredients"]:
                if isinstance(ing, dict):
                    ingredients_list.append({
                        "id": ing.get("id", ""),
                        "text": ing.get("text", ""),
                        "percent": ing.get("percent", None),
                        "vegan": ing.get("vegan", None),
                        "vegetarian": ing.get("vegetarian", None),
                        "type": "structured"
                    })
        
        return ingredients_list
    
    def _extract_materials(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract material and packaging information
        
        Args:
            product: Product data
            
        Returns:
            Dictionary of material information
        """
        return {
            "packaging": product.get("packaging", ""),
            "packaging_text": product.get("packaging_text", ""),
            "packaging_tags": product.get("packaging_tags", []),
            "materials": product.get("materials_tags", []),
        }
    
    def _extract_nutrition(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract nutrition information
        
        Args:
            product: Product data
            
        Returns:
            Dictionary of nutrition facts
        """
        nutriments = product.get("nutriments", {})
        
        if not nutriments:
            return {}
        
        # Extract common nutrition fields
        nutrition = {
            "energy_kcal": nutriments.get("energy-kcal_100g"),
            "energy_kj": nutriments.get("energy-kj_100g"),
            "fat": nutriments.get("fat_100g"),
            "saturated_fat": nutriments.get("saturated-fat_100g"),
            "carbohydrates": nutriments.get("carbohydrates_100g"),
            "sugars": nutriments.get("sugars_100g"),
            "fiber": nutriments.get("fiber_100g"),
            "proteins": nutriments.get("proteins_100g"),
            "salt": nutriments.get("salt_100g"),
            "sodium": nutriments.get("sodium_100g"),
            "nutrition_score": product.get("nutrition_score_fr"),
            "nutrition_grade": product.get("nutrition_grades", ""),
        }
        
        # Remove None values
        nutrition = {k: v for k, v in nutrition.items() if v is not None}
        
        return nutrition


# Singleton instance
_barcode_service = None

def get_barcode_service() -> BarcodeService:
    """Get or create the singleton BarcodeService instance"""
    global _barcode_service
    if _barcode_service is None:
        _barcode_service = BarcodeService()
    return _barcode_service
