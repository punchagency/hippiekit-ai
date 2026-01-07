"""
Barcode Product Lookup Service
Queries Open Food Facts, Open Beauty Facts, and Open Product Facts APIs
Includes web search fallback for products with empty ingredient data
"""

import requests
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
import logging
from services.chemical_checker import check_ingredients, calculate_safety_score, generate_recommendations
from services.web_search_service import web_search_service
from services.cache_service import cache_service

logger = logging.getLogger(__name__)


class BarcodeService:
    """Service for looking up products by barcode across multiple Open*Facts databases"""
    
    # API endpoints for different product databases
    OPENFOODFACTS_API = "https://world.openfoodfacts.org/api/v2/product/{barcode}"
    OPENBEAUTYFACTS_API = "https://world.openbeautyfacts.org/api/v2/product/{barcode}"
    OPENPRODUCTSFACTS_API = "https://world.openproductsfacts.org/api/v2/product/{barcode}"  # Note: productSfacts with 's'
    
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
        # Check cache first
        cached_result = await self.cache.get_barcode(barcode)
        if cached_result:
            logger.info(f"Returning cached result for barcode: {barcode}")
            return cached_result
        
        # Cache miss - query databases
        result = await self._fetch_from_databases(barcode)
        
        # Cache the result if found
        if result:
            await self.cache.set_barcode(barcode, result)
        
        return result
    
    async def _fetch_from_databases(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Fetch product data from Open*Facts databases in parallel
        
        Args:
            barcode: Product barcode
            
        Returns:
            Normalized product data or None if not found
        """
        # Database configurations
        databases = [
            ("OpenFoodFacts", self.OPENFOODFACTS_API.format(barcode=barcode)),
            ("OpenBeautyFacts", self.OPENBEAUTYFACTS_API.format(barcode=barcode)),
            ("OpenProductsFacts", self.OPENPRODUCTSFACTS_API.format(barcode=barcode)),
        ]
        
        # Create parallel tasks for all databases
        tasks = [
            self._query_api_async(url, db_name) 
            for db_name, url in databases
        ]
        
        # Wait for first successful result
        for task in asyncio.as_completed(tasks):
            try:
                db_name, product_data = await task
                if product_data:
                    logger.info(f"Product found in {db_name}")
                    normalized = await self._normalize_product_data(product_data, db_name)
                    return normalized
            except Exception as e:
                logger.warning(f"Error in parallel query: {str(e)}")
                continue
        
        logger.info(f"Product not found in any database for barcode: {barcode}")
        return None
    
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
                        # Check if product was found
                        if data.get("status") == 1 and data.get("product"):
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
        Includes web search fallback if ingredients are missing
        
        Args:
            product: Raw product data from API
            source: Name of the source database
            
        Returns:
            Normalized product dictionary with chemical analysis
        """
        # Extract common fields
        normalized = {
            "barcode": product.get("code", ""),
            "name": product.get("product_name", "") or product.get("product_name_en", "Unknown Product"),
            "brand": product.get("brands", ""),
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
        
        # === NEW: Web Search Fallback if ingredients empty ===
        data_source = "database"
        confidence = "high"
        ingredients_note = None
        
        if not ingredients_text or len(ingredients_text.strip()) < 10:
            logger.info(f"Ingredients empty in database for {normalized['name']}, attempting web search...")
            
            # Try web search for ingredients
            web_result = await self.web_search.search_product_ingredients(
                product_name=normalized['name'],
                brand=normalized['brand'],
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
