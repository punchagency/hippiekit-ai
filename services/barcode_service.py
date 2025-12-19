"""
Barcode Product Lookup Service
Queries Open Food Facts, Open Beauty Facts, and Open Product Facts APIs
"""

import requests
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BarcodeService:
    """Service for looking up products by barcode across multiple Open*Facts databases"""
    
    # API endpoints for different product databases
    OPENFOODFACTS_API = "https://world.openfoodfacts.org/api/v2/product/{barcode}"
    OPENBEAUTYFACTS_API = "https://world.openbeautyfacts.org/api/v2/product/{barcode}"
    OPENPRODUCTFACTS_API = "https://world.openproductfacts.org/api/v2/product/{barcode}"
    
    def __init__(self):
        self.timeout = 10  # seconds
        
    def lookup_barcode(self, barcode: str) -> Optional[Dict[str, Any]]:
        """
        Look up a product by barcode across all Open*Facts databases.
        Tries each database in order until a product is found.
        
        Args:
            barcode: Product barcode (UPC, EAN, etc.)
            
        Returns:
            Normalized product data or None if not found
        """
        # Try each database in order
        databases = [
            ("OpenFoodFacts", self.OPENFOODFACTS_API),
            ("OpenBeautyFacts", self.OPENBEAUTYFACTS_API),
            ("OpenProductFacts", self.OPENPRODUCTFACTS_API),
        ]
        
        for db_name, api_url in databases:
            try:
                logger.info(f"Querying {db_name} for barcode: {barcode}")
                product_data = self._query_api(api_url.format(barcode=barcode))
                
                if product_data:
                    logger.info(f"Product found in {db_name}")
                    normalized = self._normalize_product_data(product_data, db_name)
                    return normalized
                    
            except Exception as e:
                logger.warning(f"Error querying {db_name}: {str(e)}")
                continue
        
        logger.info(f"Product not found in any database for barcode: {barcode}")
        return None
    
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
    
    def _normalize_product_data(self, product: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Normalize product data from different Open*Facts APIs into a consistent format
        
        Args:
            product: Raw product data from API
            source: Name of the source database
            
        Returns:
            Normalized product dictionary
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
        
        return normalized
    
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
