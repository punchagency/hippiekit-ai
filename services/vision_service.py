"""
AI Vision Service for OCR and Product Analysis
Uses OpenAI Vision to extract text, ingredients, packaging, and provide recommendations.
"""

import os
import base64
from typing import Optional, Dict, Any
import logging
from openai import OpenAI
import time
from PIL import Image
import io
from services.chemical_checker import (
    check_ingredients,
    calculate_safety_score,
    generate_recommendations,
    get_condensed_chemical_list
)

logger = logging.getLogger(__name__)

# Configure logging to show INFO level
logging.basicConfig(level=logging.INFO)


class VisionService:
    """Service for AI-powered image analysis and OCR using OpenAI Vision"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not api_key.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY appears to be invalid (should start with 'sk-')")
        
        self.client = OpenAI(
            api_key=api_key,
            timeout=60.0,  # 60 second timeout for API calls
            max_retries=2
        )
        # gpt-4o-mini is 4x faster than gpt-4o for vision tasks with similar accuracy
        self.model = "gpt-4o-mini"
        logger.info(f"VisionService initialized with model {self.model}")

    def analyze_product_image(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Analyze a product image to extract visible info and provide structured assessment.
        Returns None on failure.
        """
        start_time = time.time()
        try:
            logger.info(f"Starting vision analysis for image ({len(image_bytes)} bytes)")
            
            # Compress image if needed
            logger.info("Compressing image if needed...")
            image_bytes = self._compress_image(image_bytes)
            logger.info(f"Image ready for analysis ({len(image_bytes)} bytes)")
            
            # Encode image
            logger.info("Encoding image to base64...")
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            logger.info(f"Image encoded ({len(base64_image)} chars)")
            
            prompt = self._create_prompt()
            
            logger.info(f"Calling OpenAI Vision API with model {self.model}...")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise product analyst. Only use VISIBLE information in the image. "
                            "Do not invent ingredients. Output clear, structured text matching the requested sections."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low",  # low detail is 4x faster, sufficient for product labels
                                },
                            },
                        ],
                    },
                ],
                max_tokens=1000,  # reduced from 1600 for faster responses
                temperature=0.2,
            )
            
            elapsed = time.time() - start_time
            logger.info(f"OpenAI API call completed in {elapsed:.2f}s")
            
            text = resp.choices[0].message.content or ""
            logger.info(f"Response received ({len(text)} chars)")
            
            result = self._parse_response(text)
            
            # === NEW: Post-process with chemical checker ===
            # Extract ingredients from Vision AI response
            ingredients_text = result.get("ingredients", "")
            
            if ingredients_text and ingredients_text != "Not visible in image":
                # Check for harmful chemicals
                chemical_flags = check_ingredients(ingredients_text)
                safety_score = calculate_safety_score(chemical_flags)
                
                # Get product category for targeted recommendations
                category = result.get("product_info", {}).get("category", "")
                recommendations = generate_recommendations(chemical_flags, category)
                
                # Add chemical analysis to result
                result["chemical_analysis"] = {
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
                    "recommendations": recommendations
                }
            else:
                # No ingredients visible, can't check chemicals
                result["chemical_analysis"] = {
                    "flags": [],
                    "safety_score": None,
                    "recommendations": {
                        "avoid": ["Cannot analyze - ingredients not visible in image"],
                        "look_for": ["Try scanning the ingredients panel clearly"],
                        "certifications": []
                    }
                }
            
            logger.info("Vision analysis completed successfully")
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Vision analysis failed after {elapsed:.2f}s: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None

    def _create_prompt(self) -> str:
        # Get condensed chemical list
        chemical_list = get_condensed_chemical_list()
        
        return (
            "Analyze this product image and extract ONLY what is visible. If a field isn't visible, say 'Not visible in image'.\n\n"
            f"{chemical_list}\n\n"
            "When listing ingredients, CHECK THEM against the red flag list above. Flag any harmful chemicals you see.\n\n"
            "Provide these sections exactly (plain text):\n\n"
            "PRODUCT INFO:\n"
            "Name: ...\n"
            "Brand: ...\n"
            "Category: (food/beverage/cosmetic/household/textile/other)\n"
            "Type: (e.g., shampoo, chips, detergent, pillow)\n\n"
            "VISIBLE TEXT:\n[All readable text or 'Text not readable']\n\n"
            "INGREDIENTS:\n[List each ingredient on a new line, or 'Not visible in image']\n\n"
            "CONCERNING CHEMICALS:\n[List any chemicals from the RED FLAG list found in ingredients, with their category. Or 'None visible' or 'Cannot determine']\n\n"
            "PACKAGING:\n"
            "Material: ...\n"
            "Type: ...\n"
            "Recyclable: yes/no/unknown\n\n"
            "NUTRITION:\n[Details or 'Not visible']\n\n"
            "HEALTH SCORE: X/10\n"
            "Health Risk: Low/Medium/High\n"
            "Health Concerns: ...\n"
            "Health Benefits: ...\n"
            "Explanation: 2-3 sentences based on visible info\n\n"
            "ECO SCORE: X/10\n"
            "Environmental Concerns: ...\n"
            "Positive Eco Attributes: ...\n"
            "Explanation: 2-3 sentences\n\n"
            "RECOMMENDATION:\n"
            "Avoid: Yes/No/Maybe\n"
            "Reasons: ...\n"
            "Alternatives: Generic product types or features to look for instead\n"
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        def field(name: str, after: str | None = None) -> str:
            try:
                segment = text[text.find(after):] if after and text.find(after) != -1 else text
                i = segment.find(name)
                if i == -1:
                    return "Not available"
                i += len(name)
                j = segment.find("\n", i)
                return (segment[i:] if j == -1 else segment[i:j]).strip() or "Not available"
            except Exception:
                return "Not available"

        def section(start_marker: str, end_marker: str) -> str:
            try:
                i = text.find(start_marker)
                if i == -1:
                    return "Not available"
                i += len(start_marker)
                j = text.find(end_marker, i)
                return (text[i:] if j == -1 else text[i:j]).strip() or "Not available"
            except Exception:
                return "Not available"

        def score(marker: str) -> float:
            try:
                s = field(marker)
                if "/" in s:
                    return float(s.split("/")[0].strip())
                return float(s)
            except Exception:
                return 0.0

        return {
            "product_info": {
                "name": field("Name:"),
                "brand": field("Brand:"),
                "category": field("Category:"),
                "type": field("Type:")
            },
            "visible_text": section("VISIBLE TEXT:", "INGREDIENTS:"),
            "ingredients": section("INGREDIENTS:", "CONCERNING CHEMICALS:"),
            "concerning_chemicals": section("CONCERNING CHEMICALS:", "PACKAGING:"),
            "packaging": {
                "material": field("Material:", after="PACKAGING:"),
                "type": field("Type:", after="PACKAGING:"),
                "recyclable": field("Recyclable:", after="PACKAGING:"),
            },
            "nutrition": section("NUTRITION:", "HEALTH SCORE:"),
            "health_assessment": {
                "score": score("HEALTH SCORE:"),
                "risk_level": field("Health Risk:"),
                "concerns": field("Health Concerns:"),
                "benefits": field("Health Benefits:"),
                "explanation": field("Explanation:", after="HEALTH SCORE"),
            },
            "eco_assessment": {
                "score": score("ECO SCORE:"),
                "concerns": field("Environmental Concerns:"),
                "benefits": field("Positive Eco Attributes:"),
                "explanation": field("Explanation:", after="ECO SCORE"),
            },
            "recommendation": {
                "avoid": field("Avoid:"),
                "reasons": field("Reasons:"),
                "alternatives": field("Alternatives:"),
            },
            "raw_analysis": text,
        }

    def _compress_image(self, image_bytes: bytes, max_size: int = 512, quality: int = 80) -> bytes:
        """
        Compress image to reduce size and processing time.
        Max size 512x512 is optimal for OpenAI Vision API low detail mode.
        Returns compressed image bytes.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            original_size = img.size
            
            # Convert RGBA to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            # Resize if needed
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {original_size} to {img.size}")
            
            # Convert to JPEG with quality setting
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            compressed = buffer.getvalue()
            
            compression_ratio = len(compressed) / len(image_bytes) * 100
            logger.info(f"Compressed image: {len(image_bytes)} → {len(compressed)} bytes ({compression_ratio:.1f}%)")
            
            return compressed
            
        except Exception as e:
            logger.warning(f"Image compression failed, using original: {e}")
            return image_bytes
    
    def _get_chemical_explanation(self, chemical: str, category: str) -> str:
        """Get brief explanation for why a chemical is flagged"""
        explanations = {
            "Preservatives": "Preservative that may cause irritation or hormonal disruption",
            "Surfactants": "Harsh cleaning agent that can strip natural oils and cause irritation",
            "Fragrance": "Undisclosed chemical mixture that may contain allergens and toxins",
            "Dyes": "Synthetic colorant linked to allergies and potential health risks",
            "Flavors": "Artificial flavor with potential neurological or metabolic effects",
            "Sweeteners": "Artificial sweetener with potential metabolic and neurological effects",
            "Seed Oils": "Highly processed oil prone to oxidation and inflammation",
            "Textiles": "Synthetic textile chemical that may irritate skin or release VOCs",
            "Plastics": "Plastic-derived chemical that may leach into products",
            "PFAS": "Forever chemical that accumulates in body and environment",
            "Pesticides": "Toxic agricultural chemical with potential carcinogenic effects",
            "Heavy Metals": "Toxic metal that accumulates in body and causes organ damage",
            "Sunscreen": "Chemical UV filter that may disrupt hormones and harm coral reefs",
            "Antimicrobials": "Antimicrobial agent that contributes to antibiotic resistance",
            "Toothpaste": "Potentially harmful ingredient commonly found in oral care",
            "Silicones": "Synthetic polymer that doesn't biodegrade and may clog pores",
            "Cosmetics": "Cosmetic ingredient with potential toxicity or environmental concerns",
            "Thickeners": "Thickening agent that may cause digestive or inflammatory issues",
            "Processing Aids": "Industrial chemical used in processing with health concerns"
        }
        return explanations.get(category, f"{category} chemical with potential health concerns")

    async def identify_product_from_photo(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Identify product from a front-facing photo
        Extracts product name, brand, category, and visible information
        
        Returns:
            {
                'product_name': str,
                'brand': str,
                'category': str,
                'product_type': str,
                'marketing_claims': list,
                'certifications_visible': list,
                'container_info': dict,
                'confidence': 'high' | 'medium' | 'low'
            }
        """
        try:
            logger.info(f"Identifying product from photo ({len(image_bytes)} bytes)")
            
            # Compress image if needed
            image_bytes = self._compress_image(image_bytes)
            
            # Encode image
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            prompt = """
Analyze this product photo and extract the following information from what is VISIBLE on the label:

1. PRODUCT NAME: The exact product name as shown on the package
2. BRAND: The brand/manufacturer name
3. CATEGORY: Product category (e.g., Body Wash, Shampoo, Lotion, Food, Beverage, etc.)
4. PRODUCT TYPE: Specific type (e.g., Moisturizing Body Wash, Anti-Dandruff Shampoo, etc.)
5. MARKETING CLAIMS: Any claims visible on front (e.g., "Natural", "Organic", "Moisturizing", "pH Balanced")
6. CERTIFICATIONS: Any certification logos or labels visible (e.g., USDA Organic, Cruelty-Free, Vegan, etc.)
7. CONTAINER: Material and type (e.g., "Plastic bottle", "Glass jar", "Aluminum can")
8. SIZE/QUANTITY: If visible (e.g., "16 fl oz", "500 mL", "200g")
9. WARNINGS/ALLERGENS: Any warnings or allergen info visible on front

Format your response EXACTLY as follows:
PRODUCT_NAME: [exact name]
BRAND: [brand name]
CATEGORY: [category]
PRODUCT_TYPE: [type]
MARKETING_CLAIMS: [claim 1], [claim 2], [claim 3]
CERTIFICATIONS: [cert 1], [cert 2]
CONTAINER_MATERIAL: [material]
CONTAINER_TYPE: [type]
SIZE: [size if visible]
WARNINGS: [any warnings visible]

If any field is not visible, write "Not visible" for that field.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a product identification expert. Extract only visible information from the image. Do not make assumptions."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # low detail is 4x faster, sufficient for product identification
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,  # reduced from 800 for faster responses
                temperature=0.1
            )
            
            text = response.choices[0].message.content or ""
            logger.info(f"Product identification response: {text[:200]}...")
            
            # Parse the response
            result = self._parse_product_identification(text)
            
            logger.info(f"Product identified: {result.get('brand')} {result.get('product_name')}")
            return result
            
        except Exception as e:
            logger.error(f"Product identification failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _parse_product_identification(self, text: str) -> Dict[str, Any]:
        """Parse product identification response"""
        lines = text.strip().split("\n")
        result = {
            'product_name': '',
            'brand': '',
            'category': '',
            'product_type': '',
            'marketing_claims': [],
            'certifications_visible': [],
            'container_info': {
                'material': '',
                'type': '',
                'size': ''
            },
            'warnings': '',
            'confidence': 'medium'
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if value.lower() == 'not visible':
                    value = ''
                
                if key == 'PRODUCT_NAME':
                    result['product_name'] = value
                elif key == 'BRAND':
                    result['brand'] = value
                elif key == 'CATEGORY':
                    result['category'] = value
                elif key == 'PRODUCT_TYPE':
                    result['product_type'] = value
                elif key == 'MARKETING_CLAIMS':
                    result['marketing_claims'] = [c.strip() for c in value.split(',') if c.strip()]
                elif key == 'CERTIFICATIONS':
                    result['certifications_visible'] = [c.strip() for c in value.split(',') if c.strip()]
                elif key == 'CONTAINER_MATERIAL':
                    result['container_info']['material'] = value
                elif key == 'CONTAINER_TYPE':
                    result['container_info']['type'] = value
                elif key == 'SIZE':
                    result['container_info']['size'] = value
                elif key == 'WARNINGS':
                    result['warnings'] = value
        
        # Determine confidence based on completeness
        fields_filled = sum([
            bool(result['product_name']),
            bool(result['brand']),
            bool(result['category'])
        ])
        
        if fields_filled >= 3:
            result['confidence'] = 'high'
        elif fields_filled >= 2:
            result['confidence'] = 'medium'
        else:
            result['confidence'] = 'low'
        
        return result


_vision_service: VisionService | None = None

def get_vision_service() -> VisionService:
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
