from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import logging
import re
import numpy as np
import requests
from io import BytesIO

from models import get_clip_embedder
from services import get_pinecone_service
from services.barcode_service import get_barcode_service
from services.vision_service import get_vision_service
from services.chemical_checker import check_ingredients, calculate_safety_score, generate_recommendations

logger = logging.getLogger(__name__)
router = APIRouter()


class BarcodeLookupRequest(BaseModel):
    """Request model for barcode lookup"""
    barcode: str


def normalize_ingredient_text(text: str) -> str:
    """
    Normalize ingredient text for better chemical matching.
    Handles British spelling, prefixes, and E-number variations.
    
    Args:
        text: Raw ingredient text
        
    Returns:
        Normalized text for better pattern matching
    """
    if not text:
        return ""
    
    # 1. Strip parenthetical descriptions (e.g., "POTASSIUM SORBATE (PRESERVATIVE)" → "POTASSIUM SORBATE")
    # This prevents duplicates where same ingredient appears with/without parentheses
    text = re.sub(r'\s*\([^)]+\)', '', text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. British → American spelling
    text = text.replace("colour", "color")
    text = text.replace("flavour", "flavor")
    text = text.replace("sulphur", "sulfur")
    
    # 4. Normalize color numbering: "red no. 40" → "red 40", "yellow no 5" → "yellow 5"
    text = re.sub(r'\b(red|yellow|blue|green)\s+no\.?\s+(\d+)', r'\1 \2', text)
    
    # 5. Remove common prefixes that might block matching
    text = re.sub(r'\bartificial\s+', '', text)
    text = re.sub(r'\bsynthetic\s+', '', text)
    text = re.sub(r'\bnatural\s+', '', text)
    
    # 5. Normalize E-numbers: ensure space after 'e' for consistency
    # This helps "e102" match "e 102" in the database
    text = re.sub(r'\be(\d{3,4})\b', r'e\1', text)
    
    # 6. Normalize common separators
    text = text.replace(' - ', ', ')
    text = text.replace('; ', ', ')
    
    return text


async def infer_ingredients_from_context(
    product_name: str,
    brand: str,
    category: str,
    image_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use AI to infer likely ingredients when OpenFacts data is missing or incomplete.
    Returns educated guesses based on product type, category, and industry standards.
    
    Args:
        product_name: Product name
        brand: Brand name
        category: Product category
        image_url: Optional product image URL
        
    Returns:
        Dictionary with inferred ingredients, packaging, and confidence level
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""Based on this product information, provide an educated analysis of likely ingredients and packaging.

Product: {product_name}
Brand: {brand or 'Unknown'}
Category: {category or 'General consumer product'}

Provide:
1. Most likely ingredients for this product type (5-10 common ones)
2. Potential harmful additives typically found in this category
3. Typical packaging materials used for this product type
4. Confidence level in these inferences

Be realistic and conservative. Base your analysis on typical industry formulations.

IMPORTANT: Return ONLY valid JSON in this exact format:
{{
  "likely_ingredients": ["ingredient1", "ingredient2", "ingredient3"],
  "potential_additives": ["additive1", "additive2"],
  "typical_packaging": ["plastic bottle", "cardboard box"],
  "confidence": "medium",
  "reasoning": "Brief explanation of why these ingredients are likely",
  "disclaimer": "Analysis based on typical products in this category. Not confirmed ingredient list."
}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert on consumer product formulations and ingredients. Provide realistic, conservative inferences based on industry standards. Always return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        inference_data = json.loads(content)
        logger.info(f"AI inference successful for {product_name}")
        return inference_data
        
    except Exception as e:
        logger.error(f"AI inference failed: {e}")
        return {
            "likely_ingredients": [],
            "potential_additives": [],
            "typical_packaging": [],
            "confidence": "low",
            "reasoning": "Unable to generate inference",
            "disclaimer": "Insufficient data for analysis"
        }


async def infer_packaging_from_category(
    product_name: str,
    category: str
) -> Optional[str]:
    """
    Infer likely packaging materials based on product category.
    
    Args:
        product_name: Product name
        category: Product category
        
    Returns:
        Inferred packaging material description or None
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""Based on typical industry standards, what packaging materials are MOST COMMONLY used for products in this category?

Product: {product_name}
Category: {category}

Provide ONLY a brief list of typical packaging materials (e.g., "Plastic wrapper, cardboard box"). Be realistic and conservative."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a packaging industry expert. Provide realistic, conservative estimates of typical packaging materials."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        
        packaging = response.choices[0].message.content.strip()
        logger.info(f"Inferred packaging from category: {packaging}")
        return packaging
        
    except Exception as e:
        logger.error(f"Packaging category inference failed: {e}")
        return None


async def search_packaging_info(
    product_name: str,
    brand: str,
    category: str
) -> Optional[str]:
    """
    Search for packaging information using OpenAI web search tool.
    
    Args:
        product_name: Product name
        brand: Brand name
        category: Product category
        
    Returns:
        Packaging material description or None
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
       
        
        # Use OpenAI Responses API with web_search tool
        response = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search"}],
            input=f"""Search for packaging materials used for '{brand} {product_name}'.

Find:
- Specific packaging materials (plastic pouch, cardboard box, glass jar, etc.)
- Plastic types (PET, HDPE, PP, etc.)
- Recyclability claims
- Environmental features (recycled content, biodegradable, etc.)

Return ONLY a comma-separated list of materials. If nothing found, return 'Not found'.

Example: "Plastic pouch (PP), Cardboard sleeve (recycled)" """
        )
        
        packaging_text = response.output_text.strip()
        logger.info(f"OpenAI web search result: {packaging_text}")
        
        # Filter out non-results
        if packaging_text.lower() in ['not found', 'unknown', 'none', 'n/a', 'no information']:
            return None
        
        return packaging_text
        
    except Exception as e:
        logger.error(f"OpenAI web search for packaging failed: {e}")
        return None


async def get_smart_recommendations(
    product_name: str,
    brand: str,
    harmful_chemicals: List[Dict[str, Any]],
    category: str
) -> Dict[str, Any]:
    """
    Get smart product recommendations using Pinecone vector search with AI fallback.
    
    Strategy:
    1. Try Pinecone vector DB for actual product alternatives (using WordPress service)
    2. Fallback to generic rule-based recommendations if search fails
    
    Args:
        product_name: Current product name
        brand: Current product brand
        harmful_chemicals: List of detected harmful chemicals
        category: Product category
        
    Returns:
        Dictionary with recommendations and metadata
    """
    try:
        # Try using WordPress service to get product recommendations
        # Note: This is a placeholder - in production, you'd query based on category/tags
        # For now, fallback to generic recommendations
        logger.info(f"Product recommendations requested for {product_name} in category {category}")
        
        # TODO: Implement WordPress product search by category/tags
        # This would require the WordPress API to support filtering by category
        
    except Exception as e:
        logger.warning(f"Product recommendation search failed: {e}")
    
    # Use generic rule-based recommendations
    generic_recs = generate_recommendations(harmful_chemicals, category)
    
    return {
        "type": "general_guidance",
        "recommendations": generic_recs,
        "source": "rule_based",
        "confidence": "medium",
        "message": "General safety recommendations based on detected chemicals"
    }


async def get_product_recommendations_for_barcode(
    product_data: Dict[str, Any],
    top_k: int = 10,  # Query more to allow filtering
    min_score: float = 0.3  # Lower threshold, we'll filter with AI
) -> Dict[str, Any]:
    """
    Get product recommendations for a barcode-scanned product using multimodal search + AI filtering.
    
    Strategy:
    1. Generate embeddings from BOTH image AND text (product name + category)
    2. Combine embeddings with weights (70% text, 30% image for better semantic matching)
    3. Query Pinecone for similar products
    4. Use AI to filter out irrelevant matches (e.g., don't recommend mattresses for candy)
    5. If no relevant matches: generate AI alternatives
    
    Args:
        product_data: Barcode product data with name, categories, image_url, etc.
        top_k: Number of candidates to retrieve (default: 10, will be filtered)
        min_score: Minimum similarity score threshold (default: 0.3)
        
    Returns:
        {
            "status": "success" | "partial" | "ai_fallback",
            "products": [list of relevant WordPress products],
            "ai_alternatives": [list of AI-generated alternatives],
            "message": "description"
        }
    """
    try:
        clip_embedder = get_clip_embedder()
        text_embedding = None
        image_embedding = None
        
        # Step 1: Generate text embedding (always try this first - most reliable)
        category = ""
        if product_data.get('categories'):
            category = product_data['categories'].split(',')[0].strip()
        
        product_name = product_data.get('name', '')
        brand = product_data.get('brand', '')
        text_query = f"{product_name} {brand} {category}".strip()
        
        logger.info(f"Generating text embedding for: {text_query}")
        text_embedding = clip_embedder.model.encode(text_query, convert_to_numpy=True)
        
        # Step 2: Try to get image embedding
        if product_data.get('image_url'):
            try:
                logger.info(f"Downloading product image: {product_data['image_url']}")
                response = requests.get(product_data['image_url'], timeout=10)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_embedding = clip_embedder.embed_image(image)
                logger.info("Successfully generated image embedding")
                
            except Exception as img_error:
                logger.warning(f"Failed to download/process image: {img_error}")
                image_embedding = None
        
        # Step 3: Combine embeddings (weighted: 70% text, 30% image)
        # Text is more reliable for category matching
        if text_embedding is not None and image_embedding is not None:
            combined_embedding = 0.7 * text_embedding + 0.3 * image_embedding
            # Normalize
            import numpy as np
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            embedding_method = "multimodal (70% text, 30% image)"
        elif text_embedding is not None:
            combined_embedding = text_embedding
            embedding_method = "text_only"
        else:
            # Fallback to AI alternatives if no embeddings
            logger.warning("No embeddings available, using AI fallback")
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=category,
                count=3
            )
            return {
                "status": "ai_fallback",
                "products": [],
                "ai_alternatives": ai_alternatives,
                "message": "No embeddings available. Showing AI-recommended alternatives.",
                "embedding_method": "none"
            }
        
        # Step 4: Query Pinecone
        pinecone_service = get_pinecone_service()
        candidates = pinecone_service.query_similar_products(
            query_embedding=combined_embedding,
            top_k=top_k,
            min_score=min_score
        )
        
        logger.info(f"Pinecone returned {len(candidates)} candidate products (method: {embedding_method})")
        
        if not candidates:
            # No candidates at all - use AI fallback
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=category,
                count=3
            )
            return {
                "status": "ai_fallback",
                "products": [],
                "ai_alternatives": ai_alternatives,
                "message": "No similar products found in Hippiekit database. Showing AI-recommended alternatives.",
                "embedding_method": embedding_method
            }
        
        # Step 5: AI-powered relevance filtering
        relevant_products = await filter_relevant_products(
            scanned_product_name=product_name,
            scanned_category=category,
            candidate_products=candidates,
            min_relevance_score=7  # Out of 10
        )
        
        logger.info(f"AI filtered to {len(relevant_products)} relevant products from {len(candidates)} candidates")
        
        # Step 6: Decide on final response
        if len(relevant_products) >= 2:
            # Success: Found enough relevant products
            return {
                "status": "success",
                "products": relevant_products[:3],  # Top 3 relevant
                "ai_alternatives": [],
                "message": f"Found {len(relevant_products)} relevant products from Hippiekit database",
                "embedding_method": embedding_method
            }
        
        elif len(relevant_products) == 1:
            # Partial: Found some but supplement with AI
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=category,
                count=2
            )
            return {
                "status": "partial",
                "products": relevant_products,
                "ai_alternatives": ai_alternatives,
                "message": f"Found {len(relevant_products)} relevant product. Supplemented with AI-recommended alternatives.",
                "embedding_method": embedding_method
            }
        
        else:
            # No relevant products - full AI fallback
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=category,
                count=3
            )
            return {
                "status": "ai_fallback",
                "products": [],
                "ai_alternatives": ai_alternatives,
                "message": "No relevant products found in Hippiekit database. Showing AI-recommended healthier alternatives.",
                "embedding_method": embedding_method
            }
        
    except Exception as e:
        logger.error(f"Error getting product recommendations: {e}", exc_info=True)
        
        # Final fallback: AI alternatives only
        try:
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_data.get('name', ''),
                category=product_data.get('categories', '').split(',')[0].strip() if product_data.get('categories') else '',
                count=3
            )
            
            return {
                "status": "ai_fallback",
                "products": [],
                "ai_alternatives": ai_alternatives,
                "message": "Error during search. Showing AI-recommended alternatives.",
                "embedding_method": "error"
            }
        except:
            return {
                "status": "error",
                "products": [],
                "ai_alternatives": [],
                "message": "Unable to generate recommendations at this time.",
                "embedding_method": "error"
            }


async def filter_relevant_products(
    scanned_product_name: str,
    scanned_category: str,
    candidate_products: List[Dict[str, Any]],
    min_relevance_score: int = 7
) -> List[Dict[str, Any]]:
    """
    Use AI to filter out irrelevant product recommendations.
    
    Example: If user scanned "Skittles candy", filter out "mattresses" and "straws"
    even if they have visual similarity.
    
    Args:
        scanned_product_name: Name of the scanned product
        scanned_category: Category of scanned product
        candidate_products: List of potential recommendations from Pinecone
        min_relevance_score: Minimum score (1-10) to keep a product
        
    Returns:
        Filtered list of relevant products only
    """
    if not candidate_products:
        return []
    
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Prepare candidate list for AI
        candidates_text = "\n".join([
            f"{i+1}. {p.get('name', 'Unknown')} (Category: {p.get('description', 'N/A')[:100]})"
            for i, p in enumerate(candidate_products)
        ])
        
        prompt = f"""You are evaluating product recommendations for a health-conscious consumer.

Scanned Product: "{scanned_product_name}"
Category: "{scanned_category}"

Candidate Recommendations:
{candidates_text}

Task: Rate each candidate's RELEVANCE as a healthier alternative (1-10 scale):
- 10 = Perfect match (same category, healthier version)
- 7-9 = Good match (related category, suitable alternative)
- 4-6 = Weak match (loosely related)
- 1-3 = Irrelevant (completely different product category)

Examples:
- Scanned "Skittles candy" → Recommend "Organic fruit gummies" (score: 10)
- Scanned "Skittles candy" → Recommend "Organic chocolate" (score: 8) 
- Scanned "Skittles candy" → Recommend "Reusable straw" (score: 1)
- Scanned "Skittles candy" → Recommend "Mattress topper" (score: 1)

Return ONLY a JSON array of relevance scores (1-10) for each candidate in order:
[10, 8, 1, 1, ...]

Be strict: Only food products can replace food. Don't recommend home goods for snacks."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a product categorization expert. Return only JSON arrays with no explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        scores = json.loads(content)
        
        # Filter products by relevance score
        relevant_products = []
        for i, product in enumerate(candidate_products):
            if i < len(scores) and scores[i] >= min_relevance_score:
                product['relevance_score'] = scores[i]
                relevant_products.append(product)
        
        # Sort by relevance score (highest first)
        relevant_products.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"AI relevance filtering: {len(relevant_products)}/{len(candidate_products)} products passed (min score: {min_relevance_score})")
        
        return relevant_products
        
    except Exception as e:
        logger.error(f"Error filtering products with AI: {e}", exc_info=True)
        # Fallback: return empty list (will trigger AI alternatives)
        return []


async def generate_ai_product_alternatives(
    product_name: str,
    category: str,
    count: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate AI-powered product alternatives with brand names when Pinecone search fails.
    
    Args:
        product_name: Name of the scanned product
        category: Product category
        count: Number of alternatives to generate
        
    Returns:
        List of AI-generated product recommendations:
        [
            {
                "name": "Product Name",
                "brand": "Brand Name",
                "description": "Why this is a healthier alternative",
                "type": "ai_generated"
            }
        ]
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""You are a holistic nutritionist and clean eating expert. A customer scanned "{product_name}" (category: {category}) and it contains harmful additives.

Recommend {count} REAL, SPECIFIC healthier alternative products with ACTUAL BRAND NAMES that they can buy instead.

CRITICAL RULES:
1. Use REAL brand names (e.g., "Amy's Organic", "Annie's Homegrown", "Nature's Path")
2. Focus on organic, plant-based, clean-label brands
3. Match the product category and use case
4. Explain WHY each alternative is healthier (no harmful additives, organic ingredients, etc.)
5. Be specific - not generic suggestions

Format as JSON array:
[
    {{
        "name": "Specific Product Name",
        "brand": "Actual Brand Name",
        "description": "1-2 sentences explaining why this is a healthier alternative - mention clean ingredients, certifications, etc."
    }}
]

Example (if product was "Skittles"):
[
    {{
        "name": "Organic Fruit Chews",
        "brand": "YumEarth",
        "description": "Made with real fruit extracts and organic ingredients. Free from artificial dyes, high fructose corn syrup, and synthetic additives. Certified organic and allergy-friendly."
    }},
    {{
        "name": "Fruity Snacks",
        "brand": "Annie's Homegrown",
        "description": "Uses natural fruit flavors and colors from vegetables and fruits. No synthetic dyes or artificial preservatives. Made with organic ingredients."
    }},
    {{
        "name": "Organic Gummy Bears",
        "brand": "Black Forest",
        "description": "Colored with natural fruit and vegetable juices instead of synthetic dyes. No high fructose corn syrup or artificial flavors."
    }}
]

Now generate {count} alternatives for "{product_name}" (category: {category}):"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a holistic nutritionist specializing in clean eating and organic products. You recommend specific real brands and products."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        alternatives = json.loads(content)
        
        # Add type marker
        for alt in alternatives:
            alt["type"] = "ai_generated"
        
        logger.info(f"Generated {len(alternatives)} AI product alternatives")
        return alternatives
        
    except Exception as e:
        logger.error(f"Failed to generate AI alternatives: {e}", exc_info=True)
        return []


def parse_nested_ingredients(ingredients_text: str) -> List[str]:
    """
    Parse ingredients text and extract ALL ingredients including nested ones in parentheses/brackets.
    
    Example:
        "ENRICHED WHEAT FLOUR (WHEAT, IRON, NIACIN), SUGAR" 
        → ["ENRICHED WHEAT FLOUR", "WHEAT", "IRON", "NIACIN", "SUGAR"]
    
    Args:
        ingredients_text: Raw comma-separated ingredients text
        
    Returns:
        List of all ingredients (parent + nested)
    """
    if not ingredients_text:
        return []
    
    all_ingredients = []
    
    # Strategy: Find all ingredients with parentheses FIRST (as complete units)
    # Pattern: "INGREDIENT (NESTED, STUFF, HERE)" - captures everything including nested commas
    parentheses_pattern = r'([^,]+\([^)]+\))'
    
    # Find all ingredients with parentheses
    ingredients_with_parens = re.findall(parentheses_pattern, ingredients_text)
    
    # Process each ingredient with parentheses
    for match in ingredients_with_parens:
        # Extract parent ingredient (before opening parenthesis)
        parent = re.split(r'[\(\[]', match)[0].strip()
        if parent:
            all_ingredients.append(parent)
        
        # Extract nested ingredients inside parentheses
        nested_match = re.search(r'[\(\[]([^\)\]]+)[\)\]]', match)
        if nested_match:
            nested_text = nested_match.group(1)
            # Split nested ingredients by comma or &
            nested_parts = re.split(r'[,&]', nested_text)
            for nested in nested_parts:
                nested = nested.strip()
                if nested:
                    all_ingredients.append(nested)
    
    # Remove all parenthesized ingredients from the original text to avoid double-processing
    cleaned_text = ingredients_text
    for match in ingredients_with_parens:
        cleaned_text = cleaned_text.replace(match, '')
    
    # Now split remaining text by comma to get simple ingredients
    simple_parts = [p.strip() for p in cleaned_text.split(',') if p.strip()]
    all_ingredients.extend(simple_parts)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ingredients = []
    for ing in all_ingredients:
        ing_clean = ing.strip().upper()
        if ing_clean and ing_clean not in seen:
            seen.add(ing_clean)
            unique_ingredients.append(ing.strip())
    
    logger.info(f"Parsed nested ingredients: {len(unique_ingredients)} total")
    return unique_ingredients


async def separate_ingredients_with_ai(
    ingredients_list: List[str],
    harmful_chemicals_db: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Use OpenAI to intelligently categorize ingredients as harmful or safe.
    
    CRITICAL: AI uses chemical database as REFERENCE only (not for alias matching).
    AI makes decisions based on:
    1. Whether ingredient is present in the database
    2. Its own nutritional/chemical knowledge
    
    Args:
        ingredients_list: Full list of ingredients (already parsed with nested ones extracted)
        harmful_chemicals_db: Database of harmful chemicals (REFERENCE ONLY - not for alias matching)
        
    Returns:
        {
            "harmful": ["Red 40", "Yellow 5", ...],
            "safe": ["Sugar", "Water", ...]
        }
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Extract chemical names from database as REFERENCE (not for matching)
        harmful_chemical_names = [chem['chemical'] for chem in harmful_chemicals_db]
        
        # Create prompt for AI categorization
        prompt = f"""You are a food safety expert analyzing product ingredients.

INGREDIENTS TO ANALYZE:
{', '.join(ingredients_list)}

HARMFUL CHEMICALS DATABASE (REFERENCE ONLY):
{', '.join(harmful_chemical_names[:150])}

TASK:
Categorize each ingredient as HARMFUL or SAFE using:
1. The chemical database as a reference (if ingredient is listed → harmful)
2. Your own nutritional and chemical knowledge

CRITICAL RULES:
- DO NOT try to match aliases or variations
- Use your AI knowledge to identify harmful ingredients even if not in database
- Synthetic dyes (Red 40, Yellow 5, Blue 1, etc.) → harmful
- Ultra-processed additives (maltodextrin, modified starch, HFCS) → harmful  
- Artificial preservatives, flavors → harmful
- Whole food ingredients (sugar, salt, water, wheat, etc.) → safe (even if processed)
- Return ingredient names EXACTLY as they appear in the product ingredient list
- Each ingredient appears in ONLY ONE list (either harmful OR safe, never both)

Output ONLY valid JSON:
{{
    "harmful": ["Ingredient names from product"],
    "safe": ["Ingredient names from product"]
}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at matching ingredient names against chemical databases, understanding name variations and aliases."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        logger.info(f"AI separated ingredients: {len(result.get('harmful', []))} harmful, {len(result.get('safe', []))} safe")
        
        return result
        
    except Exception as e:
        logger.error(f"AI ingredient separation failed: {e}")
        # Fallback to empty lists
        return {"harmful": [], "safe": ingredients_list}


def get_detailed_ingredient_descriptions(
    safe_ingredients: list, 
    harmful_ingredients: list,
    harmful_chemicals_db: list
) -> dict:
    """
    Generate detailed, user-friendly AI descriptions for ALREADY SEPARATED ingredients.
    
    CRITICAL: This function receives ingredients that have ALREADY been separated by AI.
    Do NOT re-separate them here.
    
    Args:
        safe_ingredients: List of safe ingredient names (from AI separation)
        harmful_ingredients: List of harmful ingredient names (from AI separation)
        harmful_chemicals_db: Database of harmful chemicals (for context in descriptions)
    
    Returns:
        {
            "safe": {"ingredient_name": "detailed description"},
            "harmful": {"chemical_name": "detailed harmful description"}
        }
    """
    try:
        # Build harmful descriptions - explain what it is AND why it's harmful
        harmful_descriptions = {}
        if harmful_ingredients:
            try:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                # Find database info for harmful ingredients
                harmful_items = []
                for ing_name in harmful_ingredients[:15]:  # Limit to avoid token limits
                    # Try to find this ingredient in the database for context
                    db_info = None
                    for chem in harmful_chemicals_db:
                        if chem['chemical'].lower() == ing_name.lower():
                            db_info = chem
                            break
                    
                    if db_info:
                        harmful_items.append(f"{ing_name} (Category: {db_info['category']}, Severity: {db_info['severity']})")
                    else:
                        harmful_items.append(f"{ing_name} (AI flagged as harmful)")
                
                harmful_str = ", ".join(harmful_items)
                
                harmful_prompt = f"""Analyze these harmful chemicals/ingredients found in a consumer product.

FLAGGED CHEMICALS (authoritative — do not dispute or soften):
{harmful_str}

For each chemical, provide a description that:
1. States what the chemical is and its industrial/synthetic use in products
2. Explains why it's harmful to human health and/or the environment from a whole-food, natural nutrition perspective
3. Calls out specific health risks, toxicity concerns, or environmental damage
4. Uses clear, direct language - no corporate apologetics or "generally regarded as safe" phrases

CRITICAL RULES:
• These ingredients are FLAGGED for a reason - do not defend them
• If it's synthetic, lab-derived, or industrially processed, emphasize that
• E-numbers should be called out as synthetic petroleum-based additives
• Dyes, artificial flavors, and preservatives are always problematic
• Never say "safe in small amounts" or cite regulatory approval as reassurance
• Assume worst-case origin (synthetic) unless explicitly stated as natural

Format as JSON: {{"Chemical Name Exactly As Listed": "critical description"}}

Harmful chemicals: {harmful_str}"""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are a strict, anti-synthetic, whole-foods-only nutritionist and ingredient auditor.

Your philosophy:
• If an ingredient is synthetic, industrially processed, artificial, or ambiguous — it is a red flag.
• You strongly favor whole, recognizable, minimally processed ingredients.
• You do NOT give manufacturers the benefit of the doubt.
• You NEVER reclassify or soften ingredients already flagged by the chemical detection system.
• "Natural flavor", "flavourings", E-numbers, dyes, modified starches, gums, emulsifiers, preservatives, or lab-derived compounds are treated as harmful unless explicitly whole-food derived.

Rules you MUST follow:
1. If an ingredient matches a flagged chemical, you MUST mark it as harmful and explain why.
2. If an ingredient is vague or non-specific (e.g. "flavourings", "colors"), assume it hides synthetic additives.
3. If an ingredient is technically allowed but highly processed, call it out as "industrial/ultra-processed".
4. Never say "generally regarded as safe" or cite regulatory approval as reassurance.
5. If an ingredient can be either natural or synthetic, assume it is synthetic unless the label explicitly states otherwise.
6. Favor short ingredient lists with recognizable whole foods.
7. If unsure, err on the side of caution and flag it.

Tone: Calm, grounded, honest, but uncompromising. You are a nutritionist who cares about real food, not industry profits.

Output must be valid JSON only. Always use the exact chemical names provided as JSON keys without modification."""
                        },
                        {"role": "user", "content": harmful_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                
                content = response.choices[0].message.content.strip()
                logger.info(f"Raw AI response for harmful chemicals: {content[:500]}...")
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                harmful_descriptions = json.loads(content)
                logger.info(f"Generated detailed descriptions for {len(harmful_descriptions)} harmful chemicals")
                
            except Exception as e:
                logger.error(f"Failed to generate harmful chemical descriptions: {e}", exc_info=True)
                # Fallback to basic descriptions
                for chem in harmful_chemicals:
                    harmful_descriptions[chem['chemical']] = f"This chemical has been flagged as {chem['severity'].lower()} concern due to its {chem['category'].lower()} properties. It may pose health and environmental risks."
        
        # Generate safe ingredient descriptions with strict whole-food perspective
        safe_descriptions = {}
        if safe_ingredients:
            try:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                # Process all ingredients (no limit)
                ingredients_str = ", ".join(safe_ingredients)
                logger.info(f"Generating descriptions for {len(safe_ingredients)} safe ingredients")
                
                safe_prompt = f"""Analyze these ingredients from a whole-food, natural nutrition perspective.

INGREDIENTS TO ANALYZE:
{ingredients_str}

For each ingredient, provide a description that:
1. States whether it's whole-food, minimally processed, ultra-processed, or synthetic
2. Explains its purpose/function in the product
3. Calls out if it's industrially processed, refined, or lab-derived
4. Notes any nutritional value or concerns from a hippie/natural perspective

CRITICAL RULES:
• If an ingredient is refined sugar, industrial syrup, or highly processed - call it out
• "Flavourings", "Modified Starch", "Maltodextrin" = ultra-processed red flags
• If it has a long chemical name or E-number, assume synthetic/industrial
• Palm oil/fat = environmental concern (deforestation)
• Glucose syrup = refined sugar (blood sugar spike concern)
• If an ingredient can be natural OR synthetic, assume synthetic unless label says otherwise
• Whole foods (fruits, vegetables, whole grains) get positive descriptions
• Industrial ingredients get honest critiques

Tone: Educational, honest, grounded in whole-food philosophy. Not alarmist, but don't sugarcoat industrial processing.

Format as JSON: {{"Ingredient Name Exactly As Listed": "honest description"}}

Ingredients: {ingredients_str}"""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are a strict, whole-foods-focused nutritionist analyzing ingredients with a hippie, natural-living philosophy.

Your perspective:
• Whole, unprocessed foods = good
• Refined, industrial, synthetic ingredients = problematic
• Ultra-processed ingredients deserve honest critique
• You do NOT give ultra-processed foods a free pass
• Sugar is sugar - refined industrial sweeteners are blood sugar bombs
• "Flavourings" = hidden synthetic chemicals until proven otherwise
• Modified starches, maltodextrins, syrups = industrial food engineering
• If something can be natural or synthetic, assume synthetic unless stated

You are NOT anti-food or fear-mongering. You are pro-real-food and anti-industrial-processing.

Classification framework:
• Whole food: recognizable plant/animal food (apple, chicken, rice)
• Minimally processed: extracted but still recognizable (olive oil, whole wheat flour)
• Ultra-processed: industrial transformation (maltodextrin, modified starch, glucose syrup)
• Synthetic: lab-created (artificial flavors, dyes, preservatives)

Tone: Educational, grounded, honest. Like a nutritionist who shops at farmers markets.

Output must be valid JSON only. Always use exact ingredient names as keys."""
                        },
                        {"role": "user", "content": safe_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2500
                )
                
                content = response.choices[0].message.content.strip()
                logger.info(f"Raw AI response for safe ingredients: {content[:500]}...")
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                safe_descriptions = json.loads(content)
                logger.info(f"Generated detailed descriptions for {len(safe_descriptions)} safe ingredients")
                
            except Exception as e:
                logger.error(f"Failed to generate safe ingredient descriptions: {e}", exc_info=True)
                # Create fallback descriptions for ALL ingredients
                safe_descriptions = {ing: "Common ingredient used in food and personal care products." for ing in safe_ingredients}
        
        return {
            "safe": safe_descriptions,
            "harmful": harmful_descriptions
        }
        
    except Exception as e:
        logger.error(f"Error in get_detailed_ingredient_descriptions: {e}")
        return {"safe": {}, "harmful": {}}


def analyze_packaging_material(packaging_text: str, packaging_tags: list) -> dict:
    """
    Analyze packaging materials for environmental and health impact.
    Considers: plastic types (BPA, microplastics), recyclability, environmental impact
    
    Returns:
        {
            "materials": ["material1", "material2"],
            "analysis": {
                "material_name": {
                    "description": "what it is",
                    "harmful": true/false,
                    "health_concerns": "BPA leaching, etc.",
                    "environmental_impact": "recyclability, pollution, etc.",
                    "severity": "low/moderate/high/critical"
                }
            },
            "overall_safety": "safe/caution/harmful",
            "summary": "overall packaging assessment"
        }
    """
    try:
        if not packaging_text and not packaging_tags:
            return {
                "materials": [],
                "analysis": {},
                "overall_safety": "unknown",
                "summary": "No packaging information available"
            }
        
        # Combine packaging info
        packaging_info = f"Packaging: {packaging_text}. Tags: {', '.join(packaging_tags)}" if packaging_tags else packaging_text
        
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""Analyze the following product packaging materials for health and environmental impact.

Product: {packaging_info}

CRITICAL RULES:
1. DO NOT use uncertainty phrases like "specific type unknown", "unknown", "unclear", "may contain", "could be"
2. Each packaging material gets its own complete, detailed analysis
3. Describe how THIS BRAND uses THIS SPECIFIC material for THIS PRODUCT
4. Be authoritative and fact-based

For EACH packaging material, provide:

**Description**: 
- How this brand uses this material for this product (e.g., "The brand packages this product in a flexible stand-up pouch with resealable zipper for convenient storage")
- What this material is made of and its properties
- Why manufacturers choose this packaging type

**Health Concerns**:
- Specific chemicals and additives present in this material type
- Leaching risks and food contact concerns
- Microplastic generation
- Endocrine disruptors or toxins
- State definitive risks, not possibilities

**Environmental Impact**:
- Recyclability status (be specific - is it actually recycled or just "recyclable"?)
- Biodegradability timeframe
- Ocean and landfill pollution contribution
- Carbon footprint and manufacturing impact
- End-of-life disposal challenges

**Safety Rating**: 
- Severity: low/moderate/high/critical
- Overall harmful: true/false

TONE: Authoritative, specific to this brand and product. State facts definitively.

Example (DO):
"The brand packages this licorice in a flexible plastic stand-up pouch with resealable zipper. This multi-layer plastic typically contains polyethylene and polypropylene layers bonded together. Manufacturers choose this format for shelf stability and consumer convenience."

Example (DON'T):
"Stand-up pouches are commonly used for snacks."

Format as JSON:

IMPORTANT: The material names in the "materials" array MUST EXACTLY MATCH the keys in the "analysis" object. Use the same format for both (lowercase with underscores, e.g., "plastic_bag", "stand_up_pouch", "cardboard_box").

{{
    "materials": ["plastic_bag", "stand_up_pouch"],
    "analysis": {{
        "plastic_bag": {{
            "description": "detailed brand-specific description of how this material is used for this product, what it's made of, and why it's chosen",
            "harmful": true or false,
            "health_concerns": "specific health risks with definitive statements (no uncertainty)",
            "environmental_impact": "detailed environmental assessment with specific facts",
            "severity": "low/moderate/high/critical"
        }},
        "stand_up_pouch": {{
            "description": "...",
            "harmful": true or false,
            "health_concerns": "...",
            "environmental_impact": "...",
            "severity": "low/moderate/high/critical"
        }}
    }},
    "overall_safety": "safe/caution/harmful",
    "summary": "authoritative 2-3 sentence assessment of the overall packaging"
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an environmental scientist and toxicologist expert in packaging materials, plastic pollution, and consumer product safety."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        packaging_analysis = json.loads(content)
        logger.info(f"Generated packaging analysis for {len(packaging_analysis.get('materials', []))} materials")
        
        return packaging_analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze packaging material: {e}")
        return {
            "materials": [],
            "analysis": {},
            "overall_safety": "unknown",
            "summary": "Unable to analyze packaging materials at this time."
        }


@router.post("/scan")
async def scan_product(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Scan an image to find matching products.
    
    Args:
        image: Uploaded image file
        
    Returns:
        Dictionary with matching products and scan info
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Load image
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Generate embedding
        print(f"Generating embedding for uploaded image...")
        clip_embedder = get_clip_embedder()
        embedding = clip_embedder.embed_image(pil_image)
        
        # Search for similar products
        print(f"Searching for similar products...")
        pinecone_service = get_pinecone_service()
        products = pinecone_service.query_similar_products(
            query_embedding=embedding,
            top_k=5,
            min_score=0.6
        )
        
        return {
            'success': True,
            'matches_found': len(products),
            'products': products,
            'message': f'Found {len(products)} matching products' if products else 'No matching products found'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during scan: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/lookup-barcode")
async def lookup_barcode(request: BarcodeLookupRequest) -> Dict[str, Any]:
    """
    Look up a product by barcode with AI ingredient and packaging analysis.
    Uses Open*Facts databases + GPT-4o-mini for detailed descriptions.
    
    Args:
        request: Barcode lookup request containing the barcode string
        
    Returns:
        Dictionary with product information, ingredient descriptions, and packaging analysis
    """
    try:
        barcode = request.barcode.strip()
        
        # Validate barcode
        if not barcode:
            raise HTTPException(
                status_code=400,
                detail="Barcode cannot be empty"
            )
        
        # Only allow numeric barcodes (UPC, EAN, etc.)
        if not barcode.isdigit():
            raise HTTPException(
                status_code=400,
                detail="Barcode must contain only digits"
            )
        
        logger.info(f"Looking up barcode: {barcode}")
        
        # Query barcode service
        barcode_service = get_barcode_service()
        product_data = await barcode_service.lookup_barcode(barcode)
        
        if not product_data:
            return {
                'success': True,
                'found': False,
                'product': None,
                'message': 'Product not found in database. Try scanning the product image instead.'
            }
        
        # Extract ingredients for analysis
        # Try to get ingredients_text first, if not available, extract from ingredients array
        ingredients_text = product_data.get("ingredients_text", "")
        
        # If ingredients_text is empty, try to extract from ingredients array
        if not ingredients_text and "ingredients" in product_data:
            ingredients_array = product_data["ingredients"]
            # First, try to find the full_text ingredient
            for ing in ingredients_array:
                if isinstance(ing, dict) and ing.get("type") == "full_text":
                    ingredients_text = ing.get("text", "")
                    break
            
            # If still no text, combine all structured ingredient texts
            if not ingredients_text:
                ingredient_texts = []
                for ing in ingredients_array:
                    if isinstance(ing, dict) and ing.get("type") == "structured":
                        text = ing.get("text") or ing.get("id", "").replace("en:", "").replace("-", " ")
                        if text:
                            ingredient_texts.append(text)
                ingredients_text = ", ".join(ingredient_texts)
        
        logger.info(f"Extracted ingredients text: {ingredients_text[:200]}...")
        
        # Parse ingredients including nested ones in parentheses/brackets
        ingredients_list = parse_nested_ingredients(ingredients_text) if ingredients_text else []
        logger.info(f"Parsed {len(ingredients_list)} ingredients (including nested)")
        
        # === TRACK DATA SOURCES FOR METADATA ===
        ingredients_source = "openfacts" if ingredients_text else "none"
        packaging_source = "openfacts"
        inference_used = False
        inference_data = None
        
        # === AI INFERENCE FOR MISSING DATA ===
        # If ingredients are missing or very minimal, use AI inference
        if not ingredients_text or len(ingredients_text.strip()) < 20:
            logger.info(f"Minimal/missing ingredient data - attempting AI inference")
            inference_data = await infer_ingredients_from_context(
                product_name=product_data.get("name", "Unknown"),
                brand=product_data.get("brands", ""),
                category=product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General",
                image_url=product_data.get("image_url")
            )
            
            if inference_data and inference_data.get("likely_ingredients"):
                # Combine inferred ingredients with any existing data
                inferred_ingredients = ", ".join(inference_data["likely_ingredients"])
                if inference_data.get("potential_additives"):
                    inferred_ingredients += ", " + ", ".join(inference_data["potential_additives"])
                
                ingredients_text = ingredients_text + ", " + inferred_ingredients if ingredients_text else inferred_ingredients
                ingredients_source = "ai_inferred"
                inference_used = True
                logger.info(f"AI inference added {len(inference_data['likely_ingredients'])} likely ingredients")
        
        # === NORMALIZE INGREDIENTS FOR BETTER MATCHING ===
        normalized_ingredients_text = normalize_ingredient_text(ingredients_text) if ingredients_text else ""
        
        # Parse ingredients and ensure all are strings
        ingredients_list = []
        for ing in ingredients_text.split(","):
            ing = ing.strip()
            if ing:
                # Ensure it's a string (in case of any unexpected data types)
                ingredients_list.append(str(ing))
        
        logger.info(f"Parsed {len(ingredients_list)} ingredients from text")
        
        # Check ingredients for harmful chemicals using NORMALIZED text
        # Note: check_ingredients expects a string, not a list
        harmful_chemicals = []
        if normalized_ingredients_text:
            try:
                # Use normalized text for better chemical detection
                harmful_chemicals = check_ingredients(normalized_ingredients_text)
                logger.info(f"Found {len(harmful_chemicals)} harmful chemicals")
            except Exception as e:
                logger.error(f"Error checking ingredients: {e}")
                # Continue without chemical analysis rather than failing
        
        # Calculate safety score
        safety_score = calculate_safety_score(harmful_chemicals)
        
        # === SMART RECOMMENDATIONS WITH PINECONE ===
        recommendations = await get_smart_recommendations(
            product_name=product_data.get("name", "Unknown"),
            brand=product_data.get("brands", ""),
            harmful_chemicals=harmful_chemicals,
            category=product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General"
        )
        
        # Build chemical analysis
        product_data["chemical_analysis"] = {
            "harmful_chemicals": harmful_chemicals,
            "safety_score": safety_score,
            "recommendations": recommendations
        }
        
        # Generate detailed AI ingredient descriptions (safe + harmful)
        if ingredients_list:
            logger.info(f"Separating {len(ingredients_list)} ingredients with AI intelligent matching")
            
            # Use AI to intelligently separate harmful vs safe (handles name variations automatically)
            separated = await separate_ingredients_with_ai(ingredients_list, harmful_chemicals)
            
            harmful_ingredient_names = separated.get("harmful", [])
            safe_ingredient_names = separated.get("safe", [])
            
            logger.info(f"AI separation complete: {len(harmful_ingredient_names)} harmful, {len(safe_ingredient_names)} safe")
            
            # Generate detailed descriptions using AI-separated lists
            ingredient_descriptions = get_detailed_ingredient_descriptions(
                safe_ingredients=safe_ingredient_names,
                harmful_ingredients=harmful_ingredient_names,
                harmful_chemicals_db=harmful_chemicals
            )
            product_data["ingredient_descriptions"] = ingredient_descriptions
            
            # Store separated lists (no duplicates - AI handles the separation)
            product_data["ingredients"] = safe_ingredient_names  # Only safe ingredients
            product_data["harmful_ingredients"] = harmful_ingredient_names  # Only harmful ingredients
        else:
            product_data["ingredient_descriptions"] = {"safe": {}, "harmful": {}}
            product_data["ingredients"] = []
            product_data["harmful_ingredients"] = []
        
        # Analyze packaging materials with AI
        # Extract packaging from the correct location (might be in materials object)
        packaging_text = product_data.get("packaging", "")
        packaging_tags = product_data.get("packaging_tags", [])
        
        # Check if packaging info is nested in materials object
        if not packaging_text and "materials" in product_data:
            materials = product_data["materials"]
            packaging_text = materials.get("packaging_text", "") or materials.get("packaging", "")
            packaging_tags = materials.get("packaging_tags", [])
        
        logger.info(f"Packaging text: '{packaging_text}', Tags: {packaging_tags}")
        
        # === INFER PACKAGING IF MISSING (INDEPENDENT OF INGREDIENT INFERENCE) ===
        if not packaging_text and not packaging_tags:
            logger.info("No packaging data in OpenFacts - attempting to find packaging info")
            
            # Strategy 1: Try web search first (most accurate)
            web_packaging = await search_packaging_info(
                product_name=product_data.get("name", "Unknown"),
                brand=product_data.get("brands", ""),
                category=product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General"
            )
            
            if web_packaging:
                packaging_text = web_packaging
                packaging_source = "web_search"
                logger.info(f"Using web-searched packaging: {packaging_text}")
            
            # Strategy 2: Use AI inference from earlier if available
            elif inference_data and inference_data.get("typical_packaging"):
                packaging_text = ", ".join(inference_data["typical_packaging"])
                packaging_source = "ai_inferred"
                logger.info(f"Using AI-inferred packaging: {packaging_text}")
            
            # Strategy 3: AI inference based on category (always provide something)
            else:
                category = product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General product"
                packaging_inference = await infer_packaging_from_category(
                    product_name=product_data.get("name", "Unknown"),
                    category=category
                )
                if packaging_inference:
                    packaging_text = packaging_inference
                    packaging_source = "ai_inferred"
                    logger.info(f"Using category-based packaging inference: {packaging_text}")
        
        if packaging_text or packaging_tags:
            logger.info(f"Analyzing packaging materials: {packaging_text}")
            packaging_analysis = analyze_packaging_material(packaging_text, packaging_tags)
            product_data["packaging_analysis"] = packaging_analysis
            logger.info(f"Packaging analysis complete: {len(packaging_analysis.get('materials', []))} materials identified")
        else:
            logger.info("No packaging information available")
            packaging_source = "none"
            product_data["packaging_analysis"] = {
                "materials": [],
                "analysis": {},
                "overall_safety": "unknown",
                "summary": "No packaging information available"
            }
        
        # === ADD ANALYSIS METADATA FOR TRANSPARENCY ===
        data_quality_score = 100
        disclaimers = []
        
        # Reduce quality score based on inferred data
        if ingredients_source == "ai_inferred":
            data_quality_score -= 30
            disclaimers.append("Ingredients inferred using AI based on similar products in this category")
        elif ingredients_source == "none":
            data_quality_score -= 50
            disclaimers.append("No ingredient data available - analysis limited")
        
        if packaging_source == "web_search":
            data_quality_score -= 10
            disclaimers.append("Packaging materials found via web search - may not be 100% accurate")
        elif packaging_source == "ai_inferred":
            data_quality_score -= 20
            disclaimers.append("Packaging materials inferred based on product category")
        elif packaging_source == "none":
            data_quality_score -= 20
        
        # Add confidence level based on data quality
        if data_quality_score >= 80:
            confidence_level = "high"
        elif data_quality_score >= 50:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Add inference disclaimer if used
        if inference_used and inference_data:
            disclaimers.append(inference_data.get("disclaimer", ""))
        
        product_data["analysis_metadata"] = {
            "ingredients_source": ingredients_source,
            "packaging_source": packaging_source,
            "recommendations_source": recommendations.get("source", "rule_based"),
            "confidence_level": confidence_level,
            "data_quality_score": max(0, data_quality_score),
            "inference_used": inference_used,
            "disclaimers": disclaimers,
            "analysis_timestamp": "2025-12-24"
        }
        
        logger.info(f"Successfully processed barcode {barcode}: {product_data.get('name', 'Unknown')} (Quality: {data_quality_score}/100, Confidence: {confidence_level})")
        
        return {
            'success': True,
            'found': True,
            'product': product_data,
            'message': f'Product found: {product_data.get("name", "Unknown")}'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during barcode lookup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error looking up barcode: {str(e)}"
        )


@router.post("/barcode/recommendations")
async def get_barcode_recommendations(request: BarcodeLookupRequest) -> Dict[str, Any]:
    """
    Get product recommendations for a barcode-scanned product (async endpoint).
    This is called separately after the main barcode lookup to avoid slowing down initial results.
    
    Args:
        request: Barcode lookup request with barcode string
        
    Returns:
        {
            "success": true,
            "recommendations": {
                "status": "success" | "partial" | "ai_fallback",
                "products": [...],
                "ai_alternatives": [...],
                "message": "..."
            }
        }
    """
    try:
        barcode_service = get_barcode_service()
        
        # Fetch product data from barcode (await async function)
        product_data = await barcode_service.lookup_barcode(request.barcode)
        
        if not product_data:
            return {
                'success': False,
                'recommendations': None,
                'message': 'Product not found'
            }
        
        # Get recommendations
        recommendations = await get_product_recommendations_for_barcode(
            product_data=product_data,
            top_k=3,
            min_score=0.4  # Lower threshold for barcode products
        )
        
        return {
            'success': True,
            'recommendations': recommendations,
            'message': 'Recommendations generated successfully'
        }
        
    except Exception as e:
        logger.error(f"Error getting barcode recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting recommendations: {str(e)}"
        )


@router.post("/scan-vision")
async def scan_product_vision(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze a product image using OpenAI Vision to extract visible text, ingredients,
    packaging information, and provide health/eco recommendations.
    """
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Received vision scan request: {image.filename}, content_type: {image.content_type}")
        
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        logger.info("Reading image bytes...")
        image_bytes = await image.read()
        logger.info(f"Image bytes read: {len(image_bytes)} bytes")

        # Analyze via Vision service
        logger.info("Getting Vision service instance...")
        vision_service = get_vision_service()
        
        logger.info("Starting image analysis...")
        analysis = vision_service.analyze_product_image(image_bytes)

        if analysis is None:
            logger.error("Vision analysis returned None")
            return {
                "success": False,
                "message": "Vision analysis failed. Check server logs for details.",
                "analysis": None,
            }

        logger.info("Vision analysis completed successfully")
        return {
            "success": True,
            "analysis": analysis,
            "message": "Vision analysis completed",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during vision scan: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error analyzing image: {type(e).__name__}: {str(e)}"
        )
