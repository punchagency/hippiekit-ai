from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
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
import asyncio

from models import get_clip_embedder
from services import get_pinecone_service
from services.barcode_service import get_barcode_service
from services.vision_service import get_vision_service
from services.chemical_checker import check_ingredients, calculate_safety_score, generate_recommendations, load_harmful_chemicals_db

logger = logging.getLogger(__name__)
router = APIRouter()


class BarcodeLookupRequest(BaseModel):
    """Request model for barcode lookup"""
    barcode: str
    product_data: Optional[Dict[str, Any]] = None  # Optional pre-fetched product data to avoid redundant API calls


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


async def search_ingredients_from_context(
    product_name: str,
    brand: str,
    category: str,
    image_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use OpenAI web search to find actual product ingredients.
    Works for all consumer products: food, skincare, pet, kitchen, bathroom, etc.
    
    Args:
        product_name: Product name
        brand: Brand name
        category: Product category
        image_url: Optional product image URL
        
    Returns:
        Dictionary with actual ingredients found via web search
        Format: {"likely_ingredients": ["ingredient1", "ingredient2", ...]}
    """
    try:
        # Import web search service
        from services.web_search_service import web_search_service
        
        logger.info(f"[INGREDIENT SEARCH] Searching web for: {brand} {product_name}")
        
        # Use web search to find actual ingredients
        search_result = await web_search_service.search_product_ingredients(
            product_name=product_name,
            brand=brand,
            category=category
        )
        
        if search_result and search_result.get('ingredients'):
            # Parse the comma-separated ingredients into a list
            ingredients_text = search_result['ingredients']
            
            # Split by comma and clean up each ingredient
            ingredients_list = [
                ing.strip() 
                for ing in ingredients_text.split(',') 
                if ing.strip()
            ]
            
            logger.info(f"[INGREDIENT SEARCH] Found {len(ingredients_list)} ingredients from {search_result.get('source', 'unknown source')}")
            
            return {
                "likely_ingredients": ingredients_list
            }
        
        logger.warning(f"[INGREDIENT SEARCH] No ingredients found for {brand} {product_name}")
        return {
            "likely_ingredients": []
        }
        
    except Exception as e:
        logger.error(f"[INGREDIENT SEARCH] Failed: {e}")
        return {
            "likely_ingredients": []
        }


async def infer_ingredients_from_category(
    product_name: str,
    brand: str,
    category: str
) -> Dict[str, Any]:
    """
    Infer ingredients for a product using AI knowledge (NO web search).
    Returns ONLY high-confidence ingredients to avoid hallucination.
    
    Args:
        product_name: Product name
        brand: Brand name
        category: Product category
        
    Returns:
        Dictionary with AI-inferred ingredients (high and medium confidence ones)
        Format: {"likely_ingredients": ["ingredient1", "ingredient2", ...], "confidence": "high|medium"}
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""What are the ingredients in this product? Rate confidence PER INGREDIENT.

Product: {product_name}
Brand: {brand}
Category: {category}

INSTRUCTIONS:
1. For each ingredient, rate your confidence: "high", "medium", or "low"
2. "high" = You have definitive knowledge from training data (well-known branded products, official formulations)
3. "medium" = You're reasonably confident based on typical formulations for this product type
4. "low" = You're guessing with little certainty
5. Be honest about your confidence level for each ingredient

Return JSON format:
{{"ingredients": [{{"name": "ingredient", "confidence": "high|medium|low"}}]}}

Examples:
- Quaker Oats → {{"ingredients": [{{"name": "Whole Grain Rolled Oats", "confidence": "high"}}]}}
- Nutella → {{"ingredients": [{{"name": "Sugar", "confidence": "high"}}, {{"name": "Palm Oil", "confidence": "high"}}, {{"name": "Hazelnuts", "confidence": "high"}}]}}
- Generic chocolate bar → {{"ingredients": [{{"name": "Sugar", "confidence": "medium"}}, {{"name": "Cocoa", "confidence": "medium"}}, {{"name": "Milk", "confidence": "medium"}}]}}
- Unknown local brand → {{"ingredients": [{{"name": "Sugar", "confidence": "low"}}, {{"name": "Flour", "confidence": "low"}}]}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a product ingredient expert. Rate each ingredient's confidence honestly: 'high' for ingredients you KNOW are in the product, 'medium' for typical ingredients based on product type, 'low' for guesses. Return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Zero temperature for consistent, factual responses
            max_tokens=600,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            result = json.loads(content)
            ingredients_data = result.get("ingredients", [])
            
            # Filter to high and medium confidence ingredients
            accepted_ingredients = []
            low_confidence_count = 0
            highest_confidence = "medium"  # Track the best confidence level found
            
            for item in ingredients_data:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    conf = item.get("confidence", "low")
                    if conf in ["high", "medium"] and name:
                        accepted_ingredients.append(name)
                        if conf == "high":
                            highest_confidence = "high"
                    else:
                        low_confidence_count += 1
                elif isinstance(item, str):
                    # Fallback if AI returns simple strings (treat as low confidence)
                    low_confidence_count += 1
            
            if accepted_ingredients:
                logger.info(f"[INGREDIENT INFERENCE] Found {len(accepted_ingredients)} HIGH/MEDIUM confidence ingredients for {brand} {product_name} (filtered out {low_confidence_count} low confidence)")
                return {
                    "likely_ingredients": accepted_ingredients,
                    "confidence": highest_confidence
                }
            else:
                logger.info(f"[INGREDIENT INFERENCE] No HIGH/MEDIUM confidence ingredients for {brand} {product_name} ({low_confidence_count} low confidence filtered out) - will try web search")
                return {
                    "likely_ingredients": [],
                    "confidence": "low"
                }
        except json.JSONDecodeError as e:
            logger.error(f"[INGREDIENT INFERENCE] Failed to parse JSON: {e}")
            return {
                "likely_ingredients": [],
                "confidence": "low"
            }
        
    except Exception as e:
        logger.error(f"[INGREDIENT INFERENCE] Failed: {e}")
        return {
            "likely_ingredients": [],
            "confidence": "low"
        }


async def infer_packaging_from_category(
    product_name: str,
    category: str
) -> Optional[List[str]]:
    """
    Infer likely packaging materials based on product category.
    
    Args:
        product_name: Product name
        category: Product category
        
    Returns:
        List of inferred packaging materials or None
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""Based on typical industry standards, what packaging materials are MOST COMMONLY used for products in this category?

Product: {product_name}
Category: {category}

Return ONLY a JSON array of packaging material names. Be realistic and conservative.
Example: ["Plastic wrapper", "Cardboard box", "Aluminum foil"]

Limit to 3-4 most common materials."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a packaging industry expert. Return only valid JSON arrays of packaging materials."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            # Try direct JSON parse
            result = json.loads(content)
            # Handle if wrapped in object like {"materials": [...]}
            if isinstance(result, dict):
                materials = result.get("materials", list(result.values())[0] if result else [])
            else:
                materials = result
                
            if isinstance(materials, list) and materials:
                logger.info(f"Inferred packaging from category: {materials}")
                return materials
            else:
                logger.warning(f"Invalid packaging inference format: {content}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse packaging inference JSON: {e}")
            return None
        
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


async def get_product_recommendations_with_image(
    product_name: str,
    brand: str = "",
    category: str = "",
    ingredients: str = "",
    marketing_claims: str = "",
    certifications: str = "",
    product_type: str = "",
    image = None,  # PIL Image object
    top_k: int = 10,
    min_score: float = 0.3
) -> Dict[str, Any]:
    """
    Get product recommendations using multimodal search (text + image) with rich OCR data.
    Combines text embedding (70%) with image embedding (30%) for better matching.
    Uses ALL OCR-extracted data for semantic similarity matching.
    
    Args:
        product_name: Product name
        brand: Brand name (optional)
        category: Product category (optional)
        ingredients: Comma-separated ingredients text from OCR
        marketing_claims: Marketing claims (e.g., "Organic, Non-GMO")
        certifications: Certifications visible on package
        product_type: Product type classification
        image: PIL Image object of the scanned product (optional)
        top_k: Number of candidates to retrieve
        min_score: Minimum similarity score threshold
        
    Returns:
        {
            "status": "success" | "partial" | "ai_fallback",
            "products": [...],
            "ai_alternatives": [...],
            "message": "..."
        }
    """
    try:
        clip_embedder = get_clip_embedder()
        text_embedding = None
        image_embedding = None
        
        # Step 1: Build rich text query from all available OCR data
        text_parts = [product_name, brand, category, product_type]
        
        # Add marketing claims (strong semantic signals)
        if marketing_claims:
            text_parts.append(marketing_claims)
        
        # Add certifications (important for health-conscious matching)
        if certifications:
            text_parts.append(certifications)
        
        # Add first 10 ingredients to avoid token limits while preserving key composition info
        if ingredients:
            ingredient_list = [ing.strip() for ing in ingredients.split(',')[:10]]
            text_parts.append(' '.join(ingredient_list))
        
        text_query = ' '.join([p for p in text_parts if p]).strip()
        logger.info(f"Rich OCR text query ({len(text_query)} chars): {text_query[:150]}...")
        
        logger.info(f"Generating text embedding for: {text_query[:100]}")
        text_embedding = clip_embedder.model.encode(text_query, convert_to_numpy=True)
        
        # Step 2: Generate image embedding if image provided
        if image is not None:
            try:
                logger.info("Generating image embedding from provided photo")
                # Ensure image is in RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_embedding = clip_embedder.embed_image(image)
                logger.info("Successfully generated image embedding")
            except Exception as img_error:
                logger.warning(f"Failed to generate image embedding: {img_error}")
                image_embedding = None
        
        # Step 3: Combine embeddings (weighted: 70% text, 30% image)
        if text_embedding is not None and image_embedding is not None:
            import numpy as np
            combined_embedding = 0.7 * text_embedding + 0.3 * image_embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            embedding_method = "multimodal (70% text, 30% image)"
        elif text_embedding is not None:
            combined_embedding = text_embedding
            embedding_method = "text_only"
        else:
            # Fallback to AI alternatives
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
            min_relevance_score=7
        )
        
        logger.info(f"AI filtered to {len(relevant_products)} relevant products from {len(candidates)} candidates")
        
        # Step 6: Decide on final response
        if len(relevant_products) >= 2:
            return {
                "status": "success",
                "products": relevant_products[:3],
                "ai_alternatives": [],
                "message": f"Found {len(relevant_products)} relevant products from Hippiekit database",
                "embedding_method": embedding_method
            }
        elif len(relevant_products) == 1:
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
        logger.error(f"Error in get_product_recommendations_with_image: {e}", exc_info=True)
        
        # Final fallback: AI alternatives only
        try:
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=category,
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
        # Extract comprehensive product context for better embedding
        categories_text = ""
        if product_data.get('categories'):
            # Handle both string and list formats
            if isinstance(product_data['categories'], str):
                categories_text = product_data['categories']
            else:
                categories_text = ', '.join(product_data['categories'])
        
        product_name = product_data.get('name', '')
        brand = product_data.get('brands', '')
        
        # Optionally include product description/ingredients for richer context
        product_description = product_data.get('ingredients_text', '')[:200]  # First 200 chars
        
        # Create rich text query combining multiple signals
        # Format: "[Product Name] [Brand] [Categories] [Description snippet]"
        text_query = f"{product_name} {brand} {categories_text} {product_description}".strip()
        
        logger.info(f"Generating text embedding for: {text_query[:150]}...")  # Truncate for logging
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
            primary_category = categories_text.split(',')[0].strip() if categories_text else ""
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=primary_category,
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
            primary_category = categories_text.split(',')[0].strip() if categories_text else ""
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=primary_category,
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
        primary_category = categories_text.split(',')[0].strip() if categories_text else ""
        relevant_products = await filter_relevant_products(
            scanned_product_name=product_name,
            scanned_category=primary_category,
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
            primary_category = categories_text.split(',')[0].strip() if categories_text else ""
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=primary_category,
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
            primary_category = categories_text.split(',')[0].strip() if categories_text else ""
            ai_alternatives = await generate_ai_product_alternatives(
                product_name=product_name,
                category=primary_category,
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
        
        # Prepare candidate list for AI with enhanced metadata (categories + description)
        candidates_text = "\n".join([
            f"{i+1}. {p.get('name', 'Unknown')} | Categories: {p.get('categories', 'N/A')} | {p.get('description', 'N/A')[:150]}..."
            for i, p in enumerate(candidate_products)
        ])
        
        prompt = f"""You are evaluating product recommendations for a health-conscious consumer.

Scanned Product: "{scanned_product_name}"
Category: "{scanned_category}"

Candidate Recommendations (with categories & descriptions):
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

IMPORTANT: Use the categories and descriptions provided to make accurate relevance judgments.

Return ONLY a JSON array of relevance scores (1-10) for each candidate in order:
[10, 8, 1, 1, ...]

Be strict: Match product categories exactly. Food replaces food. Beauty replaces beauty. Cleaning replaces cleaning. Don't cross categories."""

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
                "type": "ai_generated",
                "logo_url": "URL to brand logo image (optional)"
            }
        ]
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""You are a health and eco-conscious product expert. A customer scanned "{product_name}" (category: {category}) and it contains harmful additives.

Recommend {count} REAL, SPECIFIC healthier alternative products with ACTUAL BRAND NAMES that they can buy instead.

CRITICAL RULES:
1. Use REAL brand names (e.g., "Dr. Bronner's", "Seventh Generation", "Amy's Organic", "Honest Company")
2. Focus on organic, plant-based, clean-label, eco-friendly brands
3. Match the product category and use case (food for food, beauty for beauty, cleaning for cleaning)
4. Explain WHY each alternative is healthier/cleaner (no harmful additives, organic ingredients, eco-friendly, etc.)
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
                {"role": "system", "content": "You are a health and eco-conscious product expert specializing in clean/natural alternatives across all consumer product categories (food, beauty, cleaning, personal care). You recommend specific real brands and products."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Try to parse JSON with error recovery
        try:
            alternatives = json.loads(content)
        except json.JSONDecodeError as json_err:
            logger.warning(f"[AI ALTERNATIVES] JSON parse failed, attempting repair: {json_err}")
            
            # Try common JSON repairs
            repaired_content = content
            
            # Fix trailing commas before ] or }
            import re
            repaired_content = re.sub(r',\s*([}\]])', r'\1', repaired_content)
            
            # Fix missing commas between objects
            repaired_content = re.sub(r'}\s*{', r'},{', repaired_content)
            
            # Try parsing repaired content
            try:
                alternatives = json.loads(repaired_content)
                logger.info("[AI ALTERNATIVES] JSON repair successful")
            except json.JSONDecodeError:
                logger.error(f"[AI ALTERNATIVES] JSON repair failed, returning empty list. Raw content: {content[:500]}...")
                return []
        
        # Validate alternatives is a list
        if not isinstance(alternatives, list):
            logger.error(f"[AI ALTERNATIVES] Expected list, got {type(alternatives)}, returning empty")
            return []
        
        # Add type marker and fetch brand logos IN PARALLEL
        from services.web_search_service import web_search_service
        
        # Mark all as AI generated first
        for alt in alternatives:
            alt["type"] = "ai_generated"
        
        # Fetch all logos in parallel using asyncio.gather
        async def fetch_logo_for_alt(alt: dict) -> None:
            """Fetch logo for a single alternative (mutates alt dict)"""
            brand_name = alt.get("brand", "")
            if brand_name:
                try:
                    logo_url = await web_search_service.fetch_brand_logo(brand_name, category)
                    if logo_url:
                        alt["logo_url"] = logo_url
                        logger.info(f"[LOGO] Found logo for {brand_name}: {logo_url}")
                    else:
                        logger.warning(f"[LOGO] No logo found for {brand_name}")
                except Exception as e:
                    logger.error(f"[LOGO] Error fetching logo for {brand_name}: {e}")
        
        # Run all logo fetches in parallel
        await asyncio.gather(*[fetch_logo_for_alt(alt) for alt in alternatives], return_exceptions=True)
        
        logger.info(f"Generated {len(alternatives)} AI product alternatives with logos (parallel fetch)")
        return alternatives
        
    except Exception as e:
        logger.error(f"Failed to generate AI alternatives: {e}", exc_info=True)
        return []


async def separate_ingredients_with_ai(
    ingredients_text: str,
    harmful_chemicals_db: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Use OpenAI to intelligently PARSE and categorize ingredients as harmful or safe.
    
    The AI will:
    1. Parse the raw ingredients text (handling parentheses, nested ingredients, etc.)
    2. Extract individual ingredients correctly
    3. Categorize each as harmful or safe based on curated harmful chemicals database
    
    Args:
        ingredients_text: Raw ingredients text from product label (unparsed)
        harmful_chemicals_db: Database of harmful chemicals (REFERENCE ONLY - not for alias matching)
        
    Returns:
        {
            "harmful": ["Red 40", "Yellow 5", ...],
            "safe": ["Cocoa Butter", "Water", ...]
        }
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if not ingredients_text or not ingredients_text.strip():
            return {"harmful": [], "safe": []}
        
        logger.info(f"Raw ingredients text: {ingredients_text[:200]}...")
        
        # Load the comprehensive harmful chemicals database from file (cached)
        harmful_chemicals_text = load_harmful_chemicals_db()
        
        # STEP 1: Create prompt for AI to PARSE AND CATEGORIZE in one step
        prompt = f"""You are a health-conscious ingredient reviewer analyzing product ingredients.

RAW INGREDIENT TEXT FROM PRODUCT LABEL:
{ingredients_text}

TASKS:
1. PARSE the ingredients correctly - extract ALL individual ingredients including nested ones in parentheses
   Example: "Milk Chocolate (sugar, cocoa butter, soy lecithin)" extracts 4 ingredients:
   - Milk Chocolate
   - sugar
   - cocoa butter
   - soy lecithin

2. CATEGORIZE each ingredient as HARMFUL or SAFE based on health and environmental impact.

=== HARMFUL CHEMICALS DATABASE (MUST FLAG) ===
The following comprehensive database contains chemicals that MUST be flagged as harmful.
Use your intelligence to match ingredients even if they have different names, spellings, or aliases:

{harmful_chemicals_text}

=== CATEGORIZATION RULES ===

ALWAYS FLAG AS HARMFUL:
✗ Any ingredient matching the harmful chemicals database above (use semantic matching for aliases)
✗ Artificial/synthetic dyes and colors (Red 40, Yellow 5, Blue 1, E-numbers, etc.)
✗ Artificial preservatives (sodium benzoate, potassium sorbate, BHA, BHT, TBHQ, etc.)
✗ Artificial sweeteners (aspartame, sucralose, saccharin, acesulfame K, etc.)
✗ Natural Flavor / Natural Flavoring / Flavor / Flavoring (hides synthetic chemicals)
✗ Fragrance / Parfum (undisclosed chemical cocktails)
✗ Synthetic chemicals harmful to health (phthalates, parabens, formaldehyde, etc.)
✗ PFAS compounds (Teflon, Scotchgard, etc.)
✗ Processed seed oils (canola, soybean, corn oil, cottonseed, etc.)
✗ High fructose corn syrup, corn syrup, glucose syrup
✗ Refined/processed sugars (cane sugar, brown sugar, etc.) - FLAG THESE
✗ Hydrogenated/partially hydrogenated oils
✗ MSG and hidden MSG (yeast extract, hydrolyzed proteins, etc.)
✗ Modified starches, maltodextrin
✗ Heavily processed ingredients (protein isolates, concentrates, etc.)
✗ Ingredients with known health risks (endocrine disruptors, carcinogens, allergens)

MARK AS SAFE:
✓ Water, sea salt, Himalayan salt
✓ Whole food ingredients (even if not organic): cocoa, milk, butter, eggs, flour, oats, rice, fruits, vegetables
✓ Natural oils: olive oil, coconut oil, avocado oil, butter, ghee
✓ Natural sweeteners: honey, maple syrup, stevia, monk fruit, coconut sugar (even if not organic)
✓ Basic ingredients: baking soda, baking powder, vanilla extract, cocoa powder, chocolate
✓ Vitamins and minerals (Vitamin C, Vitamin E, Calcium, Iron, etc.)
✓ Natural extracts and spices (unless flagged in database)
✓ Eggs, milk, dairy (even if not organic)
✓ Grains and legumes (wheat, soy, corn as ingredients - not as oils)
✓ Natural thickeners: pectin, agar, gelatin

CRITICAL RULES:
• Ingredient names may vary: "Red 40" = "Allura Red" = "E129" = "Red dye 40"
• Use semantic understanding to match database entries
• "Natural flavor" ALWAYS harmful (hides chemicals)
• "Fragrance" ALWAYS harmful (undisclosed mix)
• Sugar IS harmful (flag it)
• Non-organic ingredients are OK unless they're processed/synthetic
• When in doubt about matching database: compare category and function

OUTPUT FORMAT:
Return ONLY valid JSON with NO explanations:
{{
    "harmful": ["ingredient1", "ingredient2"],
    "safe": ["ingredient3", "ingredient4"]
}}

Each ingredient name should NOT include parentheses or descriptions.
Every ingredient must appear in exactly ONE list (harmful OR safe, never both).
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Faster and cheaper for this task
            messages=[
                {"role": "system", "content": "You are an expert at parsing ingredient labels and identifying harmful chemicals. You understand ingredient aliases, chemical names, E-numbers, and can semantically match ingredients to a harmful chemicals database. You always return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Slightly higher for better semantic matching
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        logger.info(f"AI parsed and separated ingredients: {len(result.get('harmful', []))} harmful, {len(result.get('safe', []))} safe")
        
        return result
        
    except Exception as e:
        logger.error(f"AI ingredient parsing/separation failed: {e}")
        return {"harmful": [], "safe": []}
        # Fallback to empty lists
        return {"harmful": [], "safe": []}


def get_detailed_ingredient_descriptions(
    safe_ingredients: list, 
    harmful_ingredients: list,
    harmful_chemicals_db: list
) -> dict:
    """
    Generate detailed, user-friendly AI descriptions for ALREADY SEPARATED ingredients.
    
    OPTIMIZED: Uses a SINGLE batched OpenAI API call for both harmful and safe ingredients.
    
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
        # If no ingredients to describe, return early
        if not harmful_ingredients and not safe_ingredients:
            return {"safe": {}, "harmful": {}}
        
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Build harmful items with context
        harmful_items = []
        for ing_name in harmful_ingredients[:15]:  # Limit to avoid token limits
            db_info = None
            for chem in harmful_chemicals_db:
                if chem.get('name', '').lower() == ing_name.lower():
                    db_info = chem
                    break
            
            if db_info:
                harmful_items.append(f"{ing_name} (Category: {db_info.get('category', 'unknown')})")
            else:
                harmful_items.append(f"{ing_name} (AI flagged as harmful)")
        
        harmful_str = ", ".join(harmful_items) if harmful_items else "None"
        safe_str = ", ".join(safe_ingredients[:20]) if safe_ingredients else "None"
        
        logger.info(f"[BATCHED] Generating descriptions for {len(harmful_ingredients)} harmful + {len(safe_ingredients)} safe ingredients in ONE API call")
        
        # SINGLE BATCHED PROMPT for both harmful and safe
        batched_prompt = f"""Analyze these ingredients from a consumer product. Provide descriptions for BOTH harmful AND safe ingredients in a single response.

=== HARMFUL INGREDIENTS (be critical, explain dangers) ===
{harmful_str}

=== SAFE INGREDIENTS (be educational, honest about processing) ===
{safe_str}

INSTRUCTIONS:
1. For HARMFUL ingredients: Explain what it is, why it's harmful, health/environmental risks. Be uncompromising.
2. For SAFE ingredients: Brief 1-2 sentence educational description. Natural = positive, processed = honest critique.
3. Use exact ingredient names as JSON keys.

Return ONLY this JSON structure:
{{
    "harmful": {{
        "Ingredient Name": "Critical description of why this is harmful..."
    }},
    "safe": {{
        "Ingredient Name": "Brief educational description..."
    }}
}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a strict ingredient auditor for consumer products with a clean/natural product philosophy.

For HARMFUL ingredients:
• Be uncompromising - these are flagged for a reason
• Call out synthetic, industrial, and processed origins
• Never say "safe in small amounts" or cite regulatory approval
• E-numbers, dyes, artificial flavors = always problematic

For SAFE ingredients:
• Natural = positive descriptions
• Ultra-processed = honest critique
• Keep it educational, not alarmist

Output ONLY valid JSON with the exact structure requested."""
                },
                {"role": "user", "content": batched_prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        content = response.choices[0].message.content.strip()
        logger.info(f"[BATCHED] Raw AI response: {content[:300]}...")
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        harmful_descriptions = result.get("harmful", {})
        safe_descriptions = result.get("safe", {})
        
        logger.info(f"[BATCHED] Generated {len(harmful_descriptions)} harmful + {len(safe_descriptions)} safe descriptions in ONE call")
        
        return {
            "safe": safe_descriptions,
            "harmful": harmful_descriptions
        }
        
    except Exception as e:
        logger.error(f"Error in get_detailed_ingredient_descriptions (batched): {e}", exc_info=True)
        # Fallback to basic descriptions
        harmful_descriptions = {ing: "This ingredient has been flagged as potentially harmful." for ing in harmful_ingredients}
        safe_descriptions = {ing: "Common ingredient used in consumer products." for ing in safe_ingredients}
        return {"safe": safe_descriptions, "harmful": harmful_descriptions}


# === ASYNC PARALLEL VERSIONS FOR OPTIMIZED PERFORMANCE ===

async def _generate_harmful_descriptions_async(harmful_ingredients: list, harmful_chemicals_db: list) -> dict:
    """
    Async version of harmful ingredient description generation.
    Runs in thread pool to avoid blocking.
    """
    if not harmful_ingredients:
        return {}
    
    def _sync_generate():
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            harmful_items = []
            for ing_name in harmful_ingredients[:15]:
                db_info = None
                for chem in harmful_chemicals_db:
                    if chem.get('name', '').lower() == ing_name.lower():
                        db_info = chem
                        break
                
                if db_info:
                    harmful_items.append(f"{ing_name} (Category: {db_info.get('category', 'unknown')})")
                else:
                    harmful_items.append(f"{ing_name} (AI flagged as harmful)")
            
            harmful_str = ", ".join(harmful_items)
            
            harmful_prompt = f"""Analyze these harmful chemicals/ingredients found in a consumer product.

FLAGGED CHEMICALS (authoritative — do not dispute or soften):
{harmful_str}

For each chemical, provide a description that:
1. States what the chemical is and its industrial/synthetic use in consumer products
2. Explains why it's harmful to human health and/or the environment
3. Calls out specific health risks, toxicity concerns, or environmental damage
4. Uses clear, direct language - no corporate apologetics

CRITICAL RULES:
• These ingredients are FLAGGED for a reason - do not defend them
• If it's synthetic, lab-derived, or industrially processed, emphasize that
• E-numbers should be called out as synthetic petroleum-based additives
• Never say "safe in small amounts" or cite regulatory approval as reassurance

Format as JSON: {{"Chemical Name Exactly As Listed": "critical description"}}

Harmful chemicals: {harmful_str}"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a strict ingredient auditor. Output valid JSON only with ingredient names as keys."
                    },
                    {"role": "user", "content": harmful_prompt}
                ],
                temperature=0.3,
                max_tokens=1200,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Async harmful descriptions failed: {e}")
            return {ing: "Flagged as potentially harmful ingredient." for ing in harmful_ingredients}
    
    return await asyncio.get_event_loop().run_in_executor(None, _sync_generate)


async def _generate_safe_descriptions_async(safe_ingredients: list) -> dict:
    """
    Async version of safe ingredient description generation.
    Runs in thread pool to avoid blocking.
    """
    if not safe_ingredients:
        return {}
    
    def _sync_generate():
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            ingredients_str = ", ".join(safe_ingredients)
            
            safe_prompt = f"""Analyze these safe/common ingredients from a clean product perspective.

INGREDIENTS: {ingredients_str}

For each, provide a brief 1-2 sentence description:
1. What it is (natural vs processed)
2. Its purpose in products
3. Any processing concerns

Return ONLY valid JSON: {{"Ingredient Name": "Brief description"}}"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an ingredient analyst. Output valid JSON only."
                    },
                    {"role": "user", "content": safe_prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Async safe descriptions failed: {e}")
            return {ing: "Common ingredient in consumer products." for ing in safe_ingredients}
    
    return await asyncio.get_event_loop().run_in_executor(None, _sync_generate)


async def _analyze_packaging_async(
    packaging_text: str, 
    packaging_tags: list,
    normalized_materials: list = None,
    brand_name: str = None,
    product_name: str = None,
    categories: str = None
) -> dict:
    """
    Async version of packaging analysis.
    Runs in thread pool to avoid blocking.
    """
    if not packaging_text and not packaging_tags and not normalized_materials:
        return {
            "materials": [],
            "analysis": {},
            "overall_safety": "unknown",
            "summary": "No packaging information available"
        }
    
    def _sync_analyze():
        try:
            if normalized_materials:
                materials = normalized_materials
            else:
                materials = packaging_tags if packaging_tags else [packaging_text]
            
            packaging_info = f"Packaging: {packaging_text}. Tags: {', '.join(packaging_tags)}" if packaging_tags else packaging_text
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            context_parts = []
            if brand_name:
                context_parts.append(f"Brand: {brand_name}")
            if product_name:
                context_parts.append(f"Product: {product_name}")
            if categories:
                main_category = categories.split(",")[0].strip() if "," in categories else categories
                context_parts.append(f"Category: {main_category}")
            
            product_context = "\n".join(context_parts) if context_parts else "Product context not available"
            
            prompt = f"""Analyze these packaging materials for health and environmental impact.

PRODUCT CONTEXT:
{product_context}

PACKAGING INFO:
{packaging_info}
Materials: {', '.join(materials)}

For EACH material provide:
- description: what it is and why it's used
- harmful: true/false
- health_concerns: specific risks
- environmental_impact: recyclability, pollution
- severity: low/moderate/high/critical

Format as JSON:
{{
    "materials": {json.dumps(materials)},
    "analysis": {{"Material": {{"description": "...", "harmful": false, "health_concerns": "...", "environmental_impact": "...", "severity": "low"}}}},
    "overall_safety": "safe/caution/harmful",
    "summary": "2-3 sentence assessment"
}}"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a packaging safety expert. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Async packaging analysis failed: {e}")
            return {
                "materials": ["unknown"],
                "analysis": {},
                "overall_safety": "unknown",
                "summary": "Unable to analyze packaging"
            }
    
    return await asyncio.get_event_loop().run_in_executor(None, _sync_analyze)


async def get_ingredient_descriptions_parallel(
    safe_ingredients: list,
    harmful_ingredients: list,
    harmful_chemicals_db: list
) -> dict:
    """
    Generate ingredient descriptions in PARALLEL (harmful + safe simultaneously).
    This is ~2x faster than sequential generation.
    """
    logger.info(f"[PARALLEL] Starting parallel description generation: {len(harmful_ingredients)} harmful, {len(safe_ingredients)} safe")
    
    # Run both description generations in parallel
    harmful_task = _generate_harmful_descriptions_async(harmful_ingredients, harmful_chemicals_db)
    safe_task = _generate_safe_descriptions_async(safe_ingredients)
    
    harmful_desc, safe_desc = await asyncio.gather(harmful_task, safe_task)
    
    logger.info(f"[PARALLEL] Completed: {len(harmful_desc)} harmful, {len(safe_desc)} safe descriptions")
    
    return {
        "safe": safe_desc,
        "harmful": harmful_desc
    }


def analyze_packaging_material(
    packaging_text: str, 
    packaging_tags: list, 
    normalized_materials: list = None,
    brand_name: str = None,
    product_name: str = None,
    categories: str = None,
    food_contact: bool = None
) -> dict:
    """
    Analyze packaging materials for environmental and health impact.
    Considers: plastic types (BPA, microplastics), recyclability, environmental impact
    
    Args:
        packaging_text: Raw packaging text from product data
        packaging_tags: Raw packaging tags from product data
        normalized_materials: Pre-normalized material names from /separate route (RECOMMENDED)
        brand_name: Product brand name for context
        product_name: Product name for context
        categories: Product categories for context
        food_contact: Whether packaging has direct food contact
    
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
        if not packaging_text and not packaging_tags and not normalized_materials:
            return {
                "materials": [],
                "analysis": {},
                "overall_safety": "unknown",
                "summary": "No packaging information available"
            }
        
        # Use pre-normalized materials from /separate route if provided
        if normalized_materials:
            logger.info(f"Using pre-normalized material names: {normalized_materials}")
        else:
            # Fallback to packaging tags if no normalized materials provided
            normalized_materials = packaging_tags if packaging_tags else [packaging_text]
            logger.warning(f"No normalized materials provided, using fallback: {normalized_materials}")
        
        # Combine packaging info for context
        packaging_info = f"Packaging: {packaging_text}. Tags: {', '.join(packaging_tags)}" if packaging_tags else packaging_text
        
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Build product context string
        context_parts = []
        if brand_name:
            context_parts.append(f"Brand: {brand_name}")
        if product_name:
            context_parts.append(f"Product: {product_name}")
        if categories:
            # Use first category for simplicity
            main_category = categories.split(",")[0].strip() if "," in categories else categories
            context_parts.append(f"Category: {main_category}")
        if food_contact is not None:
            contact_status = "YES - direct food contact" if food_contact else "NO - no direct food contact"
            context_parts.append(f"Food Contact: {contact_status}")
        
        product_context = "\n".join(context_parts) if context_parts else "Product context not available"
        
        logger.info(f"[PACKAGING-ANALYZE] Building context from parameters:")
        logger.info(f"  - brand_name param: '{brand_name}'")
        logger.info(f"  - product_name param: '{product_name}'")
        logger.info(f"  - categories param: '{categories}'")
        logger.info(f"  - food_contact param: {food_contact}")
        logger.info(f"[PACKAGING-ANALYZE] Final product context:\n{product_context}")
        
        # Generate full descriptions for normalized materials
        prompt = f"""Analyze the following product packaging materials for health and environmental impact.

PRODUCT CONTEXT:
{product_context}

PACKAGING INFO:
{packaging_info}
Normalized materials: {', '.join(normalized_materials)}

CRITICAL RULES:
1. DO NOT use uncertainty phrases like "specific type unknown", "unknown", "unclear", "may contain", "could be"
2. Each packaging material gets its own complete, detailed analysis
3. USE THE PRODUCT CONTEXT (brand, product name, category, food contact status) to provide specific analysis
4. If food contact is YES, emphasize health risks from leaching and chemical migration
5. If product context is missing, intelligently infer based on packaging materials and common usage patterns
6. Be authoritative and fact-based

For EACH packaging material, provide:

**Description**: 
- How this specific brand uses this material for this product (reference brand name and product if available)
- For food contact packaging: mention direct contact with food and migration risks
- What this material is made of and its properties
- Why manufacturers in this category choose this packaging type

**Health Concerns**:
- If food contact: emphasize chemical leaching into food, migration of additives
- Specific chemicals and additives present in this material type
- Microplastic generation and ingestion risks
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

TONE: Authoritative, specific to this brand and product. Use product context to provide detailed analysis.

Example with context (DO):
"Cadbury packages this Dairy Milk chocolate bar in a flexible plastic wrapper with direct food contact. This multi-layer plastic film typically contains polyethylene and polypropylene layers bonded together for moisture protection. For chocolate products, manufacturers choose this format to prevent oxidation and maintain freshness while allowing easy unwrapping."

Example without context (DON'T):
"Stand-up pouches are commonly used for snacks."

Format as JSON:

IMPORTANT: Use EXACTLY these normalized material names in both the "materials" array and as keys in "analysis": {json.dumps(normalized_materials)}

{{
    "materials": {json.dumps(normalized_materials)},
    "analysis": {{
        "Material Name": {{
            "description": "detailed brand-specific description of how this material is used for this product, what it's made of, and why it's chosen",
            "harmful": true or false,
            "health_concerns": "specific health risks with definitive statements (no uncertainty)",
            "environmental_impact": "detailed environmental assessment with specific facts",
            "severity": "low/moderate/high/critical"
        }}
    }},
    "overall_safety": "safe/caution/harmful",
    "summary": "authoritative 2-3 sentence assessment of the overall packaging"
}}
"""
        
        # Log the full prompt for debugging
        logger.info(f"[PACKAGING-ANALYZE] Full prompt being sent to OpenAI:\n{prompt}")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an environmental scientist and toxicologist expert in packaging materials, plastic pollution, and consumer product safety. Always return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,  # Increased token limit to avoid truncation
            response_format={"type": "json_object"}  # Force JSON mode
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON directly first
        try:
            packaging_analysis = json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try extracting from code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Try parsing again
            try:
                packaging_analysis = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse packaging analysis JSON: {e}")
                logger.error(f"Raw response: {content[:500]}...")
                # Return fallback
                return {
                    "materials": ["unknown"],
                    "analysis": {
                        "unknown": {
                            "description": "Unable to analyze packaging materials",
                            "harmful": False,
                            "health_concerns": "Unknown",
                            "environmental_impact": "Unknown",
                            "severity": "low"
                        }
                    },
                    "overall_safety": "unknown",
                    "summary": "Unable to analyze packaging materials at this time."
                }
        
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
                ingredients_text = ingredients_text + ", " + inferred_ingredients if ingredients_text else inferred_ingredients
                ingredients_source = "ai_inferred"
                inference_used = True
                logger.info(f"AI inference added {len(inference_data['likely_ingredients'])} likely ingredients")
        
        # === AI PARSING + SEPARATION ===
        # Let AI parse and separate ingredients in one step
        harmful_ingredient_names = []
        safe_ingredient_names = []
        
        if ingredients_text and ingredients_text.strip():
            logger.info(f"Parsing and separating ingredients with AI")
            
            # Use AI to parse AND intelligently separate harmful vs safe
            separated = await separate_ingredients_with_ai(ingredients_text, [])
            
            harmful_ingredient_names = separated.get("harmful", [])
            safe_ingredient_names = separated.get("safe", [])
            
            logger.info(f"AI parsing and separation complete: {len(harmful_ingredient_names)} harmful, {len(safe_ingredient_names)} safe")
        
        # Build harmful chemicals list from AI separation
        # No need for safety score calculation - not used in frontend
        harmful_chemicals = [
            {
                "name": name, 
                "type": "additive",
                "category": "additive",  # Generic category for AI-identified ingredients
            } 
            for name in harmful_ingredient_names
        ]
        
        # No recommendations needed here - they're fetched separately via /barcode/recommendations endpoint
        # This saves processing time on the main barcode lookup
        
        # Build chemical analysis (simplified - no safety score)
        product_data["chemical_analysis"] = {
            "harmful_chemicals": harmful_chemicals,
            "harmful_count": len(harmful_chemicals),
        }
        
        # Store separated lists (no duplicates - AI handles the separation)
        product_data["ingredients"] = safe_ingredient_names
        product_data["harmful_ingredients"] = harmful_ingredient_names
        
        # === EXTRACT PACKAGING INFO BEFORE PARALLEL EXECUTION ===
        packaging_text = product_data.get("packaging", "")
        packaging_tags = product_data.get("packaging_tags", [])
        
        # Check if packaging info is nested in materials object
        if not packaging_text and "materials" in product_data:
            materials = product_data["materials"]
            packaging_text = materials.get("packaging_text", "") or materials.get("packaging", "")
            packaging_tags = materials.get("packaging_tags", [])
        
        logger.info(f"Packaging text: '{packaging_text}', Tags: {packaging_tags}")
        
        # === INFER PACKAGING IF MISSING (before parallel execution) ===
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
            
            # Strategy 2: AI inference based on category
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
        
        # === PARALLEL EXECUTION: Run ingredient descriptions + packaging analysis simultaneously ===
        logger.info(f"[PARALLEL] Starting parallel AI analysis...")
        
        # Prepare tasks
        tasks = []
        
        # Task 1: Ingredient descriptions (if we have ingredients)
        if harmful_ingredient_names or safe_ingredient_names:
            ingredient_task = get_ingredient_descriptions_parallel(
                safe_ingredients=safe_ingredient_names,
                harmful_ingredients=harmful_ingredient_names,
                harmful_chemicals_db=harmful_chemicals
            )
            tasks.append(("ingredients", ingredient_task))
        
        # Task 2: Packaging analysis (if we have packaging info)
        if packaging_text or packaging_tags:
            packaging_task = _analyze_packaging_async(
                packaging_text=packaging_text,
                packaging_tags=packaging_tags,
                brand_name=product_data.get("brands"),
                product_name=product_data.get("name"),
                categories=product_data.get("categories")
            )
            tasks.append(("packaging", packaging_task))
        
        # Execute all tasks in parallel
        if tasks:
            task_names = [t[0] for t in tasks]
            task_coroutines = [t[1] for t in tasks]
            
            logger.info(f"[PARALLEL] Executing {len(tasks)} tasks in parallel: {task_names}")
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Process results
            for i, (name, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"[PARALLEL] Task '{name}' failed: {result}")
                    if name == "ingredients":
                        product_data["ingredient_descriptions"] = {"safe": {}, "harmful": {}}
                    elif name == "packaging":
                        product_data["packaging_analysis"] = {
                            "materials": [],
                            "analysis": {},
                            "overall_safety": "unknown",
                            "summary": "Packaging analysis failed"
                        }
                else:
                    if name == "ingredients":
                        product_data["ingredient_descriptions"] = result
                        logger.info(f"[PARALLEL] Ingredient descriptions complete")
                    elif name == "packaging":
                        product_data["packaging_analysis"] = result
                        logger.info(f"[PARALLEL] Packaging analysis complete: {len(result.get('materials', []))} materials")
            
            logger.info(f"[PARALLEL] All tasks completed!")
        else:
            # No tasks to run - set defaults
            product_data["ingredient_descriptions"] = {"safe": {}, "harmful": {}}
            product_data["packaging_analysis"] = {
                "materials": [],
                "analysis": {},
                "overall_safety": "unknown",
                "summary": "No packaging information available"
            }
            packaging_source = "none"
        
        # Set packaging_source to none if no packaging was analyzed
        if not packaging_text and not packaging_tags:
            packaging_source = "none"
        
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
        
        product_data["analysis_metadata"] = {
            "ingredients_source": ingredients_source,
            "packaging_source": packaging_source,
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


@router.get("/barcode/lookup")
async def barcode_lookup(barcode: str) -> Dict[str, Any]:
    """
    FAST: Get basic product info from OpenFacts with parsed ingredients.
    This is the first endpoint to call - returns in 1-2 seconds.
    
    Args:
        barcode: Product barcode (UPC/EAN)
        
    Returns:
        Basic product data with parsed ingredients list
    """
    try:
        barcode = barcode.strip()
        
        # Validate barcode
        if not barcode:
            raise HTTPException(status_code=400, detail="Barcode cannot be empty")
        
        if not barcode.isdigit():
            raise HTTPException(status_code=400, detail="Barcode must contain only digits")
        
        logger.info(f"[FAST] Looking up barcode: {barcode}")
        
        # Query barcode service (with caching)
        barcode_service = get_barcode_service()
        product_data = await barcode_service.lookup_barcode(barcode)
        
        if not product_data:
            return {
                'success': True,
                'found': False,
                'product': None,
                'message': 'Product not found in database'
            }
        
        # Extract and parse ingredients
        ingredients_text = product_data.get("ingredients_text", "")
        
        # If ingredients_text is empty, try to extract from ingredients array
        if not ingredients_text and "ingredients" in product_data:
            ingredients_array = product_data["ingredients"]
            for ing in ingredients_array:
                if isinstance(ing, dict) and ing.get("type") == "full_text":
                    ingredients_text = ing.get("text", "")
                    break
            
            if not ingredients_text:
                ingredient_texts = []
                for ing in ingredients_array:
                    if isinstance(ing, dict) and ing.get("type") == "structured":
                        text = ing.get("text") or ing.get("id", "").replace("en:", "").replace("-", " ")
                        if text:
                            ingredient_texts.append(text)
                ingredients_text = ", ".join(ingredient_texts)
        
        # Parse ingredients into list
        ingredients_list = []
        if ingredients_text:
            for ing in ingredients_text.split(","):
                ing = ing.strip()
                if ing:
                    ingredients_list.append(str(ing))
        
        # Add parsed ingredients to response
        product_data["ingredients_text"] = ingredients_text
        product_data["ingredients_list"] = ingredients_list
        product_data["has_ingredients"] = len(ingredients_list) > 0
        
        logger.info(f"[FAST] Found product: {product_data.get('name')} with {len(ingredients_list)} ingredients")
        
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
        raise HTTPException(status_code=500, detail=f"Error looking up barcode: {str(e)}")


class IngredientsAnalyzeRequest(BaseModel):
    """Request model for ingredients analysis"""
    barcode: str
    product_data: Optional[Dict[str, Any]] = None  # Optional: reuse data from /barcode/lookup


class IngredientsSeparateRequest(BaseModel):
    """Request model for fast ingredient separation (names only)"""
    barcode: str
    product_data: Optional[Dict[str, Any]] = None


class IngredientsDescribeRequest(BaseModel):
    """Request model for ingredient descriptions"""
    barcode: str
    harmful_ingredients: list[str]
    safe_ingredients: list[str]


@router.post("/barcode/ingredients/separate")
async def separate_ingredients(request: IngredientsSeparateRequest) -> Dict[str, Any]:
    """
    FAST: Separate ingredients into harmful/safe (names only).
    Takes 2-3 seconds. Call this first to show ingredient names immediately.
    
    Args:
        request: Contains barcode and optional product_data
        
    Returns:
        Harmful/safe ingredient names only (no descriptions)
    """
    try:
        barcode = request.barcode.strip()
        
        if not barcode or not barcode.isdigit():
            raise HTTPException(status_code=400, detail="Invalid barcode")
        
        logger.info(f"[INGREDIENTS-SEPARATE] Separating ingredients for barcode: {barcode}")
        
        # Get product data (use provided or fetch fresh)
        if request.product_data:
            product_data = request.product_data
            logger.info("[INGREDIENTS-SEPARATE] Using provided product data (no API call)")
        else:
            barcode_service = get_barcode_service()
            product_data = await barcode_service.lookup_barcode(barcode)
            if not product_data:
                raise HTTPException(status_code=404, detail="Product not found")
        
        # Get ingredients list
        ingredients_list = product_data.get("ingredients_list", [])
        ingredients_text = product_data.get("ingredients_text", "")
        
        # If no ingredients in provided data, try to extract
        if not ingredients_list:
            if not ingredients_text and "ingredients" in product_data:
                ingredients_array = product_data["ingredients"]
                for ing in ingredients_array:
                    if isinstance(ing, dict) and ing.get("type") == "full_text":
                        ingredients_text = ing.get("text", "")
                        break
                
                if not ingredients_text:
                    ingredient_texts = []
                    for ing in ingredients_array:
                        if isinstance(ing, dict) and ing.get("type") == "structured":
                            text = ing.get("text") or ing.get("id", "").replace("en:", "").replace("-", " ")
                            if text:
                                ingredient_texts.append(text)
                    ingredients_text = ", ".join(ingredient_texts)
        
        # AI INFERENCE if ingredients missing or minimal
        if not ingredients_text or len(ingredients_text.strip()) < 20:
            logger.info(f"[INGREDIENTS-SEPARATE] Minimal data - attempting AI inference")
            inference_data = await infer_ingredients_from_category(
                product_name=product_data.get("name", "Unknown"),
                brand=product_data.get("brands", ""),
                category=product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General",
                # image_url=product_data.get("image_url")
            )
            
            if inference_data and inference_data.get("likely_ingredients"):
                inferred_ingredients = ", ".join(inference_data["likely_ingredients"])
                ingredients_text = ingredients_text + ", " + inferred_ingredients if ingredients_text else inferred_ingredients
                
                logger.info(f"[INGREDIENTS-SEPARATE] AI inference added {len(inference_data['likely_ingredients'])} ingredients")
            
            # WEB SEARCH FALLBACK if AI inference failed or returned minimal data
            if not ingredients_text or len(ingredients_text.strip()) < 20:
                logger.info(f"[INGREDIENTS-SEPARATE] AI inference insufficient - attempting web search")
                web_search_data = await search_ingredients_from_context(
                    product_name=product_data.get("name", "Unknown"),
                    brand=product_data.get("brands", ""),
                    category=product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General"
                )
                
                if web_search_data and web_search_data.get("ingredients_text"):
                    web_search_ingredients = web_search_data["ingredients_text"]
                    ingredients_text = ingredients_text + ", " + web_search_ingredients if ingredients_text else web_search_ingredients
                    
                    logger.info(f"[INGREDIENTS-SEPARATE] Web search added ingredients: {web_search_ingredients[:100]}...")
        
        if not ingredients_text or not ingredients_text.strip():
            return {
                'success': True,
                'has_ingredients': False,
                'harmful': [],
                'safe': [],
                'message': 'No ingredients found'
            }
        
        # AI PARSING + SEPARATION (FAST - only 1 AI call)
        logger.info(f"[INGREDIENTS-SEPARATE] Parsing and separating ingredients with AI")
        separated = await separate_ingredients_with_ai(ingredients_text, [])
        
        harmful_ingredient_names = separated.get("harmful", [])
        safe_ingredient_names = separated.get("safe", [])
        
        logger.info(f"[INGREDIENTS-SEPARATE] Complete: {len(harmful_ingredient_names)} harmful, {len(safe_ingredient_names)} safe")
        
        total_ingredients = len(harmful_ingredient_names) + len(safe_ingredient_names)
        
        return {
            'success': True,
            'has_ingredients': True,
            'harmful': harmful_ingredient_names,
            'safe': safe_ingredient_names,
            'harmful_count': len(harmful_ingredient_names),
            'safe_count': len(safe_ingredient_names),
            'total_count': total_ingredients,
            'message': f'Parsed and separated {total_ingredients} ingredients'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingredient separation: {e}")
        raise HTTPException(status_code=500, detail=f"Error separating ingredients: {str(e)}")


@router.post("/barcode/ingredients/describe")
async def describe_ingredients(request: IngredientsDescribeRequest) -> Dict[str, Any]:
    """
    SLOW: Generate AI descriptions for separated ingredients.
    Takes 3-5 seconds. Call after /separate to get detailed descriptions.
    
    Args:
        request: Contains harmful and safe ingredient lists
        
    Returns:
        AI-generated descriptions for each ingredient
    """
    try:
        barcode = request.barcode.strip()
        
        if not barcode or not barcode.isdigit():
            raise HTTPException(status_code=400, detail="Invalid barcode")
        
        logger.info(f"[INGREDIENTS-DESCRIBE] Generating descriptions for {len(request.harmful_ingredients)} harmful, {len(request.safe_ingredients)} safe ingredients")
        
        # Generate harmful chemicals list
        harmful_chemicals = [
            {"name": name, "type": "additive", "category": "additive"}
            for name in request.harmful_ingredients
        ]
        
        # GENERATE DESCRIPTIONS (SLOW - 2 AI calls)
        ingredient_descriptions = get_detailed_ingredient_descriptions(
            safe_ingredients=request.safe_ingredients,
            harmful_ingredients=request.harmful_ingredients,
            harmful_chemicals_db=harmful_chemicals
        )
        
        logger.info(f"[INGREDIENTS-DESCRIBE] Descriptions generated")
        
        return {
            'success': True,
            'descriptions': ingredient_descriptions,
            'message': 'Descriptions generated successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating ingredient descriptions: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating descriptions: {str(e)}")


@router.post("/barcode/ingredients/analyze")
async def analyze_ingredients(request: IngredientsAnalyzeRequest) -> Dict[str, Any]:
    """
    MEDIUM: AI-powered ingredient analysis (separation + descriptions).
    Takes 5-10 seconds. Call after /barcode/lookup.
    
    Args:
        request: Contains barcode and optional product_data
        
    Returns:
        Harmful/safe ingredient separation with AI descriptions
    """
    try:
        barcode = request.barcode.strip()
        
        if not barcode or not barcode.isdigit():
            raise HTTPException(status_code=400, detail="Invalid barcode")
        
        logger.info(f"[INGREDIENTS] Analyzing ingredients for barcode: {barcode}")
        
        # Get product data (use provided or fetch fresh)
        if request.product_data:
            product_data = request.product_data
            logger.info("[INGREDIENTS] Using provided product data (no API call)")
        else:
            barcode_service = get_barcode_service()
            product_data = await barcode_service.lookup_barcode(barcode)
            if not product_data:
                raise HTTPException(status_code=404, detail="Product not found")
        
        # Get ingredients list
        ingredients_list = product_data.get("ingredients_list", [])
        ingredients_text = product_data.get("ingredients_text", "")
        
        # If no ingredients in provided data, try to extract
        if not ingredients_list:
            if not ingredients_text and "ingredients" in product_data:
                ingredients_array = product_data["ingredients"]
                for ing in ingredients_array:
                    if isinstance(ing, dict) and ing.get("type") == "full_text":
                        ingredients_text = ing.get("text", "")
                        break
                
                if not ingredients_text:
                    ingredient_texts = []
                    for ing in ingredients_array:
                        if isinstance(ing, dict) and ing.get("type") == "structured":
                            text = ing.get("text") or ing.get("id", "").replace("en:", "").replace("-", " ")
                            if text:
                                ingredient_texts.append(text)
                    ingredients_text = ", ".join(ingredient_texts)
        
        # AI INFERENCE if ingredients missing or minimal
        if not ingredients_text or len(ingredients_text.strip()) < 20:
            logger.info(f"[INGREDIENTS] Minimal data - attempting AI inference")
            inference_data = await infer_ingredients_from_context(
                product_name=product_data.get("name", "Unknown"),
                brand=product_data.get("brands", ""),
                category=product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General",
                image_url=product_data.get("image_url")
            )
            
            if inference_data and inference_data.get("likely_ingredients"):
                inferred_ingredients = ", ".join(inference_data["likely_ingredients"])
                ingredients_text = ingredients_text + ", " + inferred_ingredients if ingredients_text else inferred_ingredients
                
                logger.info(f"[INGREDIENTS] AI inference added {len(inference_data['likely_ingredients'])} ingredients")
        
        if not ingredients_text or not ingredients_text.strip():
            return {
                'success': True,
                'has_ingredients': False,
                'harmful': [],
                'safe': [],
                'descriptions': {'harmful': {}, 'safe': {}},
                'message': 'No ingredients found for analysis'
            }
        
        # AI PARSING + SEPARATION
        logger.info(f"[INGREDIENTS] Parsing and separating ingredients with AI")
        separated = await separate_ingredients_with_ai(ingredients_text, [])
        
        harmful_ingredient_names = separated.get("harmful", [])
        safe_ingredient_names = separated.get("safe", [])
        
        logger.info(f"[INGREDIENTS] AI separation: {len(harmful_ingredient_names)} harmful, {len(safe_ingredient_names)} safe")
        
        # Generate harmful chemicals list
        harmful_chemicals = [
            {"name": name, "type": "additive", "category": "additive"}
            for name in harmful_ingredient_names
        ]
        
        # GENERATE DESCRIPTIONS
        logger.info(f"[INGREDIENTS] Generating AI descriptions")
        ingredient_descriptions = get_detailed_ingredient_descriptions(
            safe_ingredients=safe_ingredient_names,
            harmful_ingredients=harmful_ingredient_names,
            harmful_chemicals_db=harmful_chemicals
        )
        
        logger.info(f"[INGREDIENTS] Analysis complete")
        
        total_ingredients = len(harmful_ingredient_names) + len(safe_ingredient_names)
        
        return {
            'success': True,
            'has_ingredients': True,
            'harmful': harmful_ingredient_names,
            'safe': safe_ingredient_names,
            'harmful_count': len(harmful_ingredient_names),
            'safe_count': len(safe_ingredient_names),
            'descriptions': ingredient_descriptions,
            'message': f'Analyzed {total_ingredients} ingredients'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingredient analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing ingredients: {str(e)}")


class PackagingAnalyzeRequest(BaseModel):
    """Request model for packaging analysis"""
    barcode: str
    product_data: Optional[Dict[str, Any]] = None


class PackagingSeparateRequest(BaseModel):
    """Request model for fast packaging material extraction (names only)"""
    barcode: str
    product_data: Optional[Dict[str, Any]] = None


class PackagingDescribeRequest(BaseModel):
    """Request model for packaging material descriptions"""
    barcode: str
    packaging_text: str
    packaging_tags: list[str]
    normalized_materials: Optional[list[str]] = None  # Pre-normalized materials from /separate route
    brand_name: Optional[str] = None
    product_name: Optional[str] = None
    categories: Optional[str] = None
    categories_tags: Optional[list[str]] = None
    food_contact: Optional[bool] = None
    packagings_data: Optional[list] = None


def normalize_none_string(value: any) -> any:
    """Convert string 'None' to actual None, and strip empty strings to None"""
    if value == "None" or value == "null" or value == "undefined":
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


@router.post("/barcode/packaging/separate")
async def separate_packaging(request: PackagingSeparateRequest) -> Dict[str, Any]:
    """
    FAST: Extract packaging material names only.
    Takes 1-2 seconds. Call this first to show material names immediately.
    
    Args:
        request: Contains barcode and optional product_data
        
    Returns:
        Packaging material names only (no detailed analysis)
    """
    try:
        barcode = request.barcode.strip()
        
        if not barcode or not barcode.isdigit():
            raise HTTPException(status_code=400, detail="Invalid barcode")
        
        logger.info(f"[PACKAGING-SEPARATE] Extracting packaging materials for barcode: {barcode}")
        
        # Get product data (use provided or fetch fresh)
        if request.product_data:
            product_data = request.product_data
            logger.info("[PACKAGING-SEPARATE] Using provided product data (no API call)")
        else:
            barcode_service = get_barcode_service()
            product_data = await barcode_service.lookup_barcode(barcode)
            if not product_data:
                raise HTTPException(status_code=404, detail="Product not found")
        
        # Extract packaging info
        packaging_text = product_data.get("packaging", "")
        packaging_tags = product_data.get("packaging_tags", [])
        
        # Check if packaging info is nested in materials object
        if not packaging_text and "materials" in product_data:
            materials = product_data["materials"]
            packaging_text = materials.get("packaging_text", "") or materials.get("packaging", "")
            packaging_tags = materials.get("packaging_tags", [])
        
        logger.info(f"[PACKAGING-SEPARATE] Packaging text: '{packaging_text}', Tags: {packaging_tags}")
        
        # INFER PACKAGING IF MISSING
        packaging_source = "openfacts"
        if not packaging_text and not packaging_tags:
            logger.info("[PACKAGING-SEPARATE] No data from OpenFacts - attempting AI inference")
            
            # Fallback to AI category-based inference
            category = product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General product"
            packaging_materials = await infer_packaging_from_category(
                product_name=product_data.get("name", "Unknown"),
                category=category
            )
            if packaging_materials:
                # AI returns a list of materials directly
                packaging_tags = packaging_materials
                packaging_text = ", ".join(packaging_materials)
                packaging_source = "ai_inferred"
                logger.info(f"[PACKAGING-SEPARATE] Using AI-inferred packaging: {packaging_materials}")
        
        # Extract material names
        materials = []
        if packaging_tags:
            materials = packaging_tags
        elif packaging_text:
            # Simple extraction - split by common delimiters
            for delimiter in [',', '\n', '-', '•']:
                if delimiter in packaging_text:
                    materials = [m.strip() for m in packaging_text.split(delimiter) if m.strip()]
                    break
            if not materials:
                materials = [packaging_text.strip()]
        
        logger.info(f"[PACKAGING-SEPARATE] Extracted {len(materials)} materials: {materials}")
        
        # EXTRACT PRODUCT CONTEXT for detailed analysis
        # Note: OpenFacts may return empty strings, so we convert them to None
        brand_name = product_data.get("brands", "").strip() or None
        categories = product_data.get("categories", "").strip() or None
        categories_tags = product_data.get("categories_tags", [])
        product_name = (product_data.get("product_name", "").strip() or 
                       product_data.get("name", "").strip() or None)
        
        # Extract food contact info from packagings array
        food_contact = None
        packagings_data = product_data.get("packagings", [])
        if packagings_data and isinstance(packagings_data, list):
            # Check if any packaging has food contact
            for pkg in packagings_data:
                if pkg.get("food_contact") == 1:
                    food_contact = True
                    break
        
        logger.info(f"[PACKAGING-SEPARATE] Product context extracted:")
        logger.info(f"  - Brand: '{brand_name}'")
        logger.info(f"  - Product: '{product_name}'")
        logger.info(f"  - Categories: '{categories}'")
        logger.info(f"  - Food contact: {food_contact}")
        logger.info(f"  - Packagings data count: {len(packagings_data) if packagings_data else 0}")
        
        # NORMALIZE MATERIAL NAMES WITH AI (FAST - minimal tokens)
        normalized_materials = materials  # fallback
        if materials:
            try:
                packaging_info = f"Packaging: {packaging_text}. Tags: {', '.join(packaging_tags)}" if packaging_tags else packaging_text
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                normalize_prompt = f"""Extract and normalize the packaging material names from this text:

{packaging_info}

Return ONLY a JSON object with a "materials" array of clean, English material names.
Translate non-English names to English. Use simple, clear names.

Example input: "Plástico,Invólucro"
Example output: {{"materials": ["Plastic", "Wrapper"]}}

JSON only:"""

                normalize_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a packaging material expert. Return only JSON."},
                        {"role": "user", "content": normalize_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=100,
                    response_format={"type": "json_object"}
                )
                
                normalize_content = normalize_response.choices[0].message.content.strip()
                
                # Parse normalized names
                try:
                    normalized_data = json.loads(normalize_content)
                    normalized_materials = normalized_data.get("materials", materials)
                    logger.info(f"[PACKAGING-SEPARATE] Normalized material names: {normalized_materials}")
                except:
                    logger.warning(f"[PACKAGING-SEPARATE] Failed to parse normalized materials, using original")
                    normalized_materials = materials
            except Exception as e:
                logger.error(f"[PACKAGING-SEPARATE] Error normalizing materials: {e}")
                normalized_materials = materials
        
        return {
            'success': True,
            'has_packaging_data': len(normalized_materials) > 0,
            'materials': normalized_materials,
            'packaging_text': packaging_text,
            'packaging_tags': packaging_tags,
            'source': packaging_source,
            'product_context': {
                'brand_name': brand_name,
                'product_name': product_name,
                'categories': categories,
                'categories_tags': categories_tags,
                'food_contact': food_contact,
                'packagings_data': packagings_data
            },
            'message': f'Extracted {len(normalized_materials)} packaging materials'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during packaging separation: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting packaging materials: {str(e)}")


@router.post("/barcode/packaging/describe")
async def describe_packaging(request: PackagingDescribeRequest) -> Dict[str, Any]:
    """
    SLOW: Generate detailed analysis for packaging materials.
    Takes 3-5 seconds. Call after /separate to get detailed descriptions.
    
    Args:
        request: Contains packaging text, tags, and normalized_materials from /separate
        
    Returns:
        Detailed AI-generated analysis for each material
    """
    try:
        barcode = request.barcode.strip()
        
        if not barcode or not barcode.isdigit():
            raise HTTPException(status_code=400, detail="Invalid barcode")
        
        # Use normalized materials from /separate route (for consistency)
        normalized_materials = request.normalized_materials or request.packaging_tags or []
        
        # Clean up string 'None' values that might come from frontend
        brand_name = normalize_none_string(request.brand_name)
        product_name = normalize_none_string(request.product_name)
        categories = normalize_none_string(request.categories)
        food_contact = request.food_contact if request.food_contact is not None else None
        
        logger.info(f"[PACKAGING-DESCRIBE] Analyzing {len(normalized_materials)} packaging materials: {normalized_materials}")
        logger.info(f"[PACKAGING-DESCRIBE] Received product context (after cleanup):")
        logger.info(f"  - Brand: {repr(brand_name)}")
        logger.info(f"  - Product: {repr(product_name)}")
        logger.info(f"  - Categories: {repr(categories)}")
        logger.info(f"  - Food contact: {food_contact}")
        
        # ANALYZE PACKAGING with pre-normalized materials and product context (SLOW - 1 AI call)
        packaging_analysis = analyze_packaging_material(
            request.packaging_text, 
            request.packaging_tags,
            normalized_materials=normalized_materials,
            brand_name=brand_name,
            product_name=product_name,
            categories=categories,
            food_contact=food_contact
        )
        
        logger.info(f"[PACKAGING-DESCRIBE] Analysis complete")
        
        return {
            'success': True,
            'analysis': packaging_analysis,
            'message': 'Packaging analysis complete'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating packaging descriptions: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing packaging: {str(e)}")


@router.post("/barcode/packaging/analyze")
async def analyze_packaging(request: PackagingAnalyzeRequest) -> Dict[str, Any]:
    """
    MEDIUM: AI-powered packaging analysis.
    Takes 3-7 seconds. Call after /barcode/lookup.
    
    Args:
        request: Contains barcode and optional product_data
        
    Returns:
        Packaging materials, safety analysis, recyclability
    """
    try:
        barcode = request.barcode.strip()
        
        if not barcode or not barcode.isdigit():
            raise HTTPException(status_code=400, detail="Invalid barcode")
        
        logger.info(f"[PACKAGING] Analyzing packaging for barcode: {barcode}")
        
        # Get product data (use provided or fetch fresh)
        if request.product_data:
            product_data = request.product_data
            logger.info("[PACKAGING] Using provided product data (no API call)")
        else:
            barcode_service = get_barcode_service()
            product_data = await barcode_service.lookup_barcode(barcode)
            if not product_data:
                raise HTTPException(status_code=404, detail="Product not found")
        
        # Extract packaging info
        packaging_text = product_data.get("packaging", "")
        packaging_tags = product_data.get("packaging_tags", [])
        
        # Check if packaging info is nested in materials object
        if not packaging_text and "materials" in product_data:
            materials = product_data["materials"]
            packaging_text = materials.get("packaging_text", "") or materials.get("packaging", "")
            packaging_tags = materials.get("packaging_tags", [])
        
        logger.info(f"[PACKAGING] Packaging text: '{packaging_text}', Tags: {packaging_tags}")
        
        # INFER PACKAGING IF MISSING
        packaging_source = "openfacts"
        if not packaging_text and not packaging_tags:
            logger.info("[PACKAGING] No data from OpenFacts - attempting web search + AI inference")
            
            # Try web search first
            web_packaging = await search_packaging_info(
                product_name=product_data.get("name", "Unknown"),
                brand=product_data.get("brands", ""),
                category=product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General"
            )
            
            if web_packaging:
                packaging_text = web_packaging
                packaging_source = "web_search"
                logger.info(f"[PACKAGING] Using web-searched packaging: {packaging_text}")
            else:
                # Fallback to AI category-based inference
                category = product_data.get("categories", "").split(",")[0] if product_data.get("categories") else "General product"
                packaging_inference = await infer_packaging_from_category(
                    product_name=product_data.get("name", "Unknown"),
                    category=category
                )
                if packaging_inference:
                    packaging_text = packaging_inference
                    packaging_source = "ai_inferred"
                    logger.info(f"[PACKAGING] Using AI-inferred packaging: {packaging_text}")
        
        # ANALYZE PACKAGING
        if packaging_text or packaging_tags:
            logger.info(f"[PACKAGING] Analyzing packaging materials")
            packaging_analysis = analyze_packaging_material(packaging_text, packaging_tags)
            packaging_analysis["source"] = packaging_source
            logger.info(f"[PACKAGING] Analysis complete: {len(packaging_analysis.get('materials', []))} materials")
            
            return {
                'success': True,
                'has_packaging_data': True,
                'analysis': packaging_analysis,
                'message': 'Packaging analysis complete'
            }
        else:
            logger.info("[PACKAGING] No packaging data available")
            return {
                'success': True,
                'has_packaging_data': False,
                'analysis': {
                    "materials": [],
                    "analysis": {},
                    "overall_safety": "unknown",
                    "summary": "No packaging information available",
                    "source": "none"
                },
                'message': 'No packaging information available'
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during packaging analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing packaging: {str(e)}")


async def generate_harmful_descriptions(harmful_ingredients: list) -> dict:
    """Generate descriptions for harmful ingredients"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        harmful_str = ", ".join(harmful_ingredients[:15])
        
        prompt = f"""For these harmful ingredients, provide brief descriptions:
{harmful_str}

Return JSON only: {{"ingredient_name": "brief harmful description"}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"Failed to generate harmful descriptions: {e}")
        return {}


async def generate_safe_descriptions(safe_ingredients: list) -> dict:
    """Generate descriptions for safe ingredients"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        safe_str = ", ".join(safe_ingredients[:15])
        
        prompt = f"""For these safe ingredients, provide brief descriptions:
{safe_str}

Return JSON only: {{"ingredient_name": "brief description"}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"Failed to generate safe descriptions: {e}")
        return {}



@router.post("/lookup-barcode-basic")
async def lookup_barcode_basic(request: BarcodeLookupRequest) -> Dict[str, Any]:
    """
    Fast barcode lookup returning only essential product data.
    Use this for initial page load, then fetch detailed analysis asynchronously.
    
    This endpoint:
    - Returns in ~1-3 seconds (vs 20-30s for full endpoint)
    - Provides: name, brand, image, basic safety score
    - Skips: AI descriptions, packaging analysis, recommendations
    
    Args:
        request: Barcode lookup request containing the barcode string
        
    Returns:
        Basic product information with minimal processing
    """
    try:
        barcode = request.barcode.strip()
        
        # Validate barcode
        if not barcode:
            raise HTTPException(
                status_code=400,
                detail="Barcode cannot be empty"
            )
        
        if not barcode.isdigit():
            raise HTTPException(
                status_code=400,
                detail="Barcode must contain only digits"
            )
        
        logger.info(f"Basic lookup for barcode: {barcode}")
        
        # Query barcode service (with caching)
        barcode_service = get_barcode_service()
        product_data = await barcode_service.lookup_barcode(barcode)
        
        if not product_data:
            return {
                'success': True,
                'found': False,
                'product': None,
                'message': 'Product not found in database'
            }
        
        # Extract basic ingredients for safety check only
        ingredients_text = product_data.get("ingredients_text", "")
        
        # Quick chemical check (no AI processing)
        detected_chemicals = []
        safety_score = 100
        
        if ingredients_text:
            detected_chemicals = check_ingredients(ingredients_text)
            safety_score = calculate_safety_score(detected_chemicals)
        
        # Return minimal product data
        basic_product = {
            "barcode": product_data.get("barcode", barcode),
            "name": product_data.get("name", "Unknown Product"),
            "brands": product_data.get("brands", ""),
            "categories": product_data.get("categories", ""),
            "image_url": product_data.get("image_url", ""),
            "source": product_data.get("source", ""),
            "safety_score": safety_score,
            "has_harmful_chemicals": len(detected_chemicals) > 0,
            "harmful_count": len(detected_chemicals),
            "labels": product_data.get("labels", ""),
            "url": product_data.get("url", ""),
        }
        
        logger.info(f"Basic lookup complete for {barcode}: {basic_product['name']}")
        
        return {
            'success': True,
            'found': True,
            'product': basic_product,
            'message': f'Product found: {basic_product["name"]}',
            'note': 'This is basic data. Call /lookup-barcode for full analysis.'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during basic barcode lookup: {e}")
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
        # Use pre-fetched product data if provided, otherwise fetch from barcode
        if request.product_data:
            product_data = request.product_data
            logger.info(f"Using pre-fetched product data for barcode {request.barcode} (avoiding redundant API call)")
        else:
            barcode_service = get_barcode_service()
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
