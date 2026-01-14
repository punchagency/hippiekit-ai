"""
Product Identification Router
Identifies products from front-facing photos and retrieves complete information
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from services.vision_service import get_vision_service
from services.barcode_service import BarcodeService
from services.web_search_service import web_search_service
from services.chemical_checker import check_ingredients, calculate_safety_score, generate_recommendations
from routers.scan import get_detailed_ingredient_descriptions, analyze_packaging_material, separate_ingredients_with_ai
from openai import OpenAI
import os
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/identify", tags=["product-identification"])

barcode_service = BarcodeService()


def get_ingredient_descriptions_legacy(ingredients_list: list, harmful_chemicals: list) -> dict:
    """
    Generate AI descriptions for ingredients.
    Safe ingredients: GPT-4o-mini generates brief 1-sentence descriptions
    Harmful ingredients: Generate explanations using get_chemical_explanation
    
    Returns:
        {
            "safe": {"ingredient_name": "description"},
            "harmful": {"chemical_name": "why_flagged"}
        }
    """
    try:
        # Separate harmful chemicals from safe ingredients
        harmful_names = [chem['chemical'].lower() for chem in harmful_chemicals]
        safe_ingredients = [ing.strip() for ing in ingredients_list if ing.strip().lower() not in harmful_names]
        
        # Build harmful descriptions using get_chemical_explanation
        harmful_descriptions = {
            chem['chemical']: get_chemical_explanation(chem['category'])
            for chem in harmful_chemicals
        }
        
        # Generate safe ingredient descriptions using AI
        safe_descriptions = {}
        if safe_ingredients and len(safe_ingredients) > 0:
            try:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                # Prepare prompt
                ingredients_str = ", ".join(safe_ingredients[:20])  # Limit to first 20 to avoid token limits
                
                prompt = f"""Provide brief, simple descriptions for these food/product ingredients. 
                Each description should be 1 sentence, maximum 20 words, explaining what the ingredient is and its common use.
                Format as JSON: {{"ingredient_name": "description"}}
                
                Ingredients: {ingredients_str}
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides brief, accurate descriptions of food and product ingredients."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                
                # Parse JSON response
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                safe_descriptions = json.loads(content)
                logger.info(f"Generated descriptions for {len(safe_descriptions)} safe ingredients")
                
            except Exception as e:
                logger.error(f"Failed to generate ingredient descriptions: {e}")
                # Provide fallback descriptions
                safe_descriptions = {ing: "Common food ingredient" for ing in safe_ingredients[:10]}
        
        return {
            "safe": safe_descriptions,
            "harmful": harmful_descriptions
        }
        
    except Exception as e:
        logger.error(f"Error in get_ingredient_descriptions: {e}")
        return {"safe": {}, "harmful": {}}


@router.post("/product")
async def identify_product(image: UploadFile = File(...)):
    """
    Identify product from a photo and return complete information
    
    Workflow:
    1. Use Vision API to identify product name, brand, category from photo
    2. Search Open*Facts database for matching product
    3. If found but ingredients empty: search web for ingredients
    4. If not found: full AI analysis from image
    5. Perform chemical analysis on ingredients
    6. Return unified response
    
    Returns:
        Complete product information with chemical analysis
    """
    try:
        logger.info(f"Product identification request received: {image.filename}")
        
        # Read image
        image_bytes = await image.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        # Step 1: Identify product from photo using Vision API
        vision_service = get_vision_service()
        product_info = await vision_service.identify_product_from_photo(image_bytes)
        
        if not product_info:
            raise HTTPException(status_code=400, detail="Could not identify product from image")
        
        logger.info(f"Product identified: {product_info.get('brand')} {product_info.get('product_name')}")
        
        # Step 2: Search database for this product
        database_result = await search_database_by_name(
            product_name=product_info['product_name'],
            brand=product_info['brand']
        )
        
        # Step 3: Determine data source and get ingredients
        if database_result:
            logger.info("Product found in database")
            
            # Check if database has ingredients
            ingredients_text = database_result.get('ingredients_text', '')
            
            if not ingredients_text or len(ingredients_text.strip()) < 10:
                logger.info("Database product has no ingredients, searching web...")
                
                # Search web for ingredients
                web_result = await web_search_service.search_product_ingredients(
                    product_name=product_info['product_name'],
                    brand=product_info['brand'],
                    category=product_info.get('category')
                )
                
                if web_result and web_result.get('ingredients'):
                    ingredients_text = web_result['ingredients']
                    data_source = web_result['source']
                    confidence = web_result['confidence']
                    ingredients_note = web_result['note']
                else:
                    # No web results, use generic category info
                    data_source = 'category_generic'
                    confidence = 'low'
                    ingredients_note = "Could not find specific ingredient information"
            else:
                # Database has ingredients
                data_source = 'database'
                confidence = 'high'
                ingredients_note = None
            
            # Merge database info with vision info
            response = await merge_database_and_vision(
                database_result, 
                product_info, 
                ingredients_text,
                data_source,
                confidence,
                ingredients_note
            )
        else:
            logger.info("Product not in database, using AI-only analysis")
            
            # Step 4: No database match - do full AI analysis
            response = await ai_only_analysis(image_bytes, product_info)
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product identification failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


async def search_database_by_name(product_name: str, brand: str):
    """
    Search Open*Facts database by product name and brand
    Note: This is a simplified search - actual implementation would need
    to query the database search API
    """
    # TODO: Implement actual database search by name
    # For now, return None (database search by name not implemented)
    logger.info(f"Database search by name not yet implemented for: {brand} {product_name}")
    return None


async def merge_database_and_vision(
    db_data: dict,
    vision_data: dict,
    ingredients_text: str,
    data_source: str,
    confidence: str,
    ingredients_note: str = None
) -> dict:
    """Merge database product data with vision identification"""
    
    # Chemical analysis
    chemical_flags = check_ingredients(ingredients_text) if ingredients_text else []
    safety_score = calculate_safety_score(chemical_flags)
    category = db_data.get('categories', '').split(',')[0] if db_data.get('categories') else vision_data.get('category')
    recommendations = generate_recommendations(chemical_flags, category)
    
    # Get ingredient descriptions with AI separation (ensures no duplicates between safe/harmful)
    ingredients_list = [ing.strip() for ing in ingredients_text.split(',')] if ingredients_text else []
    
    if ingredients_list:
        logger.info(f"Separating {len(ingredients_list)} ingredients with AI intelligent matching")
        
        # Use AI to intelligently separate harmful vs safe (handles name variations automatically)
        # Pass the original text, not the list
        separated = await separate_ingredients_with_ai(ingredients_text, chemical_flags)
        
        harmful_ingredient_names = separated.get("harmful", [])
        safe_ingredient_names = separated.get("safe", [])
        
        logger.info(f"AI separation complete: {len(harmful_ingredient_names)} harmful, {len(safe_ingredient_names)} safe")
        
        # Generate detailed descriptions using AI-separated lists
        ingredient_descriptions = get_detailed_ingredient_descriptions(
            safe_ingredients=safe_ingredient_names,
            harmful_ingredients=harmful_ingredient_names,
            harmful_chemicals_db=chemical_flags
        )
    else:
        harmful_ingredient_names = []
        safe_ingredient_names = []
        ingredient_descriptions = {"safe": {}, "harmful": {}}
    
    # Analyze packaging materials with OpenAI web search fallback
    packaging_text = db_data.get('packaging', '') or vision_data.get('container_info', {}).get('material', '')
    packaging_tags = db_data.get('packaging_tags', [])
    packaging_source = "database"
    
    # If no packaging info, try OpenAI web search
    if not packaging_text and not packaging_tags:
        product_name = db_data.get('product_name') or vision_data['product_name']
        brand = db_data.get('brands') or vision_data['brand']
        category = db_data.get('categories') or vision_data['category']
        
        logger.info(f"No packaging in database, searching with OpenAI web search: {brand} {product_name}")
        packaging_text = await search_packaging_info(product_name, brand, category)
        
        if packaging_text:
            packaging_source = "web_search"
            logger.info(f"OpenAI web search found packaging: {packaging_text}")
    
    if packaging_text or packaging_tags:
        logger.info(f"Analyzing packaging materials: {packaging_text}")
        packaging_analysis = analyze_packaging_material(packaging_text, packaging_tags)
        logger.info(f"Packaging analysis complete: {len(packaging_analysis.get('materials', []))} materials identified")
    else:
        logger.info("No packaging information available")
        packaging_analysis = {
            "materials": [],
            "analysis": {},
            "overall_safety": "unknown",
            "summary": "No packaging information available"
        }
    
    response = {
        "product_name": db_data.get('product_name') or vision_data['product_name'],
        "brand": db_data.get('brands') or vision_data['brand'],
        "category": db_data.get('categories') or vision_data['category'],
        "product_type": vision_data.get('product_type', ''),
        "image_url": db_data.get('image_url', ''),
        "barcode": db_data.get('code', ''),
        "ingredients": safe_ingredient_names,  # Only safe ingredients
        "harmful_ingredients": harmful_ingredient_names,  # Only harmful ingredients
        "nutrition": db_data.get('nutrition', {}),
        "marketing_claims": vision_data.get('marketing_claims', []),
        "certifications_visible": vision_data.get('certifications_visible', []),
        "container_info": vision_data.get('container_info', {}),
        "packaging": packaging_text if packaging_source == "web_search" else db_data.get('packaging', ''),
        "quantity": db_data.get('quantity', '') or vision_data.get('container_info', {}).get('size', ''),
        "data_source": data_source,
        "confidence": confidence,
        "packaging_source": packaging_source,
        "chemical_analysis": {
            "flags": [
                {
                    "chemical": flag["chemical"],
                    "category": flag["category"],
                    "severity": flag["severity"],
                    "why_flagged": get_chemical_explanation(flag["category"])
                }
                for flag in chemical_flags
            ],
            "safety_score": safety_score,
            "recommendations": recommendations,
            "data_source": data_source,
            "confidence": confidence
        },
        "ingredient_descriptions": ingredient_descriptions,
        "packaging_analysis": packaging_analysis
    }
    
    if ingredients_note:
        response["ingredients_note"] = ingredients_note
    
    return response


async def ai_only_analysis(image_bytes: bytes, product_info: dict) -> dict:
    """
    AI-only analysis when product not in database.
    Uses web search to find ingredients if product identified.
    """
    logger.info("Product not in database, using AI-only analysis with web search")
    
    # Try web search first if we have product name and brand
    product_name = product_info.get('product_name', '')
    brand = product_info.get('brand', '')
    category = product_info.get('category', '')
    
    ingredients_text = None
    data_source = 'ai_vision'
    confidence = 'medium'
    ingredients_note = None
    
    # Try web search if we have at least a product name (brand is optional)
    # Handle cases where brand is "Not visible" or empty
    search_brand = brand if brand and brand.lower() not in ['not visible', 'unknown', ''] else ''
    
    if product_name and product_name.lower() not in ['not visible', 'unknown', '']:
        logger.info(f"Searching web for ingredients: {search_brand} {product_name}")
        web_result = await web_search_service.search_product_ingredients(
            product_name=product_name,
            brand=search_brand,
            category=category
        )
        
        if web_result and web_result.get('ingredients'):
            ingredients_text = web_result['ingredients']
            data_source = web_result['source']
            confidence = web_result['confidence']
            ingredients_note = web_result.get('note')
            logger.info(f"Web search successful: {data_source} (confidence: {confidence})")
    
    # If no ingredients from web search, fall back to OCR
    if not ingredients_text:
        logger.info("Web search failed or not attempted, using OCR analysis")
        vision_service = get_vision_service()
        full_analysis = vision_service.analyze_product_image(image_bytes)
        
        if full_analysis:
            # Return OCR analysis
            full_analysis['data_source'] = 'ai_vision'
            full_analysis['confidence'] = product_info.get('confidence', 'medium')
            return full_analysis
        else:
            ingredients_text = "Could not extract ingredients from image"
            data_source = 'none'
            confidence = 'low'
    
    # Build response with web search results
    chemical_flags = check_ingredients(ingredients_text) if ingredients_text and ingredients_text != "Could not extract ingredients from image" else []
    safety_score = calculate_safety_score(chemical_flags)
    recommendations = generate_recommendations(chemical_flags, category)
    
    # Get ingredient descriptions with AI separation (ensures no duplicates between safe/harmful)
    ingredients_list = [ing.strip() for ing in ingredients_text.split(',')] if ingredients_text and ingredients_text != "Could not extract ingredients from image" else []
    
    if ingredients_list:
        logger.info(f"Separating {len(ingredients_list)} ingredients with AI intelligent matching")
        
        # Use AI to intelligently separate harmful vs safe (handles name variations automatically)
        # Pass the original text, not the list
        separated = await separate_ingredients_with_ai(ingredients_text, chemical_flags)
        
        harmful_ingredient_names = separated.get("harmful", [])
        safe_ingredient_names = separated.get("safe", [])
        
        logger.info(f"AI separation complete: {len(harmful_ingredient_names)} harmful, {len(safe_ingredient_names)} safe")
        
        # Generate detailed descriptions using AI-separated lists
        ingredient_descriptions = get_detailed_ingredient_descriptions(
            safe_ingredients=safe_ingredient_names,
            harmful_ingredients=harmful_ingredient_names,
            harmful_chemicals_db=chemical_flags
        )
    else:
        harmful_ingredient_names = []
        safe_ingredient_names = []
        ingredient_descriptions = {"safe": {}, "harmful": {}}
    
    # Analyze packaging materials with OpenAI web search fallback
    packaging_text = product_info.get('container_info', {}).get('material', '')
    packaging_tags = []
    packaging_source = "ai_vision"
    
    # If no packaging info, try OpenAI web search
    if not packaging_text:
        logger.info(f"No packaging from vision, searching with OpenAI web search: {brand} {product_name}")
        packaging_text = await search_packaging_info(product_name, brand, category)
        
        if packaging_text:
            packaging_source = "web_search"
            logger.info(f"OpenAI web search found packaging: {packaging_text}")
    
    if packaging_text:
        logger.info(f"Analyzing packaging materials: {packaging_text}")
        packaging_analysis = analyze_packaging_material(packaging_text, packaging_tags)
        logger.info(f"Packaging analysis complete: {len(packaging_analysis.get('materials', []))} materials identified")
    else:
        logger.info("No packaging information available")
        packaging_analysis = {
            "materials": [],
            "analysis": {},
            "overall_safety": "unknown",
            "summary": "No packaging information available"
        }
    
    response = {
        "product_name": product_info.get('product_name', ''),
        "brand": product_info.get('brand', ''),
        "category": product_info.get('category', ''),
        "product_type": product_info.get('product_type', ''),
        "ingredients": safe_ingredient_names,  # Only safe ingredients
        "harmful_ingredients": harmful_ingredient_names,  # Only harmful ingredients
        "marketing_claims": product_info.get('marketing_claims', []),
        "certifications_visible": product_info.get('certifications_visible', []),
        "container_info": product_info.get('container_info', {}),
        "packaging": packaging_text if packaging_source == "web_search" else "",
        "data_source": data_source,
        "confidence": confidence,
        "packaging_source": packaging_source,
        "chemical_analysis": {
            "flags": [
                {
                    "chemical": flag["chemical"],
                    "category": flag["category"],
                    "severity": flag["severity"],
                    "why_flagged": get_chemical_explanation(flag["category"])
                }
                for flag in chemical_flags
            ],
            "safety_score": safety_score,
            "recommendations": recommendations,
            "data_source": data_source,
            "confidence": confidence
        },
        "ingredient_descriptions": ingredient_descriptions,
        "packaging_analysis": packaging_analysis
    }
    
    if ingredients_note:
        response["ingredients_note"] = ingredients_note
    
    return response


def get_chemical_explanation(category: str) -> str:
    """Get explanation for chemical category"""
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
    }
    return explanations.get(category, f"{category} chemical with potential health concerns")


@router.post("/product/recommendations")
async def get_product_recommendations(
    product_name: str = Form(None),
    brand: str = Form(None),
    category: str = Form(None),
    ingredients: str = Form(None),
    marketing_claims: str = Form(None),
    certifications: str = Form(None),
    product_type: str = Form(None),
    image: UploadFile = File(None)
):
    """
    Get product recommendations for a photo-identified product.
    Uses multimodal search (text + image) with ALL OCR-extracted data.
    
    Args:
        product_name: Product name
        brand: Brand name
        category: Product category (optional)
        ingredients: Comma-separated ingredients from OCR
        marketing_claims: Marketing claims (e.g., "Organic, Non-GMO")
        certifications: Visible certifications on packaging
        product_type: Product type classification
        image: The original scanned image (optional, for multimodal search)
        
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
        from routers.scan import get_product_recommendations_with_image
        from PIL import Image
        from io import BytesIO
        
        if not product_name:
            return {
                'success': False,
                'recommendations': None,
                'message': 'Product name is required'
            }
        
        # If image is provided, read it for multimodal search
        image_data = None
        if image:
            image_bytes = await image.read()
            image_data = Image.open(BytesIO(image_bytes))
            if image_data.mode != 'RGB':
                image_data = image_data.convert('RGB')
            logger.info(f"Using uploaded image for multimodal recommendations")
        
        # Get recommendations with rich OCR data
        recommendations = await get_product_recommendations_with_image(
            product_name=product_name,
            brand=brand or '',
            category=category or '',
            ingredients=ingredients or '',
            marketing_claims=marketing_claims or '',
            certifications=certifications or '',
            product_type=product_type or '',
            image=image_data,
            top_k=3,
            min_score=0.4
        )
        
        return {
            'success': True,
            'recommendations': recommendations,
            'message': 'Recommendations generated successfully'
        }
        
    except Exception as e:
        logger.error(f"Error getting product recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting recommendations: {str(e)}"
        )


# ===== MODULAR ENDPOINTS FOR PROGRESSIVE LOADING =====

@router.post("/product/basic")
async def identify_product_basic(image: UploadFile = File(...)):
    """
    Step 1: Get basic product info from image (FAST - 1-2s)
    Returns only product name, brand, category, type, and marketing claims
    
    This is the first call in progressive loading - provides instant feedback
    """
    try:
        logger.info(f"Basic product identification request: {image.filename}")
        
        # Read image
        image_bytes = await image.read()
        
        # Use Vision API to identify product
        vision_service = get_vision_service()
        product_info = await vision_service.identify_product_from_photo(image_bytes)
        
        if not product_info:
            raise HTTPException(status_code=400, detail="Could not identify product from image")
        
        logger.info(f"Product identified: {product_info.get('brand')} {product_info.get('product_name')}")
        
        # Return basic info only
        return {
            "product_name": product_info['product_name'],
            "brand": product_info['brand'],
            "category": product_info.get('category', ''),
            "product_type": product_info.get('product_type', ''),
            "marketing_claims": product_info.get('marketing_claims', []),
            "certifications_visible": product_info.get('certifications_visible', []),
            "container_info": product_info.get('container_info', {}),
            "confidence": product_info.get('confidence', 'medium')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Basic product identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingredients/separate")
async def separate_photo_ingredients(
    product_name: str = Form(...),
    brand: str = Form(...),
    category: str = Form(None)
):
    """
    Step 2: Separate ingredients into harmful/safe (FAST - 2-3s)
    Uses web search to find ingredients, then AI to separate them
    
    Returns just the ingredient names, descriptions come later
    """
    try:
        logger.info(f"Separating ingredients for: {brand} {product_name}")
        
        # Search web for ingredients
        web_result = await web_search_service.search_product_ingredients(
            product_name=product_name,
            brand=brand,
            category=category
        )
        
        ingredients_text = ""
        data_source = "web_search"
        
        if web_result and web_result.get('ingredients'):
            ingredients_text = web_result['ingredients']
            data_source = web_result['source']
        else:
            logger.warning(f"No ingredients found for {brand} {product_name}")
            return {
                "harmful": [],
                "safe": [],
                "data_source": "none",
                "message": "Could not find ingredient information"
            }
        
        # Check for harmful chemicals
        chemical_flags = check_ingredients(ingredients_text)
        
        # Use AI to separate ingredients (handles name variations)
        separated = await separate_ingredients_with_ai(ingredients_text, chemical_flags)
        
        logger.info(f"Separated: {len(separated.get('harmful', []))} harmful, {len(separated.get('safe', []))} safe")
        
        return {
            "harmful": separated.get("harmful", []),
            "safe": separated.get("safe", []),
            "data_source": data_source,
            "ingredients_text": ingredients_text  # Return for future calls
        }
        
    except Exception as e:
        logger.error(f"Failed to separate ingredients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingredients/describe")
async def describe_photo_ingredients(
    harmful_ingredients: str = Form(""),  # Comma-separated list (optional)
    safe_ingredients: str = Form("")  # Comma-separated list (optional)
):
    """
    Step 3: Get AI descriptions for ingredients (SLOWER - 3-5s)
    
    This is called after ingredients are separated and displayed
    Progressively enhances the UI with detailed descriptions
    """
    try:
        # Parse comma-separated lists
        harmful_list = [ing.strip() for ing in harmful_ingredients.split(',') if ing.strip()]
        safe_list = [ing.strip() for ing in safe_ingredients.split(',') if ing.strip()]
        
        # If no ingredients provided, return empty descriptions
        if not harmful_list and not safe_list:
            logger.warning("No ingredients provided for description")
            return {
                "descriptions": {
                    "harmful": {},
                    "safe": {}
                }
            }
        
        logger.info(f"Generating descriptions for {len(harmful_list)} harmful + {len(safe_list)} safe ingredients")
        
        # Generate descriptions
        descriptions = get_detailed_ingredient_descriptions(
            safe_ingredients=safe_list,
            harmful_ingredients=harmful_list,
            harmful_chemicals_db=[]  # Descriptions don't need chemical flags
        )
        
        return {
            "descriptions": descriptions
        }
        
    except Exception as e:
        logger.error(f"Failed to describe ingredients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/packaging/separate")
async def separate_photo_packaging(
    product_name: str = Form(...),
    brand: str = Form(...),
    category: str = Form(None)
):
    """
    Step 4: Get packaging material names (FAST - 1-2s)
    Uses web search to find packaging info
    
    Returns just the material names, descriptions come later
    """
    try:
        logger.info(f"Finding packaging for: {brand} {product_name}")
        
        # Use web_search_service instead of broken search_packaging_info
        web_result = await web_search_service.search_product_packaging(
            product_name=product_name,
            brand=brand,
            category=category
        )
        
        if not web_result or not web_result.get('packaging'):
            logger.warning(f"No packaging found for {brand} {product_name}")
            return {
                "materials": [],
                "packaging_text": "",
                "message": "No packaging information found"
            }
        
        # Extract data from web_search_service result
        packaging_text = web_result.get('packaging', '')
        materials = web_result.get('materials', [])
        
        logger.info(f"Found packaging materials: {materials}")
        logger.info(f"Sources: {len(web_result.get('sources', []))}")
        
        return {
            "materials": materials,
            "packaging_text": packaging_text,
            "sources": web_result.get('sources', []),
            "confidence": web_result.get('confidence', 'medium')
        }
        
    except Exception as e:
        logger.error(f"Failed to find packaging: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/packaging/describe")
async def describe_photo_packaging(
    packaging_text: str = Form(...),
    materials: str = Form(...),  # Comma-separated list
    brand_name: str = Form(None),
    product_name: str = Form(None),
    category: str = Form(None)
):
    """
    Step 5: Get detailed packaging analysis (SLOWER - 2-4s)
    
    This is called after material names are displayed
    Progressively enhances the UI with safety analysis
    
    Args:
        packaging_text: Raw packaging description from web search
        materials: Comma-separated list of material names
        brand_name: Product brand (optional, improves analysis)
        product_name: Product name (optional, improves analysis)
        category: Product category (optional, improves analysis)
    """
    try:
        # Parse materials list
        materials_list = [mat.strip() for mat in materials.split(',') if mat.strip()]
        
        logger.info(f"Analyzing packaging for {brand_name} {product_name}: {materials_list}")
        
        # Analyze packaging with AI - pass product context for better descriptions
        packaging_analysis = analyze_packaging_material(
            packaging_text=packaging_text,
            packaging_tags=materials_list,
            normalized_materials=materials_list,
            brand_name=brand_name,
            product_name=product_name,
            categories=category
        )
        
        return packaging_analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze packaging: {e}")
        raise HTTPException(status_code=500, detail=str(e))
