from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from typing import List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import logging

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


def get_detailed_ingredient_descriptions(ingredients_list: list, harmful_chemicals: list) -> dict:
    """
    Generate detailed, user-friendly AI descriptions for ingredients.
    Safe ingredients: Explain what they are and what they're used for (2-3 sentences)
    Harmful ingredients: Explain what they are AND why they're harmful (2-3 sentences)
    
    Returns:
        {
            "safe": {"ingredient_name": "detailed description"},
            "harmful": {"chemical_name": "detailed harmful description"}
        }
    """
    try:
        # Ensure all ingredients are strings and extract harmful chemical names (lowercase for comparison)
        harmful_names_lower = set()  # Use set for faster lookup
        harmful_names_original = {}  # Map lowercase to original name
        for chem in harmful_chemicals:
            if isinstance(chem, dict) and 'chemical' in chem:
                chem_lower = str(chem['chemical']).lower().strip()
                harmful_names_lower.add(chem_lower)
                harmful_names_original[chem_lower] = str(chem['chemical']).strip()
        
        # Filter safe ingredients, ensuring all are strings and NOT in harmful list
        safe_ingredients = []
        for ing in ingredients_list:
            # Convert to string and strip whitespace
            ing_str = str(ing).strip()
            ing_lower = ing_str.lower()
            # Only add to safe if it's not empty and not in harmful list
            if ing_str and ing_lower not in harmful_names_lower:
                safe_ingredients.append(ing_str)
        
        # Build harmful descriptions - explain what it is AND why it's harmful
        harmful_descriptions = {}
        if harmful_chemicals:
            try:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                # Create prompt for harmful chemicals with category context
                harmful_items = [
                    f"{chem['chemical']} (Category: {chem['category']}, Severity: {chem['severity']})"
                    for chem in harmful_chemicals[:15]  # Limit to avoid token limits
                ]
                harmful_str = ", ".join(harmful_items)
                
                harmful_prompt = f"""Provide detailed descriptions for these harmful chemicals/ingredients found in consumer products.
For each chemical, explain in 2-3 sentences:
1. What the chemical is and its common use in products
2. Why it's harmful to human health and/or the environment
3. Specific health risks or environmental concerns

IMPORTANT: Use the EXACT chemical names provided as the keys in your JSON response.

Format as JSON: {{"Chemical Name Exactly As Listed": "description"}}

Harmful chemicals: {harmful_str}
"""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a health and environmental safety expert providing clear, user-friendly explanations about harmful chemicals in consumer products. Always use the exact chemical names provided as JSON keys without modification."},
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
        
        # Generate safe ingredient descriptions
        safe_descriptions = {}
        if safe_ingredients:
            try:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                
                # Process all ingredients (no limit)
                ingredients_str = ", ".join(safe_ingredients)
                logger.info(f"Generating descriptions for {len(safe_ingredients)} safe ingredients")
                
                safe_prompt = f"""Provide detailed, user-friendly descriptions for these food/product ingredients.
For each ingredient, explain in 2-3 sentences:
1. What the ingredient is (natural source, synthetic, etc.)
2. Its common purpose or function in products
3. Any benefits or interesting facts

IMPORTANT: Use the EXACT ingredient names provided as the keys in your JSON response.

Format as JSON: {{"Ingredient Name Exactly As Listed": "description"}}

Ingredients: {ingredients_str}
"""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a food science and product formulation expert providing helpful, educational descriptions of common ingredients. Always use the exact ingredient names provided as JSON keys without modification."},
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

Packaging Information: {packaging_info}

IMPORTANT: Only analyze the materials explicitly mentioned. Do NOT infer or guess specific plastic types unless they are clearly stated. If only generic terms like "Plastic" or "Cardboard" are mentioned, analyze those generic categories.

Provide a detailed analysis covering:
1. List only the materials explicitly mentioned in the packaging information
2. For each material mentioned, explain:
   - What it typically is and its common use in packaging
   - Potential harmful substances commonly associated with this material type (BPA, phthalates, microplastics, etc.)
   - General health concerns for this material category
   - Environmental impact (recyclability, biodegradability, pollution, ocean impact, etc.)
   - Severity rating (low/moderate/high/critical) - if specific type unknown, rate based on typical concerns
3. Overall safety assessment (safe/caution/harmful/unknown)
4. Brief summary of the packaging's overall impact

If only generic materials are listed (e.g., "Plastic"), note in the description that the specific type is unknown and describe general plastic concerns.

Format as JSON:
{{
    "materials": ["material1 exactly as stated", "material2 exactly as stated"],
    "analysis": {{
        "material_name": {{
            "description": "what this material typically is and common use in packaging",
            "harmful": true or false,
            "health_concerns": "potential health risks for this material category or 'Specific type unknown - general plastic concerns apply' if generic",
            "environmental_impact": "recyclability and environmental effects for this material type",
            "severity": "low/moderate/high/critical"
        }}
    }},
    "overall_safety": "safe/caution/harmful/unknown",
    "summary": "2-3 sentence overall assessment, noting if specific material types are unknown"
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
        
        # Parse ingredients and ensure all are strings
        ingredients_list = []
        for ing in ingredients_text.split(","):
            ing = ing.strip()
            if ing:
                # Ensure it's a string (in case of any unexpected data types)
                ingredients_list.append(str(ing))
        
        logger.info(f"Parsed {len(ingredients_list)} ingredients from text")
        
        # Check ingredients for harmful chemicals
        # Note: check_ingredients expects a string, not a list
        harmful_chemicals = []
        if ingredients_text:
            try:
                # check_ingredients returns a list directly, not a dict
                harmful_chemicals = check_ingredients(ingredients_text)
                logger.info(f"Found {len(harmful_chemicals)} harmful chemicals")
            except Exception as e:
                logger.error(f"Error checking ingredients: {e}")
                # Continue without chemical analysis rather than failing
        
        # Calculate safety score
        safety_score = calculate_safety_score(harmful_chemicals)
        
        # Generate recommendations
        recommendations = generate_recommendations(harmful_chemicals)
        
        # Build chemical analysis
        product_data["chemical_analysis"] = {
            "harmful_chemicals": harmful_chemicals,
            "safety_score": safety_score,
            "recommendations": recommendations
        }
        
        # Generate detailed AI ingredient descriptions (safe + harmful)
        if ingredients_list:
            logger.info(f"Generating detailed ingredient descriptions for {len(ingredients_list)} ingredients")
            ingredient_descriptions = get_detailed_ingredient_descriptions(ingredients_list, harmful_chemicals)
            product_data["ingredient_descriptions"] = ingredient_descriptions
        else:
            product_data["ingredient_descriptions"] = {"safe": {}, "harmful": {}}
        
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
        
        if packaging_text or packaging_tags:
            logger.info(f"Analyzing packaging materials: {packaging_text}")
            packaging_analysis = analyze_packaging_material(packaging_text, packaging_tags)
            product_data["packaging_analysis"] = packaging_analysis
            logger.info(f"Packaging analysis complete: {len(packaging_analysis.get('materials', []))} materials identified")
        else:
            logger.info("No packaging information available")
            product_data["packaging_analysis"] = {
                "materials": [],
                "analysis": {},
                "overall_safety": "unknown",
                "summary": "No packaging information available"
            }
        
        logger.info(f"Successfully processed barcode {barcode}: {product_data.get('name', 'Unknown')}")
        
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
