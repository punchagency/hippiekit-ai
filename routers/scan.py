from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from typing import List, Dict, Any
from pydantic import BaseModel

from models import get_clip_embedder
from services import get_pinecone_service
from services.barcode_service import get_barcode_service

router = APIRouter()


class BarcodeLookupRequest(BaseModel):
    """Request model for barcode lookup"""
    barcode: str

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
    Look up a product by barcode using Open*Facts databases.
    
    Args:
        request: Barcode lookup request containing the barcode string
        
    Returns:
        Dictionary with product information from Open Food Facts, 
        Open Beauty Facts, or Open Product Facts
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
        
        print(f"Looking up barcode: {barcode}")
        
        # Query barcode service
        barcode_service = get_barcode_service()
        product_data = barcode_service.lookup_barcode(barcode)
        
        if product_data:
            return {
                'success': True,
                'found': True,
                'product': product_data,
                'message': f'Product found: {product_data.get("name", "Unknown")}'
            }
        else:
            return {
                'success': True,
                'found': False,
                'product': None,
                'message': 'Product not found in database. Try scanning the product image instead.'
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during barcode lookup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error looking up barcode: {str(e)}"
        )
