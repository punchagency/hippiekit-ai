"""
Test script for Product Identification System
Tests all tiers of ingredient lookup
"""

import asyncio
from services.web_search_service import web_search_service
from services.vision_service import get_vision_service
from services.chemical_checker import check_ingredients, calculate_safety_score
import json


async def test_web_search():
    """Test web search service with different scenarios"""
    
    print("\n" + "="*80)
    print("TESTING WEB SEARCH SERVICE")
    print("="*80)
    
    test_cases = [
        {
            "name": "Popular product - should find on web",
            "product_name": "Hydrating Facial Cleanser",
            "brand": "CeraVe",
            "category": "Facial Cleanser"
        },
        {
            "name": "Generic category - should use template",
            "product_name": "Unknown Brand Body Wash",
            "brand": "Unknown",
            "category": "Body Wash"
        },
        {
            "name": "Well-known product - AI knowledge",
            "product_name": "Original Beauty Bar",
            "brand": "Dove",
            "category": "Soap"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        print(f"Product: {test['brand']} {test['product_name']}")
        print(f"Category: {test['category']}")
        
        result = await web_search_service.search_product_ingredients(
            product_name=test['product_name'],
            brand=test['brand'],
            category=test['category']
        )
        
        if result:
            print(f"\n✅ SUCCESS")
            print(f"Source: {result['source']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Ingredients: {result['ingredients'][:200]}...")
            if result.get('note'):
                print(f"Note: {result['note']}")
            
            # Test chemical analysis
            flags = check_ingredients(result['ingredients'])
            score = calculate_safety_score(flags)
            print(f"\nChemical Analysis:")
            print(f"  Flags found: {len(flags)}")
            print(f"  Safety score: {score}/100")
            if flags:
                print(f"  Top concerns:")
                for flag in flags[:3]:
                    print(f"    - {flag['chemical']} ({flag['severity']})")
        else:
            print(f"\n❌ FAILED - No results")
        
        print("-" * 80)


async def test_product_identification():
    """Test product identification (requires actual image)"""
    
    print("\n" + "="*80)
    print("TESTING PRODUCT IDENTIFICATION")
    print("="*80)
    
    print("\nℹ️  This test requires an actual product image.")
    print("To test product identification:")
    print("1. Take a photo of a product front label")
    print("2. Save as 'test_product.jpg' in this directory")
    print("3. Run: python -c 'import test_product_identification; test_product_identification.test_with_image(\"test_product.jpg\")'")
    
    print("\nSkipping image test (no image provided)")


def test_with_image(image_path: str):
    """Test with actual image file"""
    async def _test():
        print(f"\nTesting with image: {image_path}")
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        vision_service = get_vision_service()
        result = await vision_service.identify_product_from_photo(image_bytes)
        
        print("\n✅ Product Identification Result:")
        print(json.dumps(result, indent=2))
    
    asyncio.run(_test())


async def test_chemical_checker():
    """Test chemical detection"""
    
    print("\n" + "="*80)
    print("TESTING CHEMICAL CHECKER")
    print("="*80)
    
    test_ingredients = """
    Water, Sodium Laureth Sulfate, Cocamidopropyl Betaine, Glycerin, 
    Fragrance, Methylparaben, Propylparaben, Triclosan, 
    Sodium Chloride, Citric Acid, DMDM Hydantoin
    """
    
    print(f"\nTest Ingredients: {test_ingredients.strip()}")
    
    flags = check_ingredients(test_ingredients)
    score = calculate_safety_score(flags)
    
    print(f"\n✅ Chemical Analysis:")
    print(f"Total flags: {len(flags)}")
    print(f"Safety score: {score}/100")
    
    if flags:
        print(f"\nFlagged chemicals by severity:")
        
        severities = {}
        for flag in flags:
            sev = flag['severity']
            if sev not in severities:
                severities[sev] = []
            severities[sev].append(flag)
        
        for severity in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW']:
            if severity in severities:
                print(f"\n  {severity} ({len(severities[severity])}):")
                for flag in severities[severity]:
                    print(f"    - {flag['chemical']} ({flag['category']})")


async def main():
    """Run all tests"""
    
    print("\n" + "🧪 " * 40)
    print("HIPPIEKIT PRODUCT IDENTIFICATION - TEST SUITE")
    print("🧪 " * 40)
    
    # Test 1: Chemical Checker (no API needed)
    await test_chemical_checker()
    
    # Test 2: Web Search (requires SERPAPI_KEY and OPENAI_API_KEY)
    try:
        await test_web_search()
    except Exception as e:
        print(f"\n❌ Web search test failed: {e}")
        print("Make sure SERPAPI_KEY and OPENAI_API_KEY are set in .env")
    
    # Test 3: Product Identification (requires image)
    await test_product_identification()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Set up API keys in .env file")
    print("2. Test with: python test_product_identification.py")
    print("3. Test with real image: python -c 'import test_product_identification; test_product_identification.test_with_image(\"your_image.jpg\")'")
    print("4. Test API endpoint: curl -X POST http://localhost:8001/identify/product -F image=@product.jpg")
    print()


if __name__ == "__main__":
    asyncio.run(main())
