"""
Test script for OpenAI web search ingredient lookup
Tests the web_search_service to verify it can find ingredients for products
"""

import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from services.web_search_service import web_search_service

# Setup logging to see detailed output
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see response structure
    format='%(levelname)s:%(name)s:%(message)s'
)

async def test_product(brand: str, product_name: str, category: str = None):
    """Test web search for a single product"""
    print(f"\n{'='*80}")
    print(f"Testing: {brand} {product_name}")
    if category:
        print(f"Category: {category}")
    print('='*80)
    
    result = await web_search_service.search_product_ingredients(
        product_name=product_name,
        brand=brand,
        category=category
    )
    
    if result:
        print("\n✅ SUCCESS - Ingredients found!")
        print(f"\nIngredients:\n{result['ingredients']}")
        print(f"\nSource: {result['source']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources found: {len(result['sources'])}")
        
        if result['sources']:
            print("\nSource URLs:")
            for url in result['sources']:
                print(f"  - {url}")
        
        print(f"\nNote: {result['note']}")
    else:
        print("\n❌ FAILED - No ingredients found")
    
    print('='*80)
    return result

async def main():
    print("\n🧪 Testing OpenAI Web Search for Product Ingredients\n")
    
    # Test cases
    test_cases = [
        {
            "brand": "CeraVe",
            "product_name": "Acne Control Cleanser",
            "category": "Skincare"
        },
        {
            "brand": "Dove",
            "product_name": "Deep Moisture Body Wash",
            "category": "Personal Care"
        },
        {
            "brand": "Heinz",
            "product_name": "Tomato Ketchup",
            "category": "Food"
        }
    ]
    
    results = []
    for test in test_cases:
        result = await test_product(**test)
        results.append({
            "product": f"{test['brand']} {test['product_name']}",
            "success": result is not None
        })
        
        # Small delay between tests to avoid rate limits
        await asyncio.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['product']}")
    
    print(f"\nSuccess rate: {success_count}/{total_count} ({success_count/total_count*100:.0f}%)")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
