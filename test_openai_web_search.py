"""
Test OpenAI Native Web Search for Product Ingredients

This test demonstrates the new OpenAI web search integration
using the Responses API with web_search tool.

Usage:
    python test_openai_web_search.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
# Import the updated service
from services.web_search_service import web_search_service


async def test_search():
    """Test the OpenAI web search for various product types"""
    
    test_products = [
        {
            "brand": "Dove",
            "product_name": "Beauty Bar Soap",
            "category": "soap"
        },
        {
            "brand": "Aveeno",
            "product_name": "Daily Moisturizing Lotion",
            "category": "skincare"
        },
        {
            "brand": "Purina",
            "product_name": "Dog Chow",
            "category": "pet food"
        }
    ]
    
    print("=" * 80)
    print("TESTING OPENAI NATIVE WEB SEARCH")
    print("=" * 80)
    print()
    
    for product in test_products:
        print(f"\n{'=' * 80}")
        print(f"Product: {product['brand']} {product['product_name']}")
        print(f"Category: {product['category']}")
        print(f"{'=' * 80}\n")
        
        result = await web_search_service.search_product_ingredients(
            product_name=product['product_name'],
            brand=product['brand'],
            category=product['category']
        )
        
        if result:
            print(f"✓ SUCCESS")
            print(f"Source: {result.get('source', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Citations: {len(result.get('citations', []))}")
            print(f"Sources: {len(result.get('sources', []))}")
            print(f"\nIngredients:")
            print(result.get('ingredients', 'N/A')[:500] + "..." if len(result.get('ingredients', '')) > 500 else result.get('ingredients', 'N/A'))
            print(f"\nNote: {result.get('note', 'N/A')}")
            
            if result.get('citations'):
                print(f"\nCited Sources:")
                for i, citation in enumerate(result['citations'][:3], 1):
                    print(f"  {i}. {citation.get('title', 'N/A')}")
                    print(f"     {citation.get('url', 'N/A')}")
        else:
            print(f"✗ No ingredients found")
        
        print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file")
        exit(1)
    
    # Run test
    asyncio.run(test_search())
