"""
Test script to verify product context extraction for packaging analysis
Run this to check if product data is being extracted correctly
"""
import requests
import json

# Test with a known barcode (Cadbury Dairy Milk)
TEST_BARCODE = "5000159461122"  # Change to your test barcode

BASE_URL = "http://localhost:8000"  # Adjust if your server runs on different port

def test_packaging_context():
    print("=" * 60)
    print("PACKAGING CONTEXT EXTRACTION TEST")
    print("=" * 60)
    
    # Step 1: Get product data from barcode lookup
    print(f"\n1. Looking up barcode: {TEST_BARCODE}")
    print("-" * 60)
    
    lookup_response = requests.post(
        f"{BASE_URL}/lookup-barcode",
        json={"barcode": TEST_BARCODE}
    )
    
    if lookup_response.status_code != 200:
        print(f"❌ Lookup failed: {lookup_response.status_code}")
        print(lookup_response.text)
        return
    
    product_data = lookup_response.json()
    print(f"✅ Product found: {product_data.get('name', 'Unknown')}")
    
    # Extract key fields
    print("\n📦 Product Data Fields:")
    print(f"  - brands: {product_data.get('brands', 'NOT FOUND')}")
    print(f"  - product_name: {product_data.get('product_name', 'NOT FOUND')}")
    print(f"  - name: {product_data.get('name', 'NOT FOUND')}")
    print(f"  - categories: {product_data.get('categories', 'NOT FOUND')}")
    print(f"  - categories_tags: {product_data.get('categories_tags', 'NOT FOUND')}")
    print(f"  - packaging: {product_data.get('packaging', 'NOT FOUND')}")
    print(f"  - packaging_tags: {product_data.get('packaging_tags', 'NOT FOUND')}")
    print(f"  - packagings: {product_data.get('packagings', 'NOT FOUND')}")
    
    # Step 2: Call /separate to extract and normalize materials
    print(f"\n2. Testing /barcode/packaging/separate")
    print("-" * 60)
    
    separate_response = requests.post(
        f"{BASE_URL}/barcode/packaging/separate",
        json={
            "barcode": TEST_BARCODE,
            "product_data": product_data  # Pass the product data
        }
    )
    
    if separate_response.status_code != 200:
        print(f"❌ Separate failed: {separate_response.status_code}")
        print(separate_response.text)
        return
    
    separate_data = separate_response.json()
    print(f"✅ Materials extracted: {separate_data.get('materials', [])}")
    
    # Check product context
    product_context = separate_data.get('product_context', {})
    print("\n🔍 Product Context Returned:")
    print(f"  - brand_name: '{product_context.get('brand_name', 'MISSING')}'")
    print(f"  - product_name: '{product_context.get('product_name', 'MISSING')}'")
    print(f"  - categories: '{product_context.get('categories', 'MISSING')}'")
    print(f"  - food_contact: {product_context.get('food_contact', 'MISSING')}")
    print(f"  - packagings_data: {len(product_context.get('packagings_data', []))} items")
    
    # Step 3: Call /describe with context
    print(f"\n3. Testing /barcode/packaging/describe WITH context")
    print("-" * 60)
    
    describe_request = {
        "barcode": TEST_BARCODE,
        "packaging_text": separate_data.get('packaging_text', ''),
        "packaging_tags": separate_data.get('packaging_tags', []),
        "normalized_materials": separate_data.get('materials', []),
        "brand_name": product_context.get('brand_name'),
        "product_name": product_context.get('product_name'),
        "categories": product_context.get('categories'),
        "food_contact": product_context.get('food_contact'),
    }
    
    print("\n📤 Request payload:")
    print(json.dumps(describe_request, indent=2))
    
    describe_response = requests.post(
        f"{BASE_URL}/barcode/packaging/describe",
        json=describe_request
    )
    
    if describe_response.status_code != 200:
        print(f"❌ Describe failed: {describe_response.status_code}")
        print(describe_response.text)
        return
    
    describe_data = describe_response.json()
    print(f"\n✅ Analysis complete!")
    
    # Show first material analysis
    analysis = describe_data.get('analysis', {}).get('analysis', {})
    if analysis:
        first_material = list(analysis.keys())[0]
        print(f"\n📊 Sample Analysis for '{first_material}':")
        print(f"  Description (first 200 chars):")
        print(f"    {analysis[first_material].get('description', '')[:200]}...")
        print(f"  Harmful: {analysis[first_material].get('harmful')}")
        print(f"  Severity: {analysis[first_material].get('severity')}")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE - Check server logs for detailed prompt")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_packaging_context()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
