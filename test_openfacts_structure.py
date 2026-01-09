"""
Direct test of OpenFacts API to verify data structure
This helps identify which fields are available in the response
"""
import requests
import json

# Test with a known barcode
TEST_BARCODE = "5000159461122"  # Cadbury Dairy Milk

def test_openfacts_api():
    print("=" * 80)
    print(f"OPENFACTS API STRUCTURE TEST - Barcode: {TEST_BARCODE}")
    print("=" * 80)
    
    url = f"https://world.openfoodfacts.org/api/v2/product/{TEST_BARCODE}.json"
    
    print(f"\n📡 Fetching from: {url}")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch: {response.status_code}")
        return
    
    data = response.json()
    
    if data.get('status') != 1:
        print(f"❌ Product not found")
        return
    
    product = data.get('product', {})
    
    print(f"\n✅ Product found!")
    print("\n" + "=" * 80)
    print("KEY FIELDS FOR PACKAGING ANALYSIS")
    print("=" * 80)
    
    # Brand info
    print(f"\n🏷️  BRAND INFORMATION:")
    print(f"  product['brands'] = {repr(product.get('brands'))}")
    print(f"  Type: {type(product.get('brands'))}")
    
    # Product name
    print(f"\n📝 PRODUCT NAME:")
    print(f"  product['product_name'] = {repr(product.get('product_name'))}")
    print(f"  product['name'] = {repr(product.get('name'))}")
    
    # Categories
    print(f"\n📂 CATEGORIES:")
    print(f"  product['categories'] = {repr(product.get('categories'))}")
    print(f"  product['categories_tags'] = {product.get('categories_tags', [])[:3]}... (showing first 3)")
    
    # Packaging info
    print(f"\n📦 PACKAGING:")
    print(f"  product['packaging'] = {repr(product.get('packaging'))}")
    print(f"  product['packaging_tags'] = {product.get('packaging_tags')}")
    print(f"  product['packaging_text'] = {repr(product.get('packaging_text'))}")
    
    # Detailed packagings array
    print(f"\n🔍 PACKAGINGS ARRAY (detailed):")
    packagings = product.get('packagings', [])
    print(f"  product['packagings'] = {len(packagings)} items")
    
    if packagings:
        for i, pkg in enumerate(packagings):
            print(f"\n  [{i}] Packaging item:")
            print(f"      material: {pkg.get('material')}")
            print(f"      shape: {pkg.get('shape')}")
            print(f"      food_contact: {pkg.get('food_contact')}")
            print(f"      recycling: {pkg.get('recycling')}")
            print(f"      number_of_units: {pkg.get('number_of_units')}")
    
    # Save full response for inspection
    output_file = "openfacts_response.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(product, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Full response saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    issues = []
    
    if not product.get('brands'):
        issues.append("⚠️  No 'brands' field")
    
    if not product.get('product_name') and not product.get('name'):
        issues.append("⚠️  No 'product_name' or 'name' field")
    
    if not product.get('categories'):
        issues.append("⚠️  No 'categories' field")
    
    if not packagings:
        issues.append("⚠️  No 'packagings' array")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All expected fields present!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        test_openfacts_api()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
