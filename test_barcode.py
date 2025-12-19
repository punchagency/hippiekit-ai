"""
Quick test for barcode lookup functionality
"""
from services.barcode_service import get_barcode_service

# Test with a known Coca-Cola barcode
barcode = "3017620422003"

service = get_barcode_service()
result = service.lookup_barcode(barcode)

print("=" * 50)
print(f"Testing barcode lookup for: {barcode}")
print("=" * 50)

if result:
    print(f"✓ Product found!")
    print(f"Name: {result.get('name', 'N/A')}")
    print(f"Brand: {result.get('brand', 'N/A')}")
    print(f"Source: {result.get('source', 'N/A')}")
    print(f"Categories: {result.get('categories', 'N/A')}")
    print(f"Has ingredients: {len(result.get('ingredients', [])) > 0}")
    print(f"Has nutrition: {len(result.get('nutrition', {})) > 0}")
else:
    print("✗ Product not found")

print("=" * 50)
