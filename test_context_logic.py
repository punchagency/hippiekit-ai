"""
Quick test to verify context extraction works correctly
"""

# Simulate what happens in the code
product_data = {
    "brands": "Snickers",
    "product_name": "Snickers",
    "categories": "Candy chocolate bars",
    "packagings": [
        {
            "food_contact": 1,
            "material": "en:plastic"
        }
    ]
}

# Test extraction logic
brand_name = product_data.get("brands", "").strip() or None
categories = product_data.get("categories", "").strip() or None
product_name = (product_data.get("product_name", "").strip() or 
               product_data.get("name", "").strip() or None)

# Extract food contact
food_contact = None
packagings_data = product_data.get("packagings", [])
if packagings_data and isinstance(packagings_data, list):
    for pkg in packagings_data:
        if pkg.get("food_contact") == 1:
            food_contact = True
            break

print("Extracted values:")
print(f"  brand_name: {repr(brand_name)}")
print(f"  product_name: {repr(product_name)}")
print(f"  categories: {repr(categories)}")
print(f"  food_contact: {food_contact}")

# Build context (same logic as in analyze_packaging_material)
context_parts = []
if brand_name:
    context_parts.append(f"Brand: {brand_name}")
if product_name:
    context_parts.append(f"Product: {product_name}")
if categories:
    main_category = categories.split(",")[0].strip() if "," in categories else categories
    context_parts.append(f"Category: {main_category}")
if food_contact is not None:
    contact_status = "YES - direct food contact" if food_contact else "NO - no direct food contact"
    context_parts.append(f"Food Contact: {contact_status}")

product_context = "\n".join(context_parts) if context_parts else "Product context not available"

print("\nFinal product context:")
print(product_context)

# Test with empty strings
print("\n" + "="*60)
print("Testing with EMPTY strings (simulating missing data):")
print("="*60)

product_data_empty = {
    "brands": "",  # Empty string
    "product_name": "",  # Empty string
    "categories": "",  # Empty string
    "packagings": []
}

brand_name = product_data_empty.get("brands", "").strip() or None
categories = product_data_empty.get("categories", "").strip() or None
product_name = (product_data_empty.get("product_name", "").strip() or 
               product_data_empty.get("name", "").strip() or None)

food_contact = None
packagings_data = product_data_empty.get("packagings", [])
if packagings_data and isinstance(packagings_data, list):
    for pkg in packagings_data:
        if pkg.get("food_contact") == 1:
            food_contact = True
            break

print("Extracted values:")
print(f"  brand_name: {repr(brand_name)}")
print(f"  product_name: {repr(product_name)}")
print(f"  categories: {repr(categories)}")
print(f"  food_contact: {food_contact}")

context_parts = []
if brand_name:
    context_parts.append(f"Brand: {brand_name}")
if product_name:
    context_parts.append(f"Product: {product_name}")
if categories:
    main_category = categories.split(",")[0].strip() if "," in categories else categories
    context_parts.append(f"Category: {main_category}")
if food_contact is not None:
    contact_status = "YES - direct food contact" if food_contact else "NO - no direct food contact"
    context_parts.append(f"Food Contact: {contact_status}")

product_context = "\n".join(context_parts) if context_parts else "Product context not available"

print("\nFinal product context:")
print(product_context)
print("\n✅ Correctly shows 'Product context not available' for empty data")
