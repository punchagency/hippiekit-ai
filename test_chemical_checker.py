"""Quick test of chemical checker"""
from services.chemical_checker import check_ingredients, calculate_safety_score, generate_recommendations

# Test ingredients
test_ingredients = """
Water, Sodium Lauryl Sulfate, Cocamidopropyl Betaine, Fragrance,
Methylparaben, Propylparaben, Red 40, Yellow 5, BPA
"""

print("=== TESTING CHEMICAL CHECKER ===\n")
print(f"Test Ingredients:\n{test_ingredients}\n")

# Check for chemicals
flags = check_ingredients(test_ingredients)
score = calculate_safety_score(flags)
recs = generate_recommendations(flags, "shampoo")

print(f"=== DETECTED {len(flags)} HARMFUL CHEMICALS ===")
for flag in flags:
    print(f"  {flag['severity'].upper():10} | {flag['chemical']:30} | {flag['category']}")

print(f"\n=== SAFETY SCORE: {score}/100 ===")

print("\n=== RECOMMENDATIONS ===")
print("\n❌ AVOID:")
for item in recs["avoid"][:3]:  # Show first 3
    print(f"  - {item}")

print("\n✅ LOOK FOR:")
for item in recs["look_for"][:3]:  # Show first 3
    print(f"  - {item}")

print("\n🏆 CERTIFICATIONS:")
for item in recs["certifications"][:3]:  # Show first 3
    print(f"  - {item}")

print("\n✅ Test complete!")
