#!/usr/bin/env python3
"""
Test script for the new modular barcode API routes.
Tests each route independently.
"""

import requests
import time
import json
import sys

BASE_URL = "http://localhost:8001"
TEST_BARCODE = "5000159461122"  # Skittles


def test_lookup():
    """Test FAST route: /barcode/lookup"""
    print("\n" + "=" * 80)
    print("🔍 TEST 1: GET /barcode/lookup (FAST - should return in 1-2s)")
    print("=" * 80)
    
    start = time.time()
    
    try:
        response = requests.get(
            f"{BASE_URL}/barcode/lookup",
            params={"barcode": TEST_BARCODE},
            timeout=10
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ SUCCESS in {elapsed:.2f}s")
            print(f"   Found: {data.get('found')}")
            
            if data.get('found') and data.get('product'):
                product = data['product']
                print(f"   Product: {product.get('name')}")
                print(f"   Brand: {product.get('brands')}")
                print(f"   Has ingredients: {product.get('has_ingredients')}")
                print(f"   Ingredients count: {len(product.get('ingredients_list', []))}")
                
                return data['product']  # Return for next tests
            else:
                print(f"   ⚠️  Product not found")
                return None
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None


def test_ingredients(product_data=None):
    """Test MEDIUM route: /barcode/ingredients/analyze"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: POST /barcode/ingredients/analyze (MEDIUM - 5-10s)")
    print("=" * 80)
    
    start = time.time()
    
    payload = {
        "barcode": TEST_BARCODE,
        "product_data": product_data  # Optional: reuse data from test 1
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/barcode/ingredients/analyze",
            json=payload,
            timeout=30
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ SUCCESS in {elapsed:.2f}s")
            print(f"   Has ingredients: {data.get('has_ingredients')}")
            print(f"   Harmful count: {data.get('harmful_count', 0)}")
            print(f"   Safe count: {data.get('safe_count', 0)}")
            
            if data.get('harmful'):
                print(f"   Harmful examples: {', '.join(data['harmful'][:3])}")
            
            if data.get('safe'):
                print(f"   Safe examples: {', '.join(data['safe'][:3])}")
            
            descriptions = data.get('descriptions', {})
            harmful_desc_count = len(descriptions.get('harmful', {}))
            safe_desc_count = len(descriptions.get('safe', {}))
            
            print(f"   Harmful descriptions: {harmful_desc_count}")
            print(f"   Safe descriptions: {safe_desc_count}")
            
            return True
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_packaging(product_data=None):
    """Test MEDIUM route: /barcode/packaging/analyze"""
    print("\n" + "=" * 80)
    print("📦 TEST 3: POST /barcode/packaging/analyze (MEDIUM - 3-7s)")
    print("=" * 80)
    
    start = time.time()
    
    payload = {
        "barcode": TEST_BARCODE,
        "product_data": product_data
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/barcode/packaging/analyze",
            json=payload,
            timeout=30
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ SUCCESS in {elapsed:.2f}s")
            print(f"   Has packaging data: {data.get('has_packaging_data')}")
            
            if data.get('has_packaging_data'):
                analysis = data['analysis']
                materials = analysis.get('materials', [])
                
                print(f"   Materials count: {len(materials)}")
                if materials:
                    print(f"   Materials: {', '.join(materials)}")
                
                print(f"   Overall safety: {analysis.get('overall_safety')}")
                print(f"   Data source: {analysis.get('source')}")
                print(f"   Summary: {analysis.get('summary', '')[:100]}...")
            
            return True
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_parallel(product_data):
    """Test running ingredients + packaging in parallel"""
    print("\n" + "=" * 80)
    print("⚡ TEST 4: Parallel execution (ingredients + packaging together)")
    print("=" * 80)
    
    start = time.time()
    
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_ingredients = executor.submit(test_ingredients, product_data)
        future_packaging = executor.submit(test_packaging, product_data)
        
        # Wait for both to complete
        results = concurrent.futures.wait(
            [future_ingredients, future_packaging],
            return_when=concurrent.futures.ALL_COMPLETED
        )
    
    elapsed = time.time() - start
    
    print(f"\n⚡ Parallel execution completed in {elapsed:.2f}s")
    print(f"   (Sequential would take ~8-17s, parallel saves time!)")


if __name__ == "__main__":
    print("\n🧪 MODULAR BARCODE API TESTS")
    print(f"Testing barcode: {TEST_BARCODE}")
    print(f"Base URL: {BASE_URL}")
    
    # Test 1: Fast lookup
    product_data = test_lookup()
    
    if not product_data:
        print("\n❌ Cannot continue - product lookup failed")
        sys.exit(1)
    
    # Test 2: Ingredients analysis
    test_ingredients(product_data)
    
    # Test 3: Packaging analysis
    test_packaging(product_data)
    
    # Test 4: Parallel execution
    test_parallel(product_data)
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Update frontend to use new barcodeAnalysisService.ts")
    print("2. Add loading skeletons for each section")
    print("3. Test with real barcode scans in the app")
