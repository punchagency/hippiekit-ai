"""
Check actual WordPress API response structure
"""

import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

WORDPRESS_URL = os.getenv('WORDPRESS_API_URL', 'https://dodgerblue-otter-660921.hostingersite.com/wp-json/wp/v2/products/')

print("\n🔍 Checking WordPress API Response...\n")

try:
    response = requests.get(WORDPRESS_URL, params={'page': 1, 'per_page': 5}, timeout=15)
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}\n")
    
    data = response.json()
    print(f"Response Type: {type(data)}")
    print(f"Number of products: {len(data) if isinstance(data, list) else 'Not a list'}\n")
    
    if isinstance(data, list) and len(data) > 0:
        print("✅ First product structure:")
        print(json.dumps(data[0], indent=2)[:1000])  # First 1000 chars
        print("\n...")
    else:
        print("Full response:")
        print(json.dumps(data, indent=2)[:2000])
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
