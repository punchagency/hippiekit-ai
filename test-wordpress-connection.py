"""
Quick test script to check WordPress connectivity.
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

WORDPRESS_URL = os.getenv('WORDPRESS_API_URL', 'https://dodgerblue-otter-660921.hostingersite.com/wp-json/wp/v2/products/')
BASE_URL = WORDPRESS_URL.split('/wp-json')[0]

print("\n🔍 Testing WordPress Connectivity...")
print(f"Base URL: {BASE_URL}")
print(f"API URL: {WORDPRESS_URL}\n")

# Test 1: Base site
print("Test 1: Accessing base WordPress site...")
try:
    response = requests.get(BASE_URL, timeout=10)
    print(f"✅ Base site accessible (Status: {response.status_code})\n")
except Exception as e:
    print(f"❌ Base site inaccessible: {e}\n")

# Test 2: WP REST API endpoint
print("Test 2: Accessing WordPress REST API...")
try:
    response = requests.get(WORDPRESS_URL, params={'page': 1, 'per_page': 1}, timeout=10)
    response.raise_for_status()
    data = response.json()
    print(f"✅ API accessible (Status: {response.status_code})")
    print(f"   Sample product: {data[0].get('name') if data else 'No products'}\n")
except Exception as e:
    print(f"❌ API inaccessible: {e}\n")

# Test 3: DNS resolution
print("Test 3: DNS resolution...")
import socket
hostname = 'dodgerblue-otter-660921.hostingersite.com'
try:
    ip = socket.gethostbyname(hostname)
    print(f"✅ DNS resolves: {hostname} → {ip}\n")
except Exception as e:
    print(f"❌ DNS failed: {e}\n")
    print("💡 This is likely a DNS or network issue.")
    print("   Try:")
    print("   1. Check your internet connection")
    print("   2. Try accessing the site in a web browser")
    print("   3. Flush DNS cache: ipconfig /flushdns")
    print("   4. Use a different DNS (8.8.8.8, 1.1.1.1)")

print("Done!")
