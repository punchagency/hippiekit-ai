"""
Quick test for OpenAI Vision API connectivity
"""
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

print("=" * 60)
print("OpenAI Vision API Test")
print("=" * 60)

# Check OpenAI package version
try:
    import openai
    print(f"OpenAI package version: {openai.__version__}")
except Exception as e:
    print(f"❌ Error importing openai: {e}")
    sys.exit(1)

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in environment")
    sys.exit(1)

print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")

# Validate key format
if not api_key.startswith("sk-"):
    print(f"⚠️  WARNING: API key doesn't start with 'sk-' (found: {api_key[:5]}...)")
else:
    print("✓ API Key format looks correct")

# Test OpenAI client initialization
try:
    print("\nInitializing OpenAI client...")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# Test a simple API call (without image)
try:
    print("\nTesting OpenAI API with simple text completion...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Say 'API test successful' if you can read this."}
        ],
        max_tokens=50,
        temperature=0.1
    )
    
    result = response.choices[0].message.content
    print(f"✓ API Response: {result}")
    print("\n" + "=" * 60)
    print("✓ OpenAI API is working correctly!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ API call failed: {type(e).__name__}: {str(e)}")
    print("\nPossible issues:")
    print("1. Invalid API key")
    print("2. No credits/quota remaining")
    print("3. Network connectivity issues")
    print("4. OpenAI API is down")
    sys.exit(1)
