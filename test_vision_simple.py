"""
Simple test for Vision API with minimal image
"""
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()

# Create a minimal 1x1 pixel PNG (valid but tiny)
# This is a base64-encoded 1x1 red pixel PNG
TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ No API key found")
    exit(1)

print(f"Testing with API key: {api_key[:10]}...{api_key[-4:]}")

client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)

print("\nSending minimal test image to Vision API...")
start = time.time()

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Just give a brief response."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{TINY_PNG}",
                            "detail": "low"  # Use low detail for faster response
                        }
                    }
                ]
            }
        ],
        max_tokens=100,
        temperature=0.2
    )
    
    elapsed = time.time() - start
    result = response.choices[0].message.content
    
    print(f"\n✓ Success in {elapsed:.2f}s")
    print(f"Response: {result}")
    
except Exception as e:
    elapsed = time.time() - start
    print(f"\n❌ Failed after {elapsed:.2f}s")
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
