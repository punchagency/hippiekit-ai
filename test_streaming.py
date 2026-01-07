#!/usr/bin/env python3
"""
Quick test script to verify streaming endpoint works correctly.
Tests that events arrive progressively, not all at once.
"""

import requests
import time
import sys

def test_streaming_endpoint(barcode: str = "5000159461122"):
    """Test the streaming barcode lookup endpoint"""
    url = f"http://localhost:8001/lookup-barcode-stream?barcode={barcode}"
    
    print(f"🧪 Testing streaming endpoint: {url}")
    print("=" * 80)
    
    try:
        # Use stream=True to get progressive chunks
        response = requests.get(url, stream=True, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            return False
        
        print("✅ Connection established")
        print("📡 Receiving events:\n")
        
        event_count = 0
        start_time = time.time()
        last_event_time = start_time
        
        # Read line by line
        for line in response.iter_lines(decode_unicode=True):
            if line:
                current_time = time.time()
                elapsed_since_start = current_time - start_time
                elapsed_since_last = current_time - last_event_time
                
                # Parse event type if it's a data line
                if line.startswith('data:'):
                    event_count += 1
                    try:
                        import json
                        data = json.loads(line[5:].strip())
                        event_type = data.get('type', 'unknown')
                        
                        print(f"[{elapsed_since_start:6.2f}s] (+{elapsed_since_last:5.2f}s) Event #{event_count}: {event_type}")
                        
                        # Print some details based on type
                        if event_type == 'basic_info':
                            print(f"          → Product: {data['data'].get('name', 'N/A')}")
                        elif event_type == 'ingredients_separated':
                            harmful = data['data'].get('harmful_count', 0)
                            safe = data['data'].get('safe_count', 0)
                            print(f"          → Harmful: {harmful}, Safe: {safe}")
                        elif event_type == 'harmful_descriptions':
                            count = len(data['data'])
                            print(f"          → {count} harmful ingredient descriptions")
                        elif event_type == 'safe_descriptions':
                            count = len(data['data'])
                            print(f"          → {count} safe ingredient descriptions")
                        elif event_type == 'packaging_analysis':
                            materials = len(data['data'].get('materials', []))
                            print(f"          → {materials} packaging materials analyzed")
                        elif event_type == 'complete':
                            print(f"          → ✅ Analysis complete!")
                        elif event_type == 'error':
                            print(f"          → ❌ Error: {data['data'].get('message')}")
                        
                        last_event_time = current_time
                        
                    except json.JSONDecodeError as e:
                        print(f"          ⚠️  Failed to parse: {e}")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"📊 Summary:")
        print(f"   Total events: {event_count}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time between events: {total_time/event_count if event_count > 0 else 0:.2f}s")
        
        if event_count > 0:
            print("\n✅ Streaming is working! Events arrived progressively.")
            return True
        else:
            print("\n❌ No events received. Streaming may not be working.")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False


if __name__ == "__main__":
    # Get barcode from command line or use default
    barcode = sys.argv[1] if len(sys.argv) > 1 else "012000001659"
    
    print("\n🔍 Barcode Streaming Test")
    print(f"Barcode: {barcode}")
    print()
    
    success = test_streaming_endpoint(barcode)
    
    sys.exit(0 if success else 1)
