# Vision API Debugging Guide

## Problem

Vision API calls are timing out after ~5 minutes with SSL protocol error:

```
javax.net.ssl.SSLProtocolException: Read error: ssl=0xb400007a3dd21668: Failure in SSL library, usually a protocol error
error:1e000065:Cipher functions:OPENSSL_internal:BAD_DECRYPT
error:1000008b:SSL routines:OPENSSL_internal:DECRYPTION_FAILED_OR_BAD_RECORD_MAC
```

## Recent Improvements Made

✅ Added 60-second timeout to OpenAI client
✅ Added comprehensive logging throughout vision_service.py
✅ Added detailed logging to /scan-vision endpoint
✅ Added API key validation
✅ Added retry logic (max 2 retries)

## Diagnostic Steps

### Step 1: Test OpenAI API Connectivity

```powershell
cd ai-service
python test_openai.py
```

This will verify:

- API key is present
- API key format is correct
- OpenAI API is accessible
- Basic text completion works

### Step 2: Test Vision API with Minimal Image

```powershell
python test_vision_simple.py
```

This tests Vision API with a tiny 1x1 pixel image to verify:

- Vision endpoint is working
- Image processing works
- Response time is reasonable

### Step 3: Restart Backend with New Logging

```powershell
# Stop current backend (Ctrl+C if running)
python main.py
```

Look for these startup logs:

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:services.vision_service:VisionService initialized with model gpt-4o
INFO:     Application startup complete.
```

### Step 4: Test from App and Check Logs

When you test Vision scan from the app, you should see logs like:

```
INFO:routers.scan:Received vision scan request: filename=photo.jpg, content_type=image/jpeg
INFO:routers.scan:Read 2458392 bytes from image
INFO:routers.scan:Retrieved VisionService
INFO:routers.scan:Starting vision analysis...
INFO:services.vision_service:Starting vision analysis for image (2458392 bytes)
INFO:services.vision_service:Encoding image to base64...
INFO:services.vision_service:Image encoded (3277856 chars)
INFO:services.vision_service:Calling OpenAI Vision API with model gpt-4o...
INFO:services.vision_service:OpenAI API call completed in 4.52s
INFO:services.vision_service:Response received (1234 chars)
INFO:services.vision_service:Vision analysis completed successfully
```

## Common Issues and Solutions

### Issue 1: No API Key Found

**Symptom**: `ValueError: OPENAI_API_KEY environment variable is required`
**Solution**:

```powershell
# Create .env file in ai-service/ with:
OPENAI_API_KEY=sk-your-key-here
```

### Issue 2: Invalid API Key

**Symptom**: `ValueError: OPENAI_API_KEY appears to be invalid`
**Solution**: Verify your API key starts with `sk-` and is from https://platform.openai.com/api-keys

### Issue 3: API Key No Credits

**Symptom**: `openai.error.RateLimitError` or `insufficient_quota`
**Solution**: Add credits to your OpenAI account at https://platform.openai.com/settings/organization/billing

### Issue 4: Timeout After 60 Seconds

**Symptom**: Logs show "OpenAI API call completed in 60.00s" then error
**Solution**: This is expected timeout behavior. The image might be too large. Try:

1. Reduce image quality in Capacitor camera settings
2. Change `detail: "high"` to `detail: "low"` in vision_service.py line 73
3. Add image compression before sending to API

### Issue 5: Ngrok Tunnel Timeout

**Symptom**: Connection drops after several minutes
**Solution**:

1. Restart ngrok: `ngrok http 8001`
2. Update frontend AI_SERVICE_URL with new ngrok URL
3. Consider deploying backend to a cloud service (Railway, Render, Fly.io)

### Issue 6: Large Image Taking Too Long

**Symptom**: Logs show encoding takes >10 seconds or API call >30 seconds
**Solution**: Add image compression in vision_service.py:

```python
from PIL import Image
import io

# Before base64 encoding:
img = Image.open(io.BytesIO(image_bytes))
# Resize to max 1024x1024
img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
# Convert back to bytes
buffer = io.BytesIO()
img.save(buffer, format='JPEG', quality=85)
image_bytes = buffer.getvalue()
```

## Performance Expectations

**Normal timings**:

- Image encoding: 0.1-1.0 seconds
- OpenAI Vision API call: 2-10 seconds
- Total request: 3-12 seconds

**Slow timings (needs optimization)**:

- Image encoding: >2 seconds → Image too large, add compression
- API call: >20 seconds → Use "low" detail or smaller image
- Total: >30 seconds → Both above issues

## Next Steps

1. Run test_openai.py to verify API works
2. Run test_vision_simple.py to verify Vision works with small image
3. If tests pass, the issue is likely:

   - Image size too large from phone camera
   - Ngrok connection unstable
   - Frontend not handling timeouts properly

4. If tests fail, the issue is:
   - Invalid API key
   - No OpenAI credits
   - Network/firewall blocking OpenAI
   - OpenAI API is down

## Frontend Improvements Needed

Current issues in frontend code:

1. No timeout on fetch() call
2. No progress indicator during long operations
3. No retry logic
4. No image compression before upload

Recommended fixes in scanService.ts:

```typescript
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

try {
  const res = await fetch(`${AI_SERVICE_URL}/scan-vision`, {
    method: 'POST',
    body: formData,
    signal: controller.signal,
  });
  clearTimeout(timeoutId);
  // ... rest of code
} catch (error) {
  clearTimeout(timeoutId);
  if (error.name === 'AbortError') {
    throw new Error(
      'Vision analysis timed out. Please try with a clearer photo.'
    );
  }
  throw error;
}
```
