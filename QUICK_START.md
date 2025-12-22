# Vision API - Quick Start Checklist

## ✅ Pre-Flight Checks

### 1. Verify OpenAI API Key

```powershell
cd ai-service
python test_openai.py
```

Expected output:

```
✓ API Key found: sk-proj-xx...xxxx
✓ API Key format looks correct
✓ OpenAI client initialized
✓ API Response: API test successful
✓ OpenAI API is working correctly!
```

### 2. Test Vision API

```powershell
python test_vision_simple.py
```

Expected output:

```
Testing with API key: sk-proj-xx...xxxx
Sending minimal test image to Vision API...
✓ Success in 3.45s
Response: [Description of the tiny test image]
```

### 3. Start Backend

```powershell
python main.py
```

Expected startup logs:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:services.vision_service:VisionService initialized with model gpt-4o
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### 4. Start Ngrok (if testing on phone)

```powershell
ngrok http 8001
```

Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)

### 5. Update Frontend (if ngrok URL changed)

Edit `frontend/src/constants/index.ts`:

```typescript
export const AI_SERVICE_URL = 'https://your-new-ngrok-url.ngrok.io';
```

### 6. Rebuild Frontend

```powershell
cd ../frontend
npm run build
npx cap sync
```

### 7. Deploy to Android

```powershell
npx cap open android
```

Then click Run in Android Studio

## 🧪 Testing Flow

1. **Open app on phone**
2. **Navigate to Scan page**
3. **Tap "AI OCR (Vision)" button**
4. **Take clear photo of product label**
   - Get close to the label
   - Ensure good lighting
   - Label should be readable in photo
5. **Wait for processing** (5-15 seconds expected)
6. **Check results** or error message

## 📊 Monitor Backend Logs

Successful Vision scan logs:

```
INFO:routers.scan:Received vision scan request: filename=photo.jpg, content_type=image/jpeg
INFO:routers.scan:Read 2458392 bytes from image
INFO:services.vision_service:Starting vision analysis for image (2458392 bytes)
INFO:services.vision_service:Compressing image if needed...
INFO:services.vision_service:Resized image from (3024, 4032) to (1152, 1536)
INFO:services.vision_service:Compressed image: 2458392 → 456123 bytes (18.5%)
INFO:services.vision_service:Encoding image to base64...
INFO:services.vision_service:Image encoded (608164 chars)
INFO:services.vision_service:Calling OpenAI Vision API with model gpt-4o...
INFO:services.vision_service:OpenAI API call completed in 4.52s
INFO:services.vision_service:Response received (1234 chars)
INFO:services.vision_service:Vision analysis completed successfully
```

## ❌ Common Errors & Fixes

### Error: "OPENAI_API_KEY environment variable is required"

**Fix**: Create `.env` file in `ai-service/` with:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Error: "OPENAI_API_KEY appears to be invalid"

**Fix**: Verify your API key:

1. Go to https://platform.openai.com/api-keys
2. Generate a new key if needed
3. Key should start with `sk-`

### Error: "insufficient_quota" or "RateLimitError"

**Fix**: Add credits to OpenAI account:

1. Go to https://platform.openai.com/settings/organization/billing
2. Add payment method and credits

### Error: "Vision analysis timed out"

**Fix**: Check backend logs to see where delay occurs:

- If encoding >2s: Image too large (should be fixed now)
- If API call >20s: OpenAI might be slow, try again
- If total >45s: Frontend timeout, backend might be stuck

### Error: "Failed to run OCR analysis" (SSL error)

**Fix**: This should be resolved now with:

- ✅ Image compression (faster processing)
- ✅ 60s timeout on backend
- ✅ 45s timeout on frontend
- If still occurs, restart ngrok and backend

## 📈 Performance Targets

| Metric            | Target   | Acceptable | Action if Slower          |
| ----------------- | -------- | ---------- | ------------------------- |
| Image compression | <1s      | <2s        | Reduce max_size           |
| Base64 encoding   | <1s      | <2s        | Image already compressed  |
| OpenAI API call   | <10s     | <20s       | Use "low" detail          |
| **Total request** | **<15s** | **<30s**   | Check logs for bottleneck |

## 🔍 Debugging Steps

If Vision scan fails:

1. **Check backend is running**

   ```powershell
   curl http://localhost:8001/
   ```

   Should return: `{"message":"AI Service is running"}`

2. **Check ngrok tunnel**

   ```powershell
   curl https://your-ngrok-url.ngrok.io/
   ```

   Should return same message

3. **Check API key works**

   ```powershell
   python test_openai.py
   ```

4. **Check Vision API works**

   ```powershell
   python test_vision_simple.py
   ```

5. **Check backend logs**
   Look for ERROR or WARNING messages

6. **Check frontend logs**
   In Android Studio: View → Tool Windows → Logcat
   Filter by "CapacitorHttp" or "Scan"

## 📝 Quick Reference

### Backend Dependencies

- OpenAI SDK 1.54.0 ✅
- Pillow 10.1.0 ✅ (for image compression)
- FastAPI ✅
- All set! Just need API key

### Frontend Changes

- Added timeout handler ✅
- Added abort controller ✅
- Better error messages ✅

### Image Optimization

- Max size: 1536x1536 px
- Quality: 85%
- Format: JPEG
- Typical reduction: 60-85% smaller

## 🚀 Next Actions

1. ✅ Run `test_openai.py` - Verify API access
2. ✅ Run `test_vision_simple.py` - Verify Vision works
3. ✅ Start backend with `python main.py`
4. ✅ Start ngrok (if needed)
5. ✅ Rebuild frontend with `npm run build`
6. ✅ Test on device
7. 📊 Monitor logs during test
8. 🎉 Enjoy faster, more reliable Vision OCR!

## 📚 Documentation

- [VISION_DEBUGGING.md](./VISION_DEBUGGING.md) - Comprehensive debugging guide
- [VISION_TIMEOUT_FIX.md](./VISION_TIMEOUT_FIX.md) - Detailed changes summary
- [README.md](./README.md) - Project overview

---

**Need help?** Check the logs first! Most issues show up in the backend terminal output.
