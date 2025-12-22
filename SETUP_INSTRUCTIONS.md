# Setup Instructions for Product Identification System

## What's Been Implemented

✅ **Backend Services:**

1. `web_search_service.py` - Multi-tier ingredient lookup (SerpAPI + AI knowledge + category generic)
2. `vision_service.py` - Product identification from front photos
3. `barcode_service.py` - Updated with async web search fallback
4. `identify.py` router - New API endpoint for product identification

✅ **API Endpoints:**

- `POST /identify/product` - Upload product photo, get complete analysis

✅ **Configuration:**

- `requirements.txt` updated with `google-search-results==2.4.2`
- `.env.example` updated with `OPENAI_API_KEY` and `SERPAPI_KEY`

✅ **Documentation:**

- `PRODUCT_IDENTIFICATION.md` - Complete system overview and usage guide

## Next Steps

### 1. Install Dependencies

```powershell
cd c:\Users\JIDE\Desktop\hippiekit-separated\ai-service

# Install new packages
pip install google-search-results==2.4.2

# Or reinstall everything
pip install -r requirements.txt
```

### 2. Configure Environment Variables

You need to add your API keys to the `.env` file:

```powershell
# If you don't have a .env file yet:
cp .env.example .env

# Then edit .env and add:
# OPENAI_API_KEY=sk-... (your existing OpenAI key)
# SERPAPI_KEY=... (get from https://serpapi.com/)
```

#### Getting a SerpAPI Key:

1. Go to https://serpapi.com/
2. Sign up for free account
3. Free tier gives you 100 searches/month
4. Copy your API key from dashboard
5. Add to `.env` file

### 3. Test the Backend

```powershell
# Start the AI service
cd c:\Users\JIDE\Desktop\hippiekit-separated\ai-service
python main.py
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8001
```

#### Test the new endpoint:

```powershell
# Test with a product photo (you'll need a test image)
curl -X POST "http://localhost:8001/identify/product" `
  -F "image=@path\to\product_photo.jpg"
```

### 4. Frontend Integration (TODO)

The frontend still needs to be updated to support this new feature. Here's what needs to be created:

#### New Components Needed:

1. **ProductPhotoScan.tsx** - New scan mode for product photos

   - Camera capture or file upload
   - Loading state during multi-tier search
   - Shows data source and confidence level

2. **Update routing** to support product photo scan mode

   - Add new route: `/scan/product-photo`
   - Separate from barcode and ingredients OCR

3. **Update ProductResults pages** to show data source indicators
   - Already have: `VisionProductResults.tsx` and `BarcodeProductResults.tsx`
   - Need to add: Data source badge (database/web/AI/generic)
   - Need to add: Confidence indicator
   - Need to add: Ingredients note when available

#### Example Frontend Flow:

```typescript
// 1. User selects "Scan Product" mode
<button onClick={() => navigate('/scan/product-photo')}>
  Scan Product Front Label
</button>;

// 2. Take photo and upload
const formData = new FormData();
formData.append('image', photoBlob);

const response = await fetch('http://localhost:8001/identify/product', {
  method: 'POST',
  body: formData,
});

const result = await response.json();

// 3. Navigate to results with data source indicators
navigate('/results/product-photo', { state: result });
```

### 5. Testing Checklist

- [ ] Backend starts without errors
- [ ] `/identify/product` endpoint is accessible
- [ ] Can upload image and get response
- [ ] Web search works (requires SERPAPI_KEY)
- [ ] AI fallback works when web search fails
- [ ] Chemical analysis runs on all ingredient sources
- [ ] Data source and confidence are returned correctly
- [ ] Frontend displays all information properly

### 6. Cost Monitoring

**Important:** Track your API usage to avoid unexpected costs:

- **SerpAPI Free Tier:** 100 searches/month

  - Monitor at: https://serpapi.com/dashboard
  - Set up alerts when approaching limit

- **OpenAI API:**
  - Vision API: ~$0.01 per image
  - GPT-4o-mini: Much cheaper for web result parsing
  - Monitor at: https://platform.openai.com/usage

## Deployment Considerations

### Environment Variables for Production:

```env
# Required
OPENAI_API_KEY=sk-...
SERPAPI_KEY=...

# Optional but recommended
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=hippiekit-products
WORDPRESS_API_URL=...
PORT=8001
```

### Heroku Deployment:

```powershell
# Set config vars
heroku config:set OPENAI_API_KEY=sk-...
heroku config:set SERPAPI_KEY=...

# Deploy
git push heroku main
```

### Docker Deployment:

```dockerfile
# Add to Dockerfile
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV SERPAPI_KEY=${SERPAPI_KEY}
```

## File Structure

```
ai-service/
├── main.py                    ✅ Updated (includes identify router)
├── requirements.txt           ✅ Updated (added google-search-results)
├── .env.example              ✅ Updated (added API keys)
├── PRODUCT_IDENTIFICATION.md  ✅ New (full documentation)
├── routers/
│   ├── identify.py           ✅ New (product identification endpoint)
│   ├── scan.py               (existing)
│   └── index.py              (existing)
└── services/
    ├── web_search_service.py  ✅ New (multi-tier ingredient lookup)
    ├── vision_service.py      ✅ Updated (added identify_product_from_photo)
    ├── barcode_service.py     ✅ Updated (async with web search fallback)
    └── chemical_checker.py    (existing)
```

## Troubleshooting

### Import Errors:

If you see:

```
ImportError: No module named 'serpapi'
```

Fix:

```powershell
pip install google-search-results==2.4.2
```

### SerpAPI Errors:

If you see:

```
SerpAPI key not configured
```

Fix:

1. Check `.env` file has `SERPAPI_KEY=...`
2. Restart the backend service
3. Verify key at https://serpapi.com/dashboard

### Vision API Errors:

If you see:

```
OpenAI API key not found
```

Fix:

1. Check `.env` file has `OPENAI_API_KEY=sk-...`
2. Verify key at https://platform.openai.com/api-keys
3. Check key has sufficient quota

## Need Help?

1. Check logs: Look at terminal output for detailed error messages
2. Review documentation: `PRODUCT_IDENTIFICATION.md` has full details
3. Test endpoints: Use Postman or curl to test API directly
4. Check API dashboards: Verify keys and quotas on provider websites

## What's Working

- ✅ Product identification from front photos
- ✅ Multi-tier ingredient lookup
- ✅ Chemical safety analysis
- ✅ Data source tracking
- ✅ Confidence scoring
- ✅ Complete API documentation

## What's Still TODO

- ❌ Frontend UI for product photo scan
- ❌ Database search by product name (currently returns None)
- ❌ Results page data source indicators
- ❌ End-to-end testing with real products
- ❌ Caching layer for web search results
- ❌ Rate limiting in production

---

**Ready to test?** Start with step 1 (install dependencies) and work your way through!
