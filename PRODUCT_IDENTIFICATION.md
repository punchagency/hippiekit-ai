# Product Identification System

## Overview

The Hippiekit AI Service now supports **advanced product identification** from front-facing photos, even when ingredients are not visible in the image. This uses a multi-tier approach to ensure complete product information.

## Features

### 1. Front-Facing Product Photo Analysis

- Take a photo of the **front of any product** (not the ingredients label)
- AI extracts: product name, brand, category, type, marketing claims, certifications, container info
- Works with any personal care, cosmetic, or household product

### 2. Multi-Tier Ingredient Lookup

When a product is identified but ingredients are not visible:

#### Tier 1: Database Search (High Confidence)

- Searches Open Food Facts and other product databases
- Returns official ingredient lists if available
- **Confidence: High**

#### Tier 2: Web Search (High Confidence)

- Uses SerpAPI (Google Search API) to find ingredients from:
  - Manufacturer websites
  - Retailer product pages
  - Beauty/skincare databases (e.g., EWG, Paula's Choice)
  - Consumer review sites
- AI extracts and validates ingredient lists from search results
- **Confidence: High** (when clear match found)

#### Tier 3: AI Knowledge (Medium Confidence)

- Uses GPT-4's training data about known products
- Good for popular/well-known products
- **Confidence: Medium**

#### Tier 4: Category Generic (Low Confidence)

- Returns typical ingredients for product category
- Example: "Body Wash" → common surfactants, preservatives, fragrances
- Only used when all other methods fail
- **Confidence: Low** with clear disclaimer

### 3. Chemical Safety Analysis

All ingredient lists (regardless of source) are analyzed for:

- 200+ potentially harmful chemicals
- Severity levels: CRITICAL, HIGH, MODERATE, LOW
- Safety score (0-100)
- Personalized recommendations

## API Endpoints

### POST /identify/product

Identifies a product from a photo and returns complete information.

**Request:**

```bash
curl -X POST "http://localhost:8001/identify/product" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@product_photo.jpg"
```

**Response:**

```json
{
  "product_name": "Moisturizing Body Wash",
  "brand": "Example Brand",
  "category": "Body Wash",
  "product_type": "Liquid Soap",
  "ingredients": "Water, Sodium Laureth Sulfate, Cocamidopropyl Betaine...",
  "data_source": "web_search",
  "confidence": "high",
  "ingredients_note": "Ingredients found from manufacturer website",
  "chemical_analysis": {
    "flags": [
      {
        "chemical": "Sodium Laureth Sulfate",
        "category": "Surfactants",
        "severity": "MODERATE",
        "why_flagged": "Harsh cleaning agent that can strip natural oils"
      }
    ],
    "safety_score": 85,
    "recommendations": {
      "avoid": ["Products with sulfates", "Synthetic fragrances"],
      "look_for": ["Gentle surfactants", "Natural preservatives"],
      "certifications": ["EWG Verified", "USDA Organic"]
    }
  },
  "marketing_claims": ["Dermatologist Tested", "Hypoallergenic"],
  "certifications_visible": ["Cruelty Free"],
  "container_info": {
    "material": "Plastic",
    "type": "Bottle",
    "size": "16 fl oz"
  }
}
```

## Environment Setup

### Required API Keys

1. **OpenAI API Key** (Required)

   - For Vision API and ingredient extraction
   - Get from: https://platform.openai.com/api-keys
   - Set: `OPENAI_API_KEY=sk-...`

2. **SerpAPI Key** (Required for web search)

   - For Google search integration
   - Get from: https://serpapi.com/
   - Free tier: 100 searches/month
   - Paid: $50/month for 5,000 searches
   - Set: `SERPAPI_KEY=your_key_here`

3. **Pinecone API Key** (Optional)
   - For vector search functionality
   - Set: `PINECONE_API_KEY=your_key_here`

### Installation

```bash
cd ai-service

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

### Running the Service

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8001
```

## How It Works

### Example Flow: Body Wash with No Visible Ingredients

1. **User takes photo** of body wash bottle (front label only)

2. **Vision AI identifies product:**

   ```json
   {
     "product_name": "Hydrating Body Wash",
     "brand": "CeraVe",
     "category": "Body Wash",
     "confidence": "high"
   }
   ```

3. **Database search** for "CeraVe Hydrating Body Wash"

   - ✅ Found in database
   - ❌ But ingredients field is empty

4. **Web search triggered:**

   - Google search: "CeraVe Hydrating Body Wash ingredients"
   - Results from: cerave.com, ulta.com, walgreens.com
   - AI extracts: "Water, Glycerin, Cocamidopropyl Betaine..."
   - **Confidence: High**

5. **Chemical analysis:**

   - Checks 200+ chemicals
   - Flags: None found (clean product)
   - Safety score: 95/100

6. **Response sent to user** with full details + data source indicator

## Data Source Indicators

Responses include metadata about where information came from:

- `"data_source": "database"` - From Open Food Facts or similar
- `"data_source": "web_search"` - Found via Google search
- `"data_source": "ai_knowledge"` - GPT-4's training data
- `"data_source": "category_generic"` - Generic template
- `"data_source": "ai_vision"` - Full AI analysis from image

Users see clear indicators in the UI about data reliability.

## Cost Considerations

### OpenAI Vision API

- ~$0.01 per image (high detail)
- GPT-4o for identification + extraction
- GPT-4o-mini for web result parsing (cheaper)

### SerpAPI

- Free: 100 searches/month
- Paid: $50/month = 5,000 searches
- Cost per search: ~$0.01

### Example Monthly Cost (1000 users)

- Vision API: $10-20 (assuming 2 images per user)
- SerpAPI: $50 (5,000 searches)
- **Total: ~$70/month**

## Limitations

1. **Database Search by Name:** Currently not fully implemented (would need full database download or search API)

2. **Web Search Accuracy:** Depends on search result quality

   - Works best for popular brands
   - May struggle with obscure/local products
   - AI validation helps filter bad results

3. **AI Knowledge Cutoff:** GPT-4's training data has a cutoff date

   - May not know about very new products
   - Falls back to web search or category generic

4. **Rate Limits:**
   - SerpAPI: 100 searches/month on free tier
   - OpenAI: Based on your tier (usually high)

## Testing

### Test with a product photo:

```bash
# Using curl
curl -X POST "http://localhost:8001/identify/product" \
  -F "image=@test_product.jpg"

# Using Python
import requests

with open("test_product.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8001/identify/product",
        files={"image": f}
    )
    print(response.json())
```

### Test scenarios:

1. ✅ **Popular product** (e.g., CeraVe, Dove) → Should use database or web search
2. ✅ **Obscure product** → Should fall back to AI knowledge or category
3. ✅ **Front photo only** → Should still extract product info
4. ✅ **Blurry photo** → Should handle gracefully with lower confidence

## Future Enhancements

- [ ] Full database search by product name (requires API or local database)
- [ ] Multiple search engine support (Bing, DuckDuckGo)
- [ ] Cache web search results to reduce API costs
- [ ] User feedback loop to improve accuracy
- [ ] Support for products in different languages
- [ ] Barcode generation from product name (reverse lookup)

## Troubleshooting

### "SerpAPI key not found"

- Check `.env` file has `SERPAPI_KEY=...`
- Restart the service after adding key

### "Web search failed"

- Check SerpAPI quota (100/month on free tier)
- Check internet connectivity
- Verify API key is valid

### "Low confidence" results

- Product might be too obscure
- Try with a clearer photo
- Check if brand name is visible
- May need to manually enter ingredients

### "AI analysis timeout"

- Image might be too large (compress it)
- Network issue with OpenAI API
- Check API key and quota

## Security & Privacy

- **No user data stored**: Images processed in memory only
- **API keys secured**: Environment variables only
- **HTTPS required**: In production, use HTTPS for image uploads
- **Rate limiting**: Implement in production to prevent abuse
- **Content filtering**: Validate image uploads (size, format, content)
