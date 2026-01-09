# OpenAI Native Web Search Integration

## Overview

Updated the web search service to use **OpenAI's native web search** feature through the Responses API instead of SerpAPI and fallback methods. This provides more reliable, up-to-date ingredient information directly from OpenAI.

## What Changed

### Before

- **3-tier search system**:
  1. AI Knowledge Base (GPT-4o training data)
  2. SerpAPI Google search (requires API key)
  3. Category-based generic fallback
- Required SERPAPI_KEY environment variable
- Multiple API calls and parsing steps
- Generic fallback for unknown products

### After

- **Single OpenAI web search** using Responses API
- Uses `web_search` tool type
- Direct access to real-time web information
- Returns citations and sources
- No additional API keys needed (only OPENAI_API_KEY)

## Implementation Details

### File: `ai-service/services/web_search_service.py`

```python
async def search_product_ingredients(
    self,
    product_name: str,
    brand: str,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for product ingredients using OpenAI's native web search.
    Uses the Responses API with web_search tool.
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        input=query,
        include=["web_search_call.action.sources"]
    )
```

### Return Format

```python
{
    'ingredients': str,  # Comma-separated ingredient list
    'source': 'openai_web_search',
    'confidence': 'high' | 'medium' | 'low',  # Based on citation count
    'citations': [  # URLs cited in response
        {'url': str, 'title': str}
    ],
    'sources': [str],  # All URLs consulted
    'note': str  # Disclaimer message
}
```

## Features

1. **Real-time Web Access**: Fetches latest ingredient information from manufacturer websites, databases, and trusted sources
2. **Source Citations**: Returns URLs and titles of all cited sources
3. **Transparency**: Includes all sources consulted during search
4. **Confidence Scoring**: Based on number of citations found
5. **Works for All Consumer Products**: Food, skincare, pet products, kitchen, bathroom, etc.

## Usage

### In Code

```python
from services.web_search_service import web_search_service

result = await web_search_service.search_product_ingredients(
    product_name="Daily Moisturizing Lotion",
    brand="Aveeno",
    category="skincare"
)

if result:
    print(result['ingredients'])
    print(f"Sources: {len(result['sources'])}")
```

### Testing

```bash
cd ai-service
python test_openai_web_search.py
```

## Benefits

1. **More Accurate**: Gets actual product ingredients from web vs guessing
2. **Simpler**: Single API call instead of multi-tier fallback
3. **No Extra Keys**: Only needs OPENAI_API_KEY (no SERPAPI_KEY)
4. **Better Citations**: Provides source URLs for verification
5. **Real-time Data**: Always gets latest information
6. **Cost Effective**: Uses built-in OpenAI feature (included in model pricing)

## API Cost

Web search incurs a tool call cost in addition to model usage. See [OpenAI Pricing](https://platform.openai.com/docs/pricing#built-in-tools) for details.

## Related Files Modified

1. **ai-service/services/web_search_service.py** - Main implementation
2. **ai-service/routers/scan.py** - Updated `infer_ingredients_from_context()` to use web search
3. **ai-service/test_openai_web_search.py** - New test file

## Integration Points

The web search is automatically used when:

- Product has no ingredients in OpenFacts database
- Ingredients text is < 20 characters
- Called from `/barcode/ingredients/separate` endpoint
- Called from `infer_ingredients_from_context()` function

## Removed Components

- SerpAPI integration (`_search_with_serpapi`)
- AI knowledge base fallback (`_search_with_ai_knowledge`)
- Category generic templates (`_get_category_generic_info`)
- SERPAPI_KEY environment variable requirement

## Kept Components

- `fetch_brand_logo()` - Still uses Brandfetch API for brand logos
- Brand matching logic (`_find_best_brand_match`)

## Next Steps

1. Test with various product types
2. Monitor API costs for web search tool calls
3. Consider caching web search results to reduce costs
4. Add retry logic for failed searches
