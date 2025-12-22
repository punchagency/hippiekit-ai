# Hippiekit AI Service

Python FastAPI microservice for AI-powered product recognition using CLIP embeddings, Pinecone vector database, and OpenAI Vision API.

## ✨ New Features

### 🔍 Product Identification from Photos

- Take a photo of **any product's front label** (no need to see ingredients!)
- AI identifies the product and searches for ingredients using:
  1. Database lookup (Open Food Facts, etc.)
  2. Web search via SerpAPI (manufacturer websites, retailers)
  3. AI knowledge base (GPT-4's training data)
  4. Category-based generic info (last resort)
- Complete chemical safety analysis with 200+ toxins
- See [PRODUCT_IDENTIFICATION.md](PRODUCT_IDENTIFICATION.md) for full details

### 🧪 Chemical Safety Analysis

- Detects 200+ potentially harmful chemicals
- Severity levels: CRITICAL, HIGH, MODERATE, LOW
- Safety score (0-100)
- Personalized recommendations for safer alternatives

## 🚀 Quick Start (Windows PowerShell)

Run the automated setup script:

```powershell
.\setup.ps1
```

This will:

- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create .env file
- ✅ Start the service

After setup, index your products:

```powershell
.\index-products.ps1 -MaxProducts 10
```

## 📖 Manual Setup

1. Create virtual environment:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file from `.env.example`:

```bash
Copy-Item .env.example .env
```

4. Add your API keys to `.env`:

   - **OpenAI API Key** (required for Vision API):

     - Get from: https://platform.openai.com/api-keys
     - Add: `OPENAI_API_KEY=sk-...`

   - **SerpAPI Key** (required for web search):

     - Get from: https://serpapi.com/ (free tier: 100 searches/month)
     - Add: `SERPAPI_KEY=...`

   - **Pinecone API Key** (optional for vector search):
     - Get from: https://app.pinecone.io
     - Add: `PINECONE_API_KEY=...`

5. Run the service:

```bash
python main.py
```

6. Index products (in a new terminal):

```bash
curl -X POST "http://localhost:8001/index/products?max_products=10"
```

## API Endpoints

### POST /identify/product ✨ NEW

Upload a photo of a product's front label to get complete information

- Body: multipart/form-data with `image` file
- Returns: Product info, ingredients (from database or web search), chemical analysis
- See [PRODUCT_IDENTIFICATION.md](PRODUCT_IDENTIFICATION.md) for details

Example:

```bash
curl -X POST "http://localhost:8001/identify/product" \
  -F "image=@product_photo.jpg"
```

### POST /scan/barcode

Scan a barcode to get product information from Open Food Facts

- Body: JSON with `barcode` string
- Returns: Product info with chemical analysis
- Automatically searches web if database has no ingredients

### POST /scan/vision

Upload ingredient label photo for OCR analysis

- Body: multipart/form-data with `image` file
- Returns: Extracted ingredients with chemical analysis

### POST /scan

Upload an image to find matching products (legacy CLIP search)

- Body: multipart/form-data with `image` file
- Returns: Array of matching products with scores

### POST /index/products

Index products from WordPress into Pinecone

- Query param: `max_products` (optional, default: all)
- Returns: Indexing status

### GET /health

Health check endpoint

## Architecture

- **CLIP Model**: ViT-B/32 for generating 512-dimensional image embeddings
- **Pinecone**: Vector database for similarity search
- **WordPress API**: Product data source
