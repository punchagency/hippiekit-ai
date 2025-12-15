# Hippiekit AI Service

Python FastAPI microservice for AI-powered product recognition using CLIP embeddings and Pinecone vector database.

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

4. Add your Pinecone API key to `.env`:

   - Sign up at https://app.pinecone.io
   - Create an API key
   - Add to `.env` file

5. Run the service:

```bash
python main.py
```

6. Index products (in a new terminal):

```bash
curl -X POST "http://localhost:8001/index/products?max_products=10"
```

## API Endpoints

### POST /scan

Upload an image to find matching products

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
