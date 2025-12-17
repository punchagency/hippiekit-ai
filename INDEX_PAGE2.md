# Indexing Products from WordPress to Pinecone

## Quick Start: Index Page 2 (20 Products)

If you want to index 20 products from WordPress page 2 into Pinecone, run this command:

```bash
cd server/python-ai-service
python index_page2.py
```

This script will:

1. ✅ Fetch 20 products from page 2 of your WordPress store
2. ✅ Download product images
3. ✅ Generate CLIP embeddings for each image
4. ✅ Upload vectors to your Pinecone index
5. ✅ Display indexing statistics

### Expected Output

```
============================================================
Indexing 20 Products from Page 2 of WordPress
============================================================

📥 Fetching 20 products from WordPress page 2...
✓ Found 20 products with images on page 2

🤖 Processing 20 products with CLIP model...
------------------------------------------------------------
[ 1/20] Processing: Product Name...
       ✓ Embedded successfully (embedding size: 512)
[ 2/20] Processing: Product Name...
       ✓ Embedded successfully (embedding size: 512)
...
------------------------------------------------------------

✓ Successfully processed 20/20 products

📤 Upserting 20 products to Pinecone...
------------------------------------------------------------
✓ Products successfully upserted to Pinecone!

📊 Pinecone Index Statistics:
------------------------------------------------------------
Total vectors in index: 20
Index dimension: 512
Index fullness: 0.00%
------------------------------------------------------------

============================================================
✅ Indexing Complete!
============================================================
```

## API Endpoint: Index via HTTP

You can also index products via the HTTP API:

### Index all products from page 2:

```bash
curl -X POST "http://localhost:8001/index/products?page=2&max_products=20"
```

### Index all products from all pages:

```bash
curl -X POST "http://localhost:8001/index/products"
```

### Check index statistics:

```bash
curl "http://localhost:8001/index/stats"
```

## What is Page 2?

In WordPress REST API pagination:

- **Page 1**: First 100 products (or per_page limit)
- **Page 2**: Next 100 products
- Each page returns up to 100 products by default

The `index_page2.py` script fetches 20 products from page 2 (the second batch).

## Requirements

Make sure your `.env` file has these variables set:

```env
WORDPRESS_API_URL=https://your-wordpress-site.com/wp-json/wp/v2/products/
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=hippiekit
PINECONE_ENVIRONMENT=your_pinecone_environment
```

## Troubleshooting

### No products found:

- Check if WordPress products exist at the specified page
- Verify the `WORDPRESS_API_URL` is correct
- Ensure products have featured images

### Embedding errors:

- Make sure CLIP model downloads (happens on first run)
- Check GPU memory if you have a GPU
- The model is ~600MB and will be cached locally

### Pinecone errors:

- Verify API key is correct
- Check index name and environment
- Ensure your Pinecone subscription supports the dimension (512)
