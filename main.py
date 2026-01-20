from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import uvicorn
import os
import time
import uuid
from datetime import datetime
import logging

# Load environment variables before importing routers so imports see keys
load_dotenv()


from routers import scan_router, index_router
from routers.identify import router as identify_router
from routers.search import router as search_router
from services.cache_service import cache_service
from models.clip_embedder import get_clip_embedder


# === Request Timing Logger Middleware ===
class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Middleware to log request timing and details for performance optimization"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID and capture start time
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store request_id in request state for use in route handlers
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Log incoming request
        print(f"\n🔵 [{timestamp}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📥 [{request_id}] {request.method} {request.url.path}")
        
        # Log query params if present
        if request.query_params:
            print(f"   🔍 Query: {dict(request.query_params)}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        status_code = response.status_code
        
        # Determine emoji based on status
        if status_code >= 400:
            status_emoji = "❌"
        elif status_code >= 300:
            status_emoji = "↪️"
        else:
            status_emoji = "✅"
        
        # Color code based on response time
        if duration_ms > 10000:
            time_emoji = "🔴"  # Very slow (>10s) - AI operations
            time_label = "VERY SLOW"
        elif duration_ms > 5000:
            time_emoji = "🟠"  # Slow (>5s)
            time_label = "SLOW"
        elif duration_ms > 2000:
            time_emoji = "🟡"  # Medium (>2s)
            time_label = "MEDIUM"
        elif duration_ms > 500:
            time_emoji = "🟢"  # Good (<2s)
            time_label = "GOOD"
        else:
            time_emoji = "⚡"  # Fast (<500ms)
            time_label = "FAST"
        
        print(f"{status_emoji} [{request_id}] {request.method} {request.url.path} → {status_code}")
        print(f"   {time_emoji} Duration: {duration_ms:.2f}ms ({time_label})")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        # Add timing header to response for frontend debugging
        response.headers["X-Request-Duration-Ms"] = f"{duration_ms:.2f}"
        response.headers["X-Request-Id"] = request_id
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    # Startup: Connect to Redis
    await cache_service.connect()
    
    # Startup: Pre-load CLIP model (avoids 15s cold start on first request!)
    # Note: First-ever run will download model (~500MB) from HuggingFace Hub
    # Subsequent runs load from cache (~/.cache/huggingface/hub/)
    print("🧠 Pre-loading CLIP model at startup (may download on first run)...")
    try:
        get_clip_embedder()
        print("✅ CLIP model ready!")
    except Exception as e:
        print(f"⚠️ CLIP model failed to load: {e}")
        print("   Recommendations endpoint may not work until model is available")
    
    yield
    # Shutdown: Close Redis connection
    await cache_service.disconnect()


# Create FastAPI app
app = FastAPI(
    title="Hippiekit AI Service",
    description="AI-powered product recognition using CLIP and Pinecone",
    version="1.0.0",
    lifespan=lifespan
)

# Add request timing middleware FIRST (so it captures total time including other middleware)
app.add_middleware(RequestTimingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add gzip compression for faster responses
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB

# Include routers
app.include_router(scan_router, tags=["scan"])
app.include_router(index_router, tags=["index"])
app.include_router(identify_router, tags=["product-identification"])
app.include_router(search_router, tags=["search"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Hippiekit AI Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hippiekit-ai"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    
    print(f"""
    ╔═══════════════════════════════════════════════════╗
    ║     Hippiekit AI Service Starting...              ║
    ╠═══════════════════════════════════════════════════╣
    ║  • Loading CLIP model (this may take a moment)    ║
    ║  • Connecting to Pinecone                         ║
    ║  • Starting server on port {port}                 ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
