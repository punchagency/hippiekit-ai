from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import uvicorn
import os

# Load environment variables before importing routers so imports see keys
load_dotenv()

from routers import scan_router, index_router
from routers.identify import router as identify_router
from services.cache_service import cache_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    # Startup: Connect to Redis
    await cache_service.connect()
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
    ║  • Loading CLIP model (this may take a moment)   ║
    ║  • Connecting to Pinecone                         ║
    ║  • Starting server on port {port}                    ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
