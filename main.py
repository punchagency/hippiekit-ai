from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import os

from routers import scan_router, index_router

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Hippiekit AI Service",
    description="AI-powered product recognition using CLIP and Pinecone",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scan_router, tags=["scan"])
app.include_router(index_router, tags=["index"])

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
