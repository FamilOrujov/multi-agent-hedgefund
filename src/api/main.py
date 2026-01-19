
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import analysis_router, agents_router, health_router
from src.api.dependencies import get_ollama_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan - startup and shutdown events."""
    # Startup
    print("🚀 Starting Aegis Flux API...")
    
    # Pre-warm the Ollama client
    try:
        client = get_ollama_client()
        if client.is_available():
            print(f"✅ Ollama connected: {client.model}")
        else:
            print("⚠️  Ollama not available - LLM features disabled")
    except Exception as e:
        print(f"⚠️  Ollama connection failed: {e}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Aegis Flux API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Aegis Flux API",
        description="Multi-Agent AI Investment Analysis System API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(agents_router, prefix="/api/v1")
    app.include_router(analysis_router, prefix="/api/v1")
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Aegis Flux API",
            "version": "0.1.0",
            "description": "Multi-Agent AI Investment Analysis System",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    
    return app


# Create app instance
app = create_app()
