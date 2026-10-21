"""
FastAPI Main Application Entry Point
Configures CORS, middleware, lifecycle events, and routes.
"""

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import router

app = FastAPI(
    title="AI-Powered Skincare Routine Checker API",
    description=(
        "Production MLOps inference and recommendation API for automated skincare routine evaluation, "
        "evidence-backed ingredient compatibility detection, and trained model serving."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "AI-Powered Skincare Routine Checker API is running.",
        "documentation": "/docs",
        "health": "/health",
        "ready": "/ready"
    }
