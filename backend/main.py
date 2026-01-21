"""
AI Photo Recognizer - FastAPI Application

Main entry point for the backend API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.core.config import settings
from backend.core.database import Base, engine, SessionLocal
from backend.routers import auth, analysis, admin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting AI Photo Recognizer API...")

    if settings.AUTO_CREATE_TABLES:
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created (AUTO_CREATE_TABLES=True)")
        except Exception:
            logger.exception("Failed to create database tables")
    else:
        logger.info("ℹSkipping table creation (AUTO_CREATE_TABLES=False)")

    app.state.ml_engine = None
    try:
        from backend.services.ml_engine import ml_engine
        app.state.ml_engine = ml_engine

        if ml_engine.is_loaded:
            logger.info("ML Model loaded: %s", ml_engine.backbone_name)
        else:
            logger.warning("⚠ML Model not loaded!")
    except Exception:
        logger.exception("Failed to initialize ML engine")

    try:
        from backend.services.metrics_loader import metrics_engine
        logger.info("Custom metrics loaded: %d", metrics_engine.metric_count)
    except Exception:
        logger.exception("Failed to load metrics")

    logger.info("API Ready!")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="AI Photo Recognizer API",
    version="2.5.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(admin.router)


@app.get("/", tags=["System"])
def root(request: Request):
    ml_engine = getattr(request.app.state, "ml_engine", None)

    is_loaded = ml_engine.is_loaded if ml_engine else False
    model_info = {
        "loaded": is_loaded,
        "type": ml_engine.model_type if ml_engine else "unknown",
        "backbone": ml_engine.backbone_name if ml_engine else "unknown",
        "threshold": ml_engine.threshold if ml_engine else 0.5,
    }

    return {
        "system": "AI Photo Recognizer",
        "version": "2.5.0",
        "status": "online",
        "device": settings.DEVICE,
        "model": model_info,
    }


@app.get("/health", tags=["System"])
def health_check(request: Request):
    ml_engine = getattr(request.app.state, "ml_engine", None)
    is_loaded = ml_engine.is_loaded if ml_engine else False

    db = None
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        logger.exception("Health check DB failed")
        db_status = "error"
    finally:
        if db:
            db.close()

    return {
        "status": "healthy" if is_loaded and db_status == "connected" else "degraded",
        "model_loaded": is_loaded,
        "database": db_status,
    }


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )