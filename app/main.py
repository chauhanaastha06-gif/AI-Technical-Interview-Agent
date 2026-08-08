from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.interview_routes import router as interview_router
from app.utils.logging import logger

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = "MOCK MODE (Deterministic LLM)" if settings.is_mock_mode else f"ANTHROPIC CLAUDE ({settings.ANTHROPIC_MODEL})"
    logger.info(f"AI Interview Agent starting up. Operating Mode: {mode}")
    logger.info(f"Configured MAX_TURNS: {settings.MAX_TURNS}")
    yield

app = FastAPI(
    title="AI Interview Agent Backend",
    description="Adaptive technical interview agent backend built for the AI cohort hackathon.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow flexible integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include interview router
app.include_router(interview_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "mock_mode": settings.is_mock_mode,
        "model": settings.ANTHROPIC_MODEL if not settings.is_mock_mode else "mock-deterministic",
        "max_turns": settings.MAX_TURNS,
    }


# Serve static frontend UI if directory exists
from app.config import BASE_DIR
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
