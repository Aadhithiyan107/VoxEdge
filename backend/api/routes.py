"""
FastAPI Routes for VoxEdge REST API
"""
from fastapi import APIRouter, HTTPException
from backend.core.config import settings
from backend.core.state import app_state
from backend.audio.stream_manager import stream_manager
from backend.schemas.events import SystemStatusPayload

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION
    }


@router.get("/status", response_model=SystemStatusPayload)
def get_status():
    return app_state.get_system_status()


@router.get("/config")
def get_config():
    return {
        "project_name": settings.PROJECT_NAME,
        "sample_rate": settings.SAMPLE_RATE,
        "window_seconds": settings.WINDOW_SECONDS,
        "step_seconds": settings.STEP_SECONDS,
        "kws_threshold": settings.KWS_THRESHOLD,
        "required_hits": settings.REQUIRED_HITS,
        "cooldown_seconds": settings.COOLDOWN_SECONDS,
        "model_path": settings.MODEL_PATH,
        "supported_keywords": settings.SUPPORTED_KEYWORDS,
    }


@router.post("/session/start")
def start_session():
    success = stream_manager.start()
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to start audio session. Check microphone connectivity or model path."
        )
    return {
        "status": "started",
        "message": "VoxEdge session started successfully.",
        "microphone": app_state.microphone_name
    }


@router.post("/session/stop")
def stop_session():
    stream_manager.stop()
    return {
        "status": "stopped",
        "message": "VoxEdge session stopped successfully."
    }
