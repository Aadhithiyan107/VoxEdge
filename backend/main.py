"""
VoxEdge FastAPI Application Entry Point
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.state import app_state
from backend.api.routes import router as api_router
from backend.audio.stream_manager import stream_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voxedge.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} backend...")
    loop = asyncio.get_running_loop()
    app_state.set_event_loop(loop)
    
    # Auto-start stream manager session
    stream_manager.start()
    
    yield
    
    # Shutdown
    logger.info("Stopping VoxEdge backend...")
    stream_manager.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS middleware for local frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API routes
app.include_router(api_router, prefix="/api")


@app.websocket("/ws/events")
async def websocket_events_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time frontend synchronization.
    Broadcasts state transitions, live audio levels, KWS detections, and latency updates.
    """
    await app_state.connect_ws(websocket)
    try:
        while True:
            # Keep connection open and accept ping/pong or control messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        app_state.disconnect_ws(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        app_state.disconnect_ws(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )
