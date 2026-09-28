"""
Central Application State Manager for VoxEdge
"""
import asyncio
import logging
import time
from typing import Set, Dict, Any, Optional
from fastapi import WebSocket

from backend.schemas.events import (
    AppState,
    ASRStatus,
    LatencyMetrics,
    SystemStatusPayload,
    WSEvent,
)
from backend.core.config import settings

logger = logging.getLogger("voxedge.state")


class ApplicationStateManager:
    def __init__(self):
        self.state: AppState = AppState.IDLE
        self.microphone_connected: bool = False
        self.microphone_name: Optional[str] = None
        self.kws_status: str = "Multi-Keyword KWS"
        self.active_model: str = "voxedge_multikeyword"
        self.asr_status: ASRStatus = ASRStatus.NOT_CONFIGURED
        self.asr_message: str = "ASR provider not configured. System ready for ASR streaming integration."
        self.transcript: Optional[str] = None
        self.detected_keyword: Optional[str] = None
        self.last_confidence: float = 0.0
        self.class_index: Optional[int] = None
        
        # Latency timestamps (seconds)
        self.speech_onset_time: Optional[float] = None
        self.keyword_detection_time: Optional[float] = None
        self.stream_start_time: Optional[float] = None
        self.asr_first_receive_time: Optional[float] = None
        self.transcript_complete_time: Optional[float] = None
        
        # Connected WebSocket clients
        self.active_connections: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect_ws(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")
        # Send initial full system status
        initial_event = WSEvent(
            type="system_status",
            data=self.get_system_status().dict()
        )
        await websocket.send_json(initial_event.dict())

    def disconnect_ws(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Remaining clients: {len(self.active_connections)}")

    def get_latency_metrics(self) -> LatencyMetrics:
        metrics = LatencyMetrics(
            speech_onset_time=self.speech_onset_time,
            keyword_detection_time=self.keyword_detection_time,
            stream_start_time=self.stream_start_time,
            asr_first_receive_time=self.asr_first_receive_time,
            transcript_complete_time=self.transcript_complete_time
        )
        
        # Keyword detection latency = keyword_detection_time - speech_onset_time
        if self.keyword_detection_time and self.speech_onset_time and self.keyword_detection_time >= self.speech_onset_time:
            metrics.keyword_detection_latency_ms = round((self.keyword_detection_time - self.speech_onset_time) * 1000, 2)
            
        # Keyword end to ASR latency = stream_start_time - keyword_detection_time
        if self.stream_start_time and self.keyword_detection_time and self.stream_start_time >= self.keyword_detection_time:
            metrics.keyword_end_to_asr_latency_ms = round((self.stream_start_time - self.keyword_detection_time) * 1000, 2)

        # ASR first result latency = asr_first_receive_time - stream_start_time
        if self.asr_first_receive_time and self.stream_start_time and self.asr_first_receive_time >= self.stream_start_time:
            metrics.asr_first_result_latency_ms = round((self.asr_first_receive_time - self.stream_start_time) * 1000, 2)
            
        # Total latency = transcript_complete_time - speech_onset_time
        if self.transcript_complete_time and self.speech_onset_time and self.transcript_complete_time >= self.speech_onset_time:
            metrics.total_latency_ms = round((self.transcript_complete_time - self.speech_onset_time) * 1000, 2)
            
        return metrics

    def get_system_status(self) -> SystemStatusPayload:
        return SystemStatusPayload(
            state=self.state,
            microphone_connected=self.microphone_connected,
            microphone_name=self.microphone_name,
            kws_status=self.kws_status,
            active_model=self.active_model,
            asr_status=self.asr_status,
            asr_message=self.asr_message,
            transcript=self.transcript,
            detected_keyword=self.detected_keyword,
            last_confidence=round(self.last_confidence, 4),
            class_index=self.class_index,
            latency=self.get_latency_metrics()
        )

    def update_state(self, new_state: AppState):
        if self.state != new_state:
            logger.info(f"State transition: {self.state.value} -> {new_state.value}")
            self.state = new_state
            self.broadcast_event("state_change", {"state": new_state.value})

    def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        event = WSEvent(type=event_type, data=data)
        payload = event.dict()
        
        if not self.active_connections:
            return

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_to_all(payload), self._loop)
        else:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self._send_to_all(payload), loop)
            except RuntimeError:
                pass

    async def _send_to_all(self, payload: Dict[str, Any]):
        disconnected = set()
        for ws in self.active_connections:
            try:
                await ws.send_json(payload)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.active_connections.discard(ws)


# Global singleton application state
app_state = ApplicationStateManager()
