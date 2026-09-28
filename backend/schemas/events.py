"""
Event schemas and payload models for VoxEdge application.
"""
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AppState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEECH_DETECTED = "SPEECH_DETECTED"
    VERIFYING = "VERIFYING"
    KEYWORD_DETECTED = "KEYWORD_DETECTED"
    STREAMING = "STREAMING"
    TRANSCRIBING = "TRANSCRIBING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class ASRStatus(str, Enum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    READY = "READY"
    STREAMING = "STREAMING"
    TRANSCRIBING = "TRANSCRIBING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class KWSResult(BaseModel):
    detected: bool = False
    keyword: Optional[str] = None
    confidence: float = 0.0
    class_index: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    model: str = "voxedge_multikeyword"


class AudioActivityPayload(BaseModel):
    rms: float
    speech_active: bool
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class KWSDetectionPayload(BaseModel):
    keyword: str
    confidence: float
    class_index: int
    model: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class LatencyMetrics(BaseModel):
    speech_onset_time: Optional[float] = None
    keyword_detection_time: Optional[float] = None
    stream_start_time: Optional[float] = None
    asr_first_receive_time: Optional[float] = None
    transcript_complete_time: Optional[float] = None
    
    keyword_detection_latency_ms: Optional[float] = None
    keyword_end_to_asr_latency_ms: Optional[float] = None
    asr_first_result_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None


class SystemStatusPayload(BaseModel):
    state: AppState
    microphone_connected: bool
    microphone_name: Optional[str] = None
    kws_status: str
    active_model: str
    asr_status: ASRStatus
    asr_message: str
    transcript: Optional[str] = None
    detected_keyword: Optional[str] = None
    last_confidence: float = 0.0
    class_index: Optional[int] = None
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class WSEvent(BaseModel):
    type: str  # e.g., "state_change", "audio_activity", "kws_detection", "asr_update", "latency_update", "error"
    data: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
