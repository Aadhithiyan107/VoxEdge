"""
Abstract ASR Interface for VoxEdge
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from backend.schemas.events import ASRStatus


class ASRInterface(ABC):
    """
    Generic ASR Streaming Interface.
    Designed so remote ASR providers (Whisper, Vosk, Deepgram, Google Cloud Speech, etc.)
    can be plugged in without changing backend or frontend contracts.
    """

    @abstractmethod
    def start_stream(self) -> bool:
        """Initialize streaming connection to ASR engine."""
        pass

    @abstractmethod
    def send_audio(self, audio_chunk: bytes) -> bool:
        """Send raw audio chunk to ASR stream."""
        pass

    @abstractmethod
    def finish_stream(self) -> Optional[str]:
        """Finalize ASR stream and return final transcript if available."""
        pass

    @abstractmethod
    def get_transcript(self) -> Optional[str]:
        """Get latest partial or final transcript."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return ASR status and configuration info."""
        pass
