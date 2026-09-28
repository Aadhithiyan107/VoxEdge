"""
Abstract KWS Interface and Result Models for VoxEdge
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from backend.schemas.events import KWSResult


class KWSInterface(ABC):
    """
    Generic KWS Engine Interface.
    Designed to support binary and multi-keyword models without breaking API contracts.
    """

    @abstractmethod
    def start(self) -> bool:
        """Initialize resources and load KWS model."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release resources."""
        pass

    @abstractmethod
    def process_audio(self, audio_chunk: np.ndarray) -> KWSResult:
        """
        Process a 1-second audio frame (16kHz float32) and return standardized KWSResult.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return engine metadata and operational state."""
        pass
