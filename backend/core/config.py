"""
VoxEdge Application Configuration
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from D:\VoxEdge\.env if present
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "voxedge_multikeyword_best.keras"
if not DEFAULT_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = MODELS_DIR / "voxedge_multikeyword_final.keras"


class Settings(BaseModel):
    PROJECT_NAME: str = "VoxEdge"
    PROJECT_DESCRIPTION: str = "Low Latency and Efficient Voice Activator for Edge Devices"
    VERSION: str = "1.0.0"
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    WINDOW_SECONDS: float = 1.0
    WINDOW_SAMPLES: int = int(16000 * 1.0)
    STEP_SECONDS: float = 0.25
    STEP_SAMPLES: int = int(16000 * 0.25)
    MIN_RMS: float = 0.003
    
    # Feature extraction (MFCC)
    N_MFCC: int = 10
    N_FFT: int = 480
    HOP_LENGTH: int = 320
    
    # KWS decision parameters
    KWS_THRESHOLD: float = 0.65
    REQUIRED_HITS: int = 2
    HIT_WINDOW_SECONDS: float = 1.5
    COOLDOWN_SECONDS: float = 2.0
    
    # Model path
    MODEL_PATH: str = str(DEFAULT_MODEL_PATH)
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Deepgram ASR Settings (Read SECURELY from environment)
    DEEPGRAM_API_KEY: Optional[str] = os.getenv("DEEPGRAM_API_KEY")
    DEEPGRAM_PRIMARY_ENDPOINT: str = os.getenv("DEEPGRAM_PRIMARY_ENDPOINT", "wss://api.in.deepgram.com/v1/listen")
    DEEPGRAM_FALLBACK_ENDPOINT: str = os.getenv("DEEPGRAM_FALLBACK_ENDPOINT", "wss://api.deepgram.com/v1/listen")
    DEEPGRAM_MODEL: str = "nova-3"
    DEEPGRAM_LANGUAGE: str = "en-US"
    DEEPGRAM_ENDPOINTING: int = 300
    
    # Keyword taxonomy
    SUPPORTED_KEYWORDS: List[Dict[str, Any]] = [
        {"keyword": "VoxEdge", "status": "Active", "validated": True},
        {"keyword": "Jarvis", "status": "Active", "validated": True},
        {"keyword": "Friday", "status": "Active", "validated": True},
        {"keyword": "EDITH", "status": "Active", "validated": True},
        {"keyword": "KAREN", "status": "Active", "validated": True},
    ]


settings = Settings()
