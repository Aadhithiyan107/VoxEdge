"""
VoxEdge Binary KWS Implementation
Consumes models/voxedge_binary_best.keras using the exact validated preprocessing pipeline.
"""
import time
import logging
from typing import Dict, Any, List
import numpy as np
import librosa

from backend.kws.interface import KWSInterface
from backend.schemas.events import KWSResult
from backend.core.config import settings

logger = logging.getLogger("voxedge.kws")


KWS_CLASSES = [
    "voxedge",
    "jarvis",
    "friday",
    "edith",
    "karen",
    "unknown"
]

DISPLAY_KEYWORDS = {
    "voxedge": "VoxEdge",
    "jarvis": "Jarvis",
    "friday": "Friday",
    "edith": "EDITH",
    "karen": "KAREN",
    "unknown": "unknown"
}


class VoxEdgeBinaryKWSService(KWSInterface):
    """
    Multi-Keyword DS-CNN KWS service consuming voxedge_multikeyword_best.keras.
    Supports 6 output classes: voxedge (0), jarvis (1), friday (2), edith (3), karen (4), unknown (5).
    """

    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        # State tracking: list of (timestamp, class_index)
        self.recent_hits: List[tuple] = []
        self.cooldown_until: float = 0.0
        
        # Hyperparameters matching training/live_test.py
        self.sample_rate = settings.SAMPLE_RATE
        self.window_samples = settings.WINDOW_SAMPLES
        self.threshold = settings.KWS_THRESHOLD
        self.required_hits = settings.REQUIRED_HITS
        self.hit_window_seconds = settings.HIT_WINDOW_SECONDS
        self.cooldown_seconds = settings.COOLDOWN_SECONDS
        self.min_rms = settings.MIN_RMS
        self.n_mfcc = settings.N_MFCC
        self.n_fft = settings.N_FFT
        self.hop_length = settings.HOP_LENGTH

    def start(self) -> bool:
        if self.is_loaded:
            return True
            
        logger.info(f"Loading KWS model from: {self.model_path}")
        try:
            import tensorflow as tf
            # Suppress verbose TF logging
            tf.get_logger().setLevel('ERROR')
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"VoxEdge Multi-Keyword KWS Model loaded successfully from {self.model_path}.")
            return True
        except Exception as e:
            logger.error(f"Failed to load VoxEdge KWS model: {e}")
            self.is_loaded = False
            return False

    def stop(self) -> None:
        self.model = None
        self.is_loaded = False
        self.recent_hits.clear()
        logger.info("VoxEdge KWS Service stopped.")

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Exact robust audio normalization from training/live_test.py."""
        audio = np.asarray(audio, dtype=np.float32).flatten()
        audio = audio - np.mean(audio)

        frame_length = 400
        hop_length = 160

        if len(audio) >= frame_length:
            rms_frames = librosa.feature.rms(
                y=audio,
                frame_length=frame_length,
                hop_length=hop_length,
                center=False
            )[0]

            if len(rms_frames) > 0:
                peak_rms = np.max(rms_frames)
                activity_threshold = max(self.min_rms, peak_rms * 0.20)
                active = np.where(rms_frames >= activity_threshold)[0]

                if len(active) > 0:
                    start_frame = active[0]
                    end_frame = active[-1]

                    start_sample = start_frame * hop_length
                    end_sample = end_frame * hop_length + frame_length

                    context = int(0.10 * self.sample_rate)
                    start_sample = max(0, start_sample - context)
                    end_sample = min(len(audio), end_sample + context)

                    active_audio = audio[start_sample:end_sample]

                    if len(active_audio) > 0:
                        normalized = np.zeros(self.window_samples, dtype=np.float32)
                        if len(active_audio) >= self.window_samples:
                            active_audio = active_audio[:self.window_samples]
                            normalized[:] = active_audio
                        else:
                            start = (self.window_samples - len(active_audio)) // 2
                            normalized[start:start + len(active_audio)] = active_audio
                        audio = normalized

        rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
        if rms > 1e-6:
            audio = audio * (0.08 / rms)

        return np.clip(audio, -1.0, 1.0)

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features identical to training/live_test.py."""
        audio = self.normalize_audio(audio)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        mfcc = mfcc.astype(np.float32)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)
        return mfcc

    def process_audio(self, audio_chunk: np.ndarray) -> KWSResult:
        now = time.time()
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

        if not self.is_loaded or self.model is None:
            return KWSResult(
                detected=False,
                keyword=None,
                confidence=0.0,
                class_index=5,
                timestamp=timestamp_iso,
                model="voxedge_multikeyword"
            )

        # Check cooldown
        if now < self.cooldown_until:
            return KWSResult(
                detected=False,
                keyword=None,
                confidence=0.0,
                class_index=5,
                timestamp=timestamp_iso,
                model="voxedge_multikeyword"
            )

        # Basic activity check
        raw_rms = float(np.sqrt(np.mean(audio_chunk ** 2) + 1e-8))
        if raw_rms < self.min_rms:
            self.recent_hits.clear()
            return KWSResult(
                detected=False,
                keyword="unknown",
                confidence=0.0,
                class_index=5,
                timestamp=timestamp_iso,
                model="voxedge_multikeyword"
            )

        # Run multi-class inference
        try:
            features = self.preprocess_audio(audio_chunk)
            prediction = self.model.predict(features, verbose=0)
            probs = prediction[0]
            if len(probs) >= 6:
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
            else:
                top_idx = 0
                confidence = float(probs[0])

            top_keyword = KWS_CLASSES[top_idx] if top_idx < len(KWS_CLASSES) else "unknown"
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return KWSResult(
                detected=False,
                keyword="unknown",
                confidence=0.0,
                class_index=5,
                timestamp=timestamp_iso,
                model="voxedge_multikeyword"
            )

        # UNKNOWN class (5) must NOT be treated as a wake keyword!
        if top_idx == 5 or top_keyword == "unknown" or confidence < self.threshold:
            self.recent_hits = [h for h in self.recent_hits if now - h[0] <= self.hit_window_seconds]
            return KWSResult(
                detected=False,
                keyword=DISPLAY_KEYWORDS.get(top_keyword, top_keyword),
                confidence=confidence,
                class_index=top_idx,
                timestamp=timestamp_iso,
                model="voxedge_multikeyword"
            )

        # Hit tracking for valid keyword classes (0 to 4)
        self.recent_hits.append((now, top_idx))
        self.recent_hits = [h for h in self.recent_hits if now - h[0] <= self.hit_window_seconds]

        # Check multi-hit confirmation for top_idx
        hits_for_class = sum(1 for _, c_idx in self.recent_hits if c_idx == top_idx)

        if hits_for_class >= self.required_hits:
            self.recent_hits.clear()
            self.cooldown_until = now + self.cooldown_seconds
            
            display_keyword = DISPLAY_KEYWORDS.get(top_keyword, top_keyword)

            logger.info(f"KEYWORD CONFIRMED! Keyword='{display_keyword}' (class_index={top_idx}), Confidence={confidence * 100:.2f}%")
            return KWSResult(
                detected=True,
                keyword=display_keyword,
                confidence=confidence,
                class_index=top_idx,
                timestamp=timestamp_iso,
                model="voxedge_multikeyword"
            )

        display_keyword = DISPLAY_KEYWORDS.get(top_keyword, top_keyword)
        return KWSResult(
            detected=False,
            keyword=display_keyword,
            confidence=confidence,
            class_index=top_idx,
            timestamp=timestamp_iso,
            model="voxedge_multikeyword"
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_path": self.model_path,
            "model_type": "voxedge_multikeyword",
            "supported_classes": KWS_CLASSES,
            "threshold": self.threshold,
            "required_hits": self.required_hits,
            "cooldown_seconds": self.cooldown_seconds,
        }
