"""
VoxEdge Audio Stream Manager
Manages microphone capture, VAD activity detection, continuous KWS sliding window inference,
and post-keyword audio stream dispatch to Deepgram Speech-to-Text.
"""
import time
import logging
import threading
from typing import List, Dict, Any, Optional
import numpy as np
import sounddevice as sd

from backend.core.config import settings
from backend.core.state import app_state
from backend.schemas.events import AppState, ASRStatus
from backend.kws.service import VoxEdgeBinaryKWSService
from backend.asr.service import DeepgramASRService

logger = logging.getLogger("voxedge.audio")


class AudioStreamManager:
    def __init__(self):
        self.kws_service = VoxEdgeBinaryKWSService(settings.MODEL_PATH)
        self.asr_service = DeepgramASRService()
        
        self.sample_rate = settings.SAMPLE_RATE
        self.window_samples = settings.WINDOW_SAMPLES
        self.step_samples = settings.STEP_SAMPLES
        self.min_rms = settings.MIN_RMS
        
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.latest_step_chunk = np.zeros(self.step_samples, dtype=np.float32)
        self.new_step_available = False
        
        # Post-keyword audio capture queue (captures audio ONLY after keyword confirmation while Deepgram connects)
        self.is_capturing_post_keyword = False
        self.post_keyword_pcm_queue: List[tuple] = []
        self.buffer_lock = threading.Lock()
        
        self.device_index: Optional[int] = None
        self.device_name: Optional[str] = None
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Rate limiters for audio visualizer broadcasting
        self.last_audio_broadcast_time = 0.0

    def discover_microphones(self) -> List[Dict[str, Any]]:
        try:
            devices = sd.query_devices()
            inputs = []
            for index, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    inputs.append({
                        "index": index,
                        "name": device["name"],
                        "samplerate": device["default_samplerate"],
                        "max_channels": device["max_input_channels"]
                    })
            return inputs
        except Exception as e:
            logger.error(f"Error querying microphone devices: {e}")
            return []

    def select_best_microphone(self) -> Optional[int]:
        """Device-independent microphone selection matching training/live_test.py."""
        devices = self.discover_microphones()
        if not devices:
            return None

        # 1. Try system default
        try:
            default_input = sd.default.device[0]
            if default_input is not None and default_input >= 0:
                dev_info = sd.query_devices(default_input)
                if self._test_microphone_index(default_input):
                    self.device_name = dev_info["name"]
                    return default_input
        except Exception:
            pass

        # 2. Search for first working input
        for dev in devices:
            idx = dev["index"]
            if self._test_microphone_index(idx):
                self.device_name = dev["name"]
                return idx

        return None

    def _test_microphone_index(self, index: int) -> bool:
        try:
            test_stream = sd.InputStream(
                device=index,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.step_samples
            )
            test_stream.start()
            time.sleep(0.1)
            test_stream.stop()
            test_stream.close()
            return True
        except Exception:
            return False

    def _scale_audio_for_asr(self, samples: np.ndarray) -> tuple[bytes, float, float, float, float]:
        """
        Scales audio copy for ASR (Deepgram) to preserve normal microphone amplitude (~0.04-0.08 RMS).
        Does NOT affect KWS float32 samples.
        Returns: (pcm_bytes, raw_rms, asr_rms, gain, peak)
        """
        raw_rms = float(np.sqrt(np.mean(samples ** 2) + 1e-8))
        if raw_rms > 1e-6:
            # Target RMS around 0.06 (-24 dBFS), max gain of 25x, min gain 1.0x
            gain = max(min(0.06 / (raw_rms + 1e-6), 25.0), 1.0)
        else:
            gain = 1.0

        asr_samples = np.clip(samples * gain, -1.0, 1.0)
        asr_rms = float(np.sqrt(np.mean(asr_samples ** 2) + 1e-8))
        peak = float(np.max(np.abs(asr_samples)))
        pcm_bytes = (asr_samples * 32767.0).astype(np.int16).tobytes()

        return pcm_bytes, raw_rms, asr_rms, gain, peak

    def _audio_callback(self, indata, frames, callback_time, status):
        if status:
            logger.warning(f"Audio stream status: {status}")

        samples = indata[:, 0].astype(np.float32)
        pcm_bytes, raw_rms, asr_rms, gain, peak = self._scale_audio_for_asr(samples)

        with self.buffer_lock:
            # Capture ONLY post-keyword command PCM chunks while Deepgram is connecting
            if self.is_capturing_post_keyword:
                self.post_keyword_pcm_queue.append((pcm_bytes, raw_rms, asr_rms, gain, peak))

            if len(samples) >= self.step_samples:
                self.latest_step_chunk = samples[-self.step_samples:].copy()
            else:
                self.latest_step_chunk = samples.copy()
            self.new_step_available = True

            # Update rolling 1-second window for KWS
            if len(samples) >= self.window_samples:
                self.audio_buffer = samples[-self.window_samples:].copy()
            else:
                self.audio_buffer = np.concatenate(
                    (self.audio_buffer[len(samples):], samples)
                )

    def start(self) -> bool:
        if self.is_running:
            return True

        logger.info("Initializing VoxEdge Audio Stream Manager...")
        
        # 1. Initialize KWS Engine
        if not self.kws_service.start():
            app_state.kws_status = "Error loading model"
            app_state.update_state(AppState.ERROR)
            return False
            
        app_state.kws_status = "Ready"

        # 2. Select microphone
        self.device_index = self.select_best_microphone()
        if self.device_index is None:
            logger.error("No functional audio input device detected.")
            app_state.microphone_connected = False
            app_state.microphone_name = None
            app_state.update_state(AppState.ERROR)
            return False

        app_state.microphone_connected = True
        app_state.microphone_name = self.device_name
        logger.info(f"Audio stream using device [{self.device_index}]: {self.device_name}")

        # 3. Start audio stream & worker loop
        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.step_samples,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_running = True
            
            # Start background processing loop
            self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._worker_thread.start()

            app_state.update_state(AppState.LISTENING)
            return True
        except Exception as e:
            logger.error(f"Failed to start sounddevice stream: {e}")
            app_state.microphone_connected = False
            app_state.update_state(AppState.ERROR)
            return False

    def stop(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            self.stream = None

        self.kws_service.stop()
        if self.asr_service:
            self.asr_service.finish_stream()

        app_state.update_state(AppState.IDLE)
        logger.info("Audio Stream Manager stopped.")

    def _processing_loop(self):
        """Continuous background audio evaluation loop."""
        logger.info("Background audio processing loop started.")
        time.sleep(settings.WINDOW_SECONDS)  # Allow buffer warm-up

        asr_stream_active = False
        silence_start_time: Optional[float] = None
        asr_session_start_time: Optional[float] = None
        live_chunks_count = 0

        while self.is_running:
            time.sleep(settings.STEP_SECONDS)
            now = time.time()

            with self.buffer_lock:
                chunk = self.audio_buffer.copy()
                step_chunk = self.latest_step_chunk.copy()
                has_new_step = self.new_step_available
                self.new_step_available = False

            # Calculate actual RMS audio level
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-8))
            speech_active = rms >= self.min_rms

            # Broadcast audio activity every 150ms for smooth UI audio visualizer
            if now - self.last_audio_broadcast_time >= 0.15:
                app_state.broadcast_event("audio_activity", {
                    "rms": round(rms, 6),
                    "speech_active": speech_active
                })
                self.last_audio_broadcast_time = now

            # If ASR streaming is active, send non-overlapping linear16 PCM step audio to Deepgram
            if asr_stream_active:
                if has_new_step and len(step_chunk) > 0:
                    pcm_data, r_rms, a_rms, g_mult, pk = self._scale_audio_for_asr(step_chunk)
                    self.asr_service.send_audio(pcm_data)
                    live_chunks_count += 1
                    logger.info(
                        f"ASR live chunk #{live_chunks_count} sent: "
                        f"raw_rms={r_rms:.6f}, asr_rms={a_rms:.6f}, gain={g_mult:.2f}x, peak={pk:.4f}, bytes={len(pcm_data)}"
                    )

                # Track local silence for post-keyword inactivity timeout
                if not speech_active:
                    if silence_start_time is None:
                        silence_start_time = now
                else:
                    silence_start_time = None

                # Check completion or timeout conditions:
                # 1. Deepgram explicitly finished (speech_final / UtteranceEnd / COMPLETED)
                # 2. Deepgram reported an error
                # 3. Post-keyword inactivity timeout (10.0s after Deepgram stream becomes active)
                # 4. Max session safety cap (12.0s)
                has_transcript = bool(app_state.transcript and app_state.transcript.strip())
                deepgram_completed = app_state.asr_status in [ASRStatus.COMPLETED, ASRStatus.ERROR]
                inactivity_timeout = (not has_transcript) and (asr_session_start_time is not None) and (now - asr_session_start_time >= 10.0)
                max_session_timeout = (asr_session_start_time is not None) and (now - asr_session_start_time >= 12.0)

                asr_done = deepgram_completed or inactivity_timeout or max_session_timeout

                if asr_done:
                    logger.info(
                        f"ASR: Session finishing... (Reason: deepgram_completed={deepgram_completed}, "
                        f"live_chunks_sent={live_chunks_count}, transcript='{app_state.transcript}')"
                    )
                    final_tx = self.asr_service.finish_stream()
                    if final_tx and not app_state.transcript:
                        app_state.transcript = final_tx
                    
                    if not app_state.transcript_complete_time:
                        app_state.transcript_complete_time = now

                    app_state.update_state(AppState.COMPLETED)
                    app_state.broadcast_event("latency_update", app_state.get_latency_metrics().dict())
                    
                    asr_stream_active = False
                    silence_start_time = None
                    asr_session_start_time = None
                    live_chunks_count = 0

                    # Keep COMPLETED state & transcript on UI for 3.0s before returning to LISTENING
                    time.sleep(3.0)
                    app_state.update_state(AppState.LISTENING)

                continue

            # In LISTENING state: check speech activity
            if not speech_active:
                if app_state.state in [AppState.SPEECH_DETECTED, AppState.VERIFYING]:
                    app_state.update_state(AppState.LISTENING)
                continue

            # Speech onset timestamp
            if app_state.speech_onset_time is None or (now - app_state.speech_onset_time > 4.0):
                app_state.speech_onset_time = now

            if app_state.state == AppState.LISTENING:
                app_state.update_state(AppState.VERIFYING)

            # Process window through KWS Engine
            result = self.kws_service.process_audio(chunk)

            # Update confidence score on state
            if result.confidence > 0:
                app_state.last_confidence = result.confidence
                app_state.class_index = getattr(result, "class_index", None)

            # Handle Keyword Detection
            if result.detected and result.keyword and result.keyword.lower() != "unknown":
                app_state.keyword_detection_time = now
                app_state.detected_keyword = result.keyword
                app_state.last_confidence = result.confidence
                app_state.class_index = getattr(result, "class_index", None)
                
                # Reset ASR timestamps & transcript
                app_state.asr_first_receive_time = None
                app_state.transcript_complete_time = None
                app_state.transcript = None

                app_state.update_state(AppState.KEYWORD_DETECTED)
                
                # Broadcast detection event containing actual keyword and class_index
                app_state.broadcast_event("kws_detection", {
                    "keyword": result.keyword,
                    "confidence": round(result.confidence, 4),
                    "class_index": getattr(result, "class_index", 0),
                    "model": result.model,
                    "timestamp": result.timestamp
                })
                
                # Broadcast latency update
                app_state.broadcast_event("latency_update", app_state.get_latency_metrics().dict())

                logger.info(f"KEYWORD CONFIRMED! Keyword='{result.keyword}', Confidence={result.confidence:.4f}, ClassIndex={getattr(result, 'class_index', 0)}")

                # Start capturing ONLY post-keyword command PCM chunks while Deepgram connects
                with self.buffer_lock:
                    self.post_keyword_pcm_queue.clear()
                    self.is_capturing_post_keyword = True

                # Transition state to STREAMING
                app_state.stream_start_time = time.time()
                app_state.update_state(AppState.STREAMING)

                # Initiate ASR stream with Deepgram
                asr_started = self.asr_service.start_stream()

                # Stop post-keyword capture mode and snapshot queued chunks
                with self.buffer_lock:
                    self.is_capturing_post_keyword = False
                    post_keyword_chunks = list(self.post_keyword_pcm_queue)
                    self.post_keyword_pcm_queue.clear()

                if asr_started:
                    asr_stream_active = True
                    asr_session_start_time = time.time()
                    silence_start_time = None
                    live_chunks_count = 0
                    app_state.update_state(AppState.TRANSCRIBING)

                    # Flush ONLY post-keyword queued chunks (captured while Deepgram was connecting)
                    flushed_bytes = 0
                    for idx, item in enumerate(post_keyword_chunks, start=1):
                        pcm_bytes, r_rms, a_rms, g_mult, pk = item
                        if len(pcm_bytes) > 0:
                            self.asr_service.send_audio(pcm_bytes)
                            flushed_bytes += len(pcm_bytes)
                            logger.info(
                                f"ASR post-keyword chunk #{idx}/{len(post_keyword_chunks)} sent: "
                                f"raw_rms={r_rms:.6f}, asr_rms={a_rms:.6f}, gain={g_mult:.2f}x, peak={pk:.4f}, bytes={len(pcm_bytes)}"
                            )

                    logger.info(
                        f"ASR stream active. Pre-keyword wake word audio excluded. "
                        f"Post-keyword queued chunks={len(post_keyword_chunks)}, flushed_bytes={flushed_bytes}."
                    )
                else:
                    logger.info("ASR not configured or failed to start. Returning to LISTENING.")
                    time.sleep(1.2)
                    app_state.update_state(AppState.LISTENING)


# Global Audio Stream Manager instance
stream_manager = AudioStreamManager()
