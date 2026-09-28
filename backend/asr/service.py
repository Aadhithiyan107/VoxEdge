"""
Deepgram Live Streaming ASR Service Implementation
Supports real-time WebSocket audio streaming to Deepgram's Speech-to-Text API (nova-3).
"""
import asyncio
import json
import logging
import threading
import time
import urllib.parse
from typing import Dict, Any, Optional
import numpy as np

import websockets
try:
    from websockets.protocol import State as WSState
except ImportError:
    WSState = None

from backend.asr.interface import ASRInterface
from backend.core.config import settings
from backend.core.state import app_state
from backend.schemas.events import ASRStatus, AppState

logger = logging.getLogger("voxedge.asr")


class DeepgramASRService(ASRInterface):
    """
    Production Deepgram Live Streaming Speech-to-Text Integration.
    Uses official WebSocket protocol with persistent connection per ASR session.
    """

    def __init__(self):
        self.api_key = settings.DEEPGRAM_API_KEY
        if self.api_key:
            self.status = ASRStatus.READY
            self.message = "Deepgram ASR ready (nova-3)."
            app_state.asr_status = ASRStatus.READY
            app_state.asr_message = self.message
        else:
            self.status = ASRStatus.NOT_CONFIGURED
            self.message = "DEEPGRAM_API_KEY missing. System running KWS only."
            app_state.asr_status = ASRStatus.NOT_CONFIGURED
            app_state.asr_message = self.message

        self.ws = None
        self.is_streaming = False
        self.latest_transcript: Optional[str] = None
        
        # Diagnostic audio transmission counters
        self.chunks_sent = 0
        self.total_bytes_sent = 0
        self.last_log_time = 0.0

        # Async loop and background thread management
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._stop_event = threading.Event()

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "configured": bool(self.api_key),
            "provider": "Deepgram",
            "model": settings.DEEPGRAM_MODEL,
            "message": self.message,
            "transcript": self.latest_transcript,
            "chunks_sent": self.chunks_sent,
            "total_bytes_sent": self.total_bytes_sent
        }

    def start_stream(self) -> bool:
        """
        Starts a persistent Deepgram WebSocket streaming session.
        """
        self.api_key = settings.DEEPGRAM_API_KEY
        if not self.api_key:
            logger.warning("ASR start_stream called, but DEEPGRAM_API_KEY is missing.")
            self.status = ASRStatus.NOT_CONFIGURED
            self.message = "DEEPGRAM_API_KEY missing. Running in local KWS mode."
            app_state.asr_status = ASRStatus.NOT_CONFIGURED
            app_state.asr_message = self.message
            return False

        # If previous session thread is shutting down, wait for join
        if self._thread and self._thread.is_alive():
            self.finish_stream()
            if self._thread:
                self._thread.join(timeout=1.5)

        logger.info("ASR: Initiating Deepgram streaming session...")
        self._stop_event.clear()
        self.latest_transcript = None
        self.chunks_sent = 0
        self.total_bytes_sent = 0
        self.last_log_time = 0.0
        app_state.transcript = None
        
        # Start background asyncio event loop thread for WebSocket connection
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        
        # Wait up to 3.5s for WebSocket connection establishment
        start_wait = time.time()
        while not self.is_streaming and time.time() - start_wait < 3.5:
            if self.status == ASRStatus.ERROR:
                break
            time.sleep(0.05)

        return self.is_streaming

    def _build_url(self, endpoint: str) -> str:
        params = {
            "model": settings.DEEPGRAM_MODEL,
            "language": settings.DEEPGRAM_LANGUAGE,
            "encoding": "linear16",
            "sample_rate": settings.SAMPLE_RATE,
            "channels": 1,
            "smart_format": "true",
            "interim_results": "true",
            "vad_events": "true",
            "endpointing": settings.DEEPGRAM_ENDPOINTING,
        }
        return f"{endpoint}?{urllib.parse.urlencode(params)}"

    def _run_async_loop(self):
        """Dedicated background event loop for handling Deepgram WebSocket async tasks."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # Create queue inside the new event loop before connecting
        self._audio_queue = asyncio.Queue()
        try:
            self._loop.run_until_complete(self._websocket_task())
        except Exception as e:
            logger.error(f"ASR: Event loop error: {e}")
        finally:
            # Clean up pending tasks before closing event loop
            try:
                pending = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logger.debug(f"ASR: Pending task cleanup: {e}")

            try:
                self._loop.close()
            except Exception as e:
                logger.warning(f"ASR: Error closing loop: {e}")
            self._loop = None
            self._audio_queue = None

    async def _websocket_task(self):
        endpoints = [
            settings.DEEPGRAM_PRIMARY_ENDPOINT,
            settings.DEEPGRAM_FALLBACK_ENDPOINT,
        ]
        
        headers = {
            "Authorization": f"Token {self.api_key}"
        }

        connected = False
        for endpoint in endpoints:
            url = self._build_url(endpoint)
            logger.info(f"ASR: Connecting to Deepgram endpoint: {endpoint}...")
            try:
                async with websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=10,
                    ping_timeout=10,
                    open_timeout=5.0
                ) as ws:
                    self.ws = ws
                    connected = True
                    self.is_streaming = True
                    self.status = ASRStatus.STREAMING
                    self.message = "Connected to Deepgram ASR."
                    app_state.asr_status = ASRStatus.STREAMING
                    app_state.asr_message = self.message
                    
                    logger.info("ASR: Deepgram WebSocket connected successfully.")
                    app_state.broadcast_event("asr_update", self.get_status())

                    # Run sender and receiver concurrently
                    receiver_task = asyncio.create_task(self._receiver_loop())
                    sender_task = asyncio.create_task(self._sender_loop())
                    
                    await asyncio.gather(receiver_task, sender_task, return_exceptions=True)
                    break
            except Exception as e:
                logger.warning(f"ASR: Connection to {endpoint} failed: {e}")
                continue

        if not connected:
            logger.error("ASR: Failed to connect to Deepgram (primary and fallback endpoints failed).")
            self.status = ASRStatus.ERROR
            self.message = "ASR connection failed."
            app_state.asr_status = ASRStatus.ERROR
            app_state.asr_message = self.message
            app_state.broadcast_event("asr_update", self.get_status())
            self.is_streaming = False

    def _is_ws_open(self) -> bool:
        if self.ws is None:
            return False
        if WSState and hasattr(self.ws, "state"):
            return self.ws.state == WSState.OPEN
        return True

    async def _sender_loop(self):
        """Asynchronously pulls audio bytes from queue and sends to Deepgram WebSocket."""
        try:
            while not self._stop_event.is_set():
                if self._audio_queue is None:
                    await asyncio.sleep(0.02)
                    continue
                try:
                    chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                    if chunk is None:  # Shutdown signal
                        break
                    if self._is_ws_open() and len(chunk) > 0:
                        await self.ws.send(chunk)
                        self.chunks_sent += 1
                        self.total_bytes_sent += len(chunk)

                        # Throttled diagnostic logging (Task 1 & Task 6)
                        now = time.time()
                        if now - self.last_log_time >= 1.0 or self.chunks_sent == 1:
                            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                            chunk_rms = float(np.sqrt(np.mean(samples ** 2) + 1e-8)) if len(samples) > 0 else 0.0
                            logger.info(
                                f"ASR audio: chunks={self.chunks_sent}, bytes={self.total_bytes_sent}, "
                                f"chunk_bytes={len(chunk)}, rms={chunk_rms:.4f}"
                            )
                            self.last_log_time = now

                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    logger.info("ASR: WebSocket closed while sending audio.")
                    break
        except Exception as e:
            logger.error(f"ASR: Error sending audio to Deepgram: {e}")
        finally:
            if self.ws and self._is_ws_open():
                try:
                    await self.ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass

    async def _receiver_loop(self):
        """Asynchronously receives and processes transcript payloads from Deepgram."""
        try:
            async for message in self.ws:
                if self._stop_event.is_set():
                    break
                    
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "").strip()
                        confidence = alternatives[0].get("confidence", 0.0)
                        is_final = data.get("is_final", False)
                        speech_final = data.get("speech_final", False)
                        
                        if transcript:
                            now = time.time()
                            
                            # First ASR result timestamp
                            if app_state.asr_first_receive_time is None:
                                app_state.asr_first_receive_time = now
                                handoff_latency = round((now - (app_state.stream_start_time or now)) * 1000, 2)
                                logger.info(f"ASR: First transcript received in {handoff_latency}ms! Transcript='{transcript}'")

                            self.latest_transcript = transcript
                            app_state.transcript = transcript
                            
                            if is_final or speech_final:
                                app_state.transcript_complete_time = now
                                logger.info(f"ASR: Final Transcript: '{transcript}' (Confidence: {confidence:.2%})")
                                self.status = ASRStatus.COMPLETED
                                app_state.asr_status = ASRStatus.COMPLETED
                                app_state.update_state(AppState.COMPLETED)
                            else:
                                logger.info(f"ASR: Interim Transcript: '{transcript}'")
                                self.status = ASRStatus.TRANSCRIBING
                                app_state.asr_status = ASRStatus.TRANSCRIBING
                                app_state.update_state(AppState.TRANSCRIBING)

                            # Broadcast transcript update to frontend UI
                            app_state.broadcast_event("asr_update", {
                                "status": self.status.value,
                                "message": "Receiving streaming transcript...",
                                "transcript": transcript,
                                "is_final": is_final or speech_final,
                                "confidence": round(confidence, 4)
                            })
                            
                            # Broadcast latency update
                            app_state.broadcast_event("latency_update", app_state.get_latency_metrics().dict())

                elif msg_type == "SpeechStarted":
                    logger.info("ASR: Deepgram reported SpeechStarted event!")

                elif msg_type == "UtteranceEnd":
                    logger.info("ASR: Deepgram reported UtteranceEnd event.")
                    app_state.transcript_complete_time = time.time()
                    self.status = ASRStatus.COMPLETED
                    app_state.asr_status = ASRStatus.COMPLETED
                    app_state.update_state(AppState.COMPLETED)
                    app_state.broadcast_event("latency_update", app_state.get_latency_metrics().dict())
                    self._stop_event.set()
                    break

        except websockets.ConnectionClosed:
            logger.info("ASR: Deepgram WebSocket connection closed cleanly.")
        except Exception as e:
            logger.error(f"ASR: Error receiving Deepgram message: {e}")
        finally:
            self.is_streaming = False

    def send_audio(self, audio_bytes: bytes) -> bool:
        """
        Thread-safe method to push raw PCM audio bytes to the streaming queue.
        """
        if self._audio_queue is None or self._loop is None:
            return False
            
        try:
            self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, audio_bytes)
            return True
        except Exception as e:
            logger.error(f"ASR: Error queuing audio bytes: {e}")
            return False

    def get_transcript(self) -> Optional[str]:
        return self.latest_transcript

    def finish_stream(self) -> Optional[str]:
        """
        Finalizes active ASR streaming session cleanly.
        """
        logger.info(f"ASR: Finalizing Deepgram stream... (Total chunks sent: {self.chunks_sent}, Total bytes: {self.total_bytes_sent})")
        self._stop_event.set()
        
        if self._audio_queue and self._loop:
            try:
                self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, None)
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        self.is_streaming = False
        return self.latest_transcript
