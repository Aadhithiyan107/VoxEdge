# VoxEdge
> **SIH 2026 Project**: Low Latency and Efficient Voice Activator for Edge Devices

VoxEdge is an end-to-end, high-efficiency continuous Keyword Spotting (KWS) and Voice Activation framework designed for edge computing hardware. It features device-independent microphone capture, sliding-window audio feature extraction, speech onset Voice Activity Detection (VAD), and local neural keyword verification.

---

## 🏛️ Application Architecture

```
                    MICROPHONE
                        |
                        v
                       VAD (RMS Activity Gate)
                        |
                        v
                 LOCAL KWS ENGINE (DS-CNN Model)
                        |
              +---------+---------+
              |                   |
          No keyword          Keyword detected
              |                   |
              |                   v
              |             AUDIO CAPTURE
              |                   |
              |                   v
              |             STREAM MANAGER
              |                   |
              |                   v
              |             REMOTE ASR (Interface)
              |                   |
              |                   v
              |              TRANSCRIPT
              |                   |
              +---------+---------+
                        |
                        v
                 FASTAPI BACKEND (WebSocket / REST)
                        |
                        v
                DASHBOARD FRONTEND UI
```

The system architecture clearly decouples the 7 core components:
1. **Audio Input**: Continuous device-independent audio capture via `sounddevice`.
2. **VAD**: RMS energy gate & active speech slice isolation.
3. **Local KWS Engine**: Custom DS-CNN model evaluated over 1.0s sliding windows every 250ms with hit confirmation.
4. **Audio Stream Manager**: Coordinates buffer windowing, state transitions, and post-keyword audio dispatch.
5. **Remote ASR Interface**: Generic pluggable interface for streaming ASR (Whisper, Vosk, Deepgram, Google Cloud Speech).
6. **Central Application State**: Thread-safe central state machine (`IDLE`, `LISTENING`, `SPEECH_DETECTED`, `VERIFYING`, `KEYWORD_DETECTED`, `STREAMING`, `TRANSCRIBING`, `COMPLETED`, `ERROR`).
7. **Frontend UI**: React + Vite dark technical dashboard connected over WebSocket (`/ws/events`).

---

## 🔮 Future-Proof Multi-Keyword Ready Contract

The backend and frontend consume a standardized generic KWS result schema:

```json
{
    "detected": true,
    "keyword": "VoxEdge",
    "confidence": 0.842,
    "timestamp": "2026-09-28T21:15:52Z",
    "model": "voxedge_binary"
}
```

Neither backend nor frontend logic hardcodes binary-only assumptions. If a multi-keyword model (e.g. supporting `Jarvis`, `Friday`, `EDITH`, `KAREN`) is loaded in the future, the exact same contract will be returned without requiring any redesign of the UI or API layers.

### Multi-Keyword Taxonomy Status
- **VoxEdge** — Active & Validated (`models/voxedge_binary_best.keras`)
- **Jarvis** — Planned
- **Friday** — Planned
- **EDITH** — Planned
- **KAREN** — Planned

---

## 📁 Repository Structure

```
VoxEdge/
├── backend/
│   ├── api/             # REST endpoints (/health, /status, /config, /session/start, /session/stop)
│   ├── asr/             # ASR Interface & Unconfigured ASR Service implementation
│   ├── audio/           # Continuous Audio Stream Manager & sounddevice integration
│   ├── core/            # Configuration & Application State Manager
│   ├── kws/             # Generic KWS Interface & VoxEdge DS-CNN Binary KWS Engine
│   ├── schemas/         # Pydantic schemas and WebSocket event data models
│   ├── main.py          # FastAPI server entry point with WebSocket (/ws/events)
│   └── requirements.txt # Python backend dependencies
├── frontend/
│   ├── src/             # React App, Dashboard UI components, index.css styling
│   ├── index.html       # Vite HTML entry
│   ├── package.json     # Frontend dependencies (React, Vite, Lucide-React)
│   └── vite.config.js   # Vite server and API/WebSocket proxy config
├── models/
│   └── voxedge_binary_best.keras # Validated KWS neural network model
├── training/
│   └── live_test.py     # Working standalone live testing script
└── README.md
```

---

## 🚀 Quick Start & How to Run

### 1. Backend Setup

```bash
# Navigate to backend directory or project root
# Activate virtual environment
.venv\Scripts\activate

# Install backend dependencies (if not already installed)
pip install -r backend/requirements.txt

# Run FastAPI backend server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The backend server will run on `http://localhost:8000`.

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install node dependencies
npm install

# Start Vite development server
npm run dev
```

The frontend dashboard will run on `http://localhost:3000`.

---

## 🌐 API Endpoints

- `GET /api/health` - Health check status
- `GET /api/status` - Full system status payload (microphones, state, latencies)
- `GET /api/config` - Application parameters & keyword taxonomy
- `POST /api/session/start` - Start audio capture & continuous KWS evaluation
- `POST /api/session/stop` - Stop audio capture session
- `WebSocket /ws/events` - Real-time event stream (`state_change`, `audio_activity`, `kws_detection`, `latency_update`, `asr_update`)

---

## 📊 Latency Metrics & Verification Policy

VoxEdge measures actual timestamps:
- `speech_onset_time`
- `keyword_detection_time`
- `stream_start_time`
- `asr_first_receive_time`
- `transcript_complete_time`

Calculated latencies are displayed only when valid timestamps exist. If a component (such as an external remote ASR provider) is not currently configured, the application accurately displays `ASR_NOT_CONFIGURED` and `--` for unmeasured metrics rather than fabricating fake values.

---

## 🔒 Untouched Core Components

The validated core KWS model (`models/voxedge_binary_best.keras`) and training scripts (`training/live_test.py`) remain completely untouched and preserved.
