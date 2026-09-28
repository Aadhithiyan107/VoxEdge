import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, 
  Activity, 
  Zap, 
  Layers, 
  CheckCircle2, 
  Play, 
  Square,
  Clock,
  Radio,
  Server,
  Sparkles
} from 'lucide-react';

export default function App() {
  const [systemStatus, setSystemStatus] = useState({
    state: 'IDLE',
    microphone_connected: false,
    microphone_name: null,
    kws_status: 'Multi-Keyword KWS',
    active_model: 'voxedge_multikeyword',
    asr_status: 'NOT_CONFIGURED',
    asr_message: 'ASR provider not configured',
    transcript: null,
    detected_keyword: null,
    last_confidence: 0,
    class_index: null,
    latency: {
      keyword_detection_latency_ms: null,
      keyword_end_to_asr_latency_ms: null,
      asr_first_result_latency_ms: null,
      total_latency_ms: null
    }
  });

  const [supportedKeywords, setSupportedKeywords] = useState([
    { keyword: "VoxEdge", status: "Active", validated: true },
    { keyword: "Jarvis", status: "Active", validated: true },
    { keyword: "Friday", status: "Active", validated: true },
    { keyword: "EDITH", status: "Active", validated: true },
    { keyword: "KAREN", status: "Active", validated: true }
  ]);

  const [audioRms, setAudioRms] = useState(0);
  const [speechActive, setSpeechActive] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [detectionHighlight, setDetectionHighlight] = useState(false);

  const wsRef = useRef(null);

  // Fetch initial REST config & status
  const fetchInitialStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
      const cfgRes = await fetch('/api/config');
      if (cfgRes.ok) {
        const cfgData = await cfgRes.json();
        if (cfgData.supported_keywords) {
          setSupportedKeywords(cfgData.supported_keywords);
        }
      }
    } catch (e) {
      console.warn("Initial status fetch error:", e);
    }
  };

  // Connect WebSocket
  useEffect(() => {
    fetchInitialStatus();

    const connectWS = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/events`;
      
      console.log("Connecting WebSocket to:", wsUrl);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected");
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const { type, data } = payload;

          if (type === 'system_status') {
            setSystemStatus(data);
          } else if (type === 'state_change') {
            setSystemStatus(prev => ({ ...prev, state: data.state }));
          } else if (type === 'audio_activity') {
            setAudioRms(data.rms);
            setSpeechActive(data.speech_active);
          } else if (type === 'kws_detection') {
            setSystemStatus(prev => ({
              ...prev,
              detected_keyword: data.keyword,
              last_confidence: data.confidence,
              class_index: data.class_index,
              state: 'KEYWORD_DETECTED'
            }));
            setDetectionHighlight(true);
            setTimeout(() => setDetectionHighlight(false), 3000);
          } else if (type === 'latency_update') {
            setSystemStatus(prev => ({
              ...prev,
              latency: {
                keyword_detection_latency_ms: data.keyword_detection_latency_ms,
                keyword_end_to_asr_latency_ms: data.keyword_end_to_asr_latency_ms,
                asr_first_result_latency_ms: data.asr_first_result_latency_ms,
                total_latency_ms: data.total_latency_ms
              }
            }));
          } else if (type === 'asr_update') {
            setSystemStatus(prev => ({
              ...prev,
              asr_status: data.status,
              asr_message: data.message,
              transcript: data.transcript !== undefined ? data.transcript : prev.transcript
            }));
          }
        } catch (err) {
          console.error("WS Parse error:", err);
        }
      };

      ws.onclose = () => {
        console.log("WebSocket disconnected. Retrying in 2s...");
        setWsConnected(false);
        setTimeout(connectWS, 2000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
      };
    };

    connectWS();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleStartSession = async () => {
    try {
      await fetch('/api/session/start', { method: 'POST' });
    } catch (e) {
      console.error("Error starting session:", e);
    }
  };

  const handleStopSession = async () => {
    try {
      await fetch('/api/session/stop', { method: 'POST' });
    } catch (e) {
      console.error("Error stopping session:", e);
    }
  };

  // State badge styling
  const getStatePillClass = () => {
    switch (systemStatus.state) {
      case 'LISTENING': return 'listening';
      case 'KEYWORD_DETECTED': return 'detected';
      case 'STREAMING':
      case 'TRANSCRIBING': return 'listening';
      case 'SPEECH_DETECTED':
      case 'VERIFYING': return 'verifying';
      case 'ERROR': return 'error';
      default: return 'idle';
    }
  };

  const getAsrBadgeStyle = () => {
    switch (systemStatus.asr_status) {
      case 'TRANSCRIBING':
      case 'STREAMING':
        return { background: 'rgba(56, 189, 248, 0.2)', color: '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.4)' };
      case 'COMPLETED':
        return { background: 'rgba(16, 185, 129, 0.2)', color: '#10b981', border: '1px solid rgba(16, 185, 129, 0.4)' };
      case 'READY':
        return { background: 'rgba(129, 140, 248, 0.2)', color: '#818cf8', border: '1px solid rgba(129, 140, 248, 0.4)' };
      case 'ERROR':
        return { background: 'rgba(244, 63, 94, 0.2)', color: '#f43f5e', border: '1px solid rgba(244, 63, 94, 0.4)' };
      default:
        return { background: 'rgba(148, 163, 184, 0.15)', color: '#94a3b8', border: '1px solid rgba(148, 163, 184, 0.3)' };
    }
  };

  // Audio level bar heights calculation (14 bars)
  const barHeights = Array.from({ length: 14 }).map((_, i) => {
    if (!speechActive || audioRms <= 0) return 6;
    const normRms = Math.min(audioRms * 120, 1.0);
    const varFactor = Math.sin((i + 1) * 0.7) * 0.3 + 0.7;
    return Math.max(6, Math.floor(normRms * varFactor * 65));
  });

  return (
    <div className="app-container">
      {/* HEADER */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo">
            <Zap />
          </div>
          <div>
            <h1 className="brand-title">VoxEdge</h1>
            <p className="brand-subtitle">Low-Latency & Efficient Voice Activator for Edge Devices</p>
          </div>
        </div>

        <div className="header-controls">
          <div className="status-pill">
            <span className={`status-dot ${getStatePillClass()}`}></span>
            <span>{systemStatus.state.replace('_', ' ')}</span>
          </div>

          {systemStatus.state === 'IDLE' ? (
            <button className="btn-primary" onClick={handleStartSession}>
              <Play size={16} /> Start Session
            </button>
          ) : (
            <button className="btn-danger" onClick={handleStopSession}>
              <Square size={16} /> Stop Session
            </button>
          )}
        </div>
      </header>

      {/* DASHBOARD GRID */}
      <div className="dashboard-grid">
        
        {/* MAIN KWS CARD (8 cols) */}
        <div className="card col-span-8">
          <div className="card-header">
            <div className="card-title">
              <Mic size={20} />
              <span>Multi-Keyword Wake Word Detector</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--text-secondary)' }}>
              <Radio size={14} color={systemStatus.microphone_connected ? '#10b981' : '#64748b'} />
              <span>{systemStatus.microphone_name || 'Searching Microphone...'}</span>
            </div>
          </div>

          <div className="kws-display">
            {/* HERO WAKE WORD BOX */}
            <div className={`wake-word-hero ${detectionHighlight ? 'detected' : ''}`}>
              <div className="target-label">Detected Wake Word</div>
              <div className={`keyword-value ${detectionHighlight ? 'highlight' : ''}`}>
                {systemStatus.detected_keyword || "Listening..."}
              </div>
              {detectionHighlight && (
                <div style={{ marginTop: '8px', color: 'var(--color-emerald)', fontSize: '14px', fontWeight: '600', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                  <CheckCircle2 size={16} /> KEYWORD CONFIRMED: {systemStatus.detected_keyword?.toUpperCase()}
                </div>
              )}
            </div>

            {/* CONFIDENCE BAR */}
            <div className="confidence-container">
              <div className="confidence-header">
                <span style={{ color: 'var(--text-secondary)' }}>Model Confidence Level</span>
                <span className="confidence-score">
                  {(systemStatus.last_confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="confidence-track">
                <div 
                  className="threshold-marker" 
                  style={{ left: '65%' }} 
                  title="KWS Threshold (65%)"
                />
                <div 
                  className="confidence-fill" 
                  style={{ width: `${Math.min(systemStatus.last_confidence * 100, 100)}%` }} 
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                <span>0%</span>
                <span style={{ color: 'var(--color-amber)' }}>Gate Threshold (65%)</span>
                <span>100%</span>
              </div>
            </div>

            {/* LIVE AUDIO WAVEFORM */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Activity size={14} /> Microphone Stream VAD Activity
                </span>
                <span>
                  RMS: {audioRms.toFixed(5)} {speechActive ? '(SPEECH ACTIVE)' : '(SILENCE/NOISE)'}
                </span>
              </div>
              <div className="waveform-box">
                {barHeights.map((h, idx) => (
                  <div 
                    key={idx}
                    className={`audio-bar ${detectionHighlight ? 'detected' : speechActive ? 'active' : ''}`}
                    style={{ height: `${h}px` }}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* SYSTEM STATUS CARD (4 cols) */}
        <div className="card col-span-4">
          <div className="card-header">
            <div className="card-title">
              <Server size={20} />
              <span>System Health</span>
            </div>
          </div>

          <div className="sys-list">
            <div className="sys-item">
              <span className="sys-key">Microphone</span>
              <span className="sys-val" style={{ color: systemStatus.microphone_connected ? 'var(--color-emerald)' : 'var(--color-rose)' }}>
                {systemStatus.microphone_connected ? 'Connected' : 'Not Available'}
              </span>
            </div>

            <div className="sys-item">
              <span className="sys-key">KWS Engine</span>
              <span className="sys-val" style={{ color: 'var(--color-cyan)' }}>
                {systemStatus.kws_status || 'Multi-Keyword KWS'}
              </span>
            </div>

            <div className="sys-item">
              <span className="sys-key">Model</span>
              <span className="sys-val">
                {systemStatus.active_model || 'voxedge_multikeyword'}
              </span>
            </div>

            <div className="sys-item">
              <span className="sys-key">ASR Provider</span>
              <span className="sys-val" style={{ color: systemStatus.asr_status !== 'NOT_CONFIGURED' ? 'var(--color-emerald)' : 'var(--text-muted)' }}>
                {systemStatus.asr_status !== 'NOT_CONFIGURED' ? 'Deepgram (nova-3)' : 'Not Configured'}
              </span>
            </div>

            <div className="sys-item">
              <span className="sys-key">Backend WebSocket</span>
              <span className="sys-val" style={{ color: wsConnected ? 'var(--color-emerald)' : 'var(--color-rose)' }}>
                {wsConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>

        {/* ASR INTEGRATION CARD (6 cols) */}
        <div className="card col-span-6">
          <div className="card-header">
            <div className="card-title">
              <Layers size={20} />
              <span>Deepgram Streaming ASR</span>
            </div>
            <span className="badge-unconfigured" style={getAsrBadgeStyle()}>
              {systemStatus.asr_status}
            </span>
          </div>

          <div className="asr-content">
            <div className="asr-status-row">
              <span style={{ color: 'var(--text-secondary)' }}>Status Message:</span>
              <span style={{ color: 'var(--text-muted)' }}>{systemStatus.asr_message}</span>
            </div>

            <div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Speech Transcript Output
              </div>
              <div className="transcript-box" style={{ color: systemStatus.transcript ? '#f8fafc' : 'var(--text-muted)', fontStyle: systemStatus.transcript ? 'normal' : 'italic', fontWeight: systemStatus.transcript ? '500' : '400' }}>
                {systemStatus.transcript ? `"${systemStatus.transcript}"` : '"No transcript yet"'}
              </div>
            </div>
          </div>
        </div>

        {/* LATENCY METRICS CARD (6 cols) */}
        <div className="card col-span-6">
          <div className="card-header">
            <div className="card-title">
              <Clock size={20} />
              <span>Latency Metrics</span>
            </div>
          </div>

          <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
            <div className="metric-item">
              <div className="metric-label">KWS Detection</div>
              <div className={`metric-value ${systemStatus.latency.keyword_detection_latency_ms ? '' : 'unavailable'}`}>
                {systemStatus.latency.keyword_detection_latency_ms != null
                  ? `${systemStatus.latency.keyword_detection_latency_ms} ms`
                  : '--'}
              </div>
            </div>

            <div className="metric-item">
              <div className="metric-label">KWS → ASR Hand-off</div>
              <div className={`metric-value ${systemStatus.latency.keyword_end_to_asr_latency_ms ? '' : 'unavailable'}`}>
                {systemStatus.latency.keyword_end_to_asr_latency_ms != null
                  ? `${systemStatus.latency.keyword_end_to_asr_latency_ms} ms`
                  : '--'}
              </div>
            </div>

            <div className="metric-item">
              <div className="metric-label">ASR First Result</div>
              <div className={`metric-value ${systemStatus.latency.asr_first_result_latency_ms ? '' : 'unavailable'}`}>
                {systemStatus.latency.asr_first_result_latency_ms != null
                  ? `${systemStatus.latency.asr_first_result_latency_ms} ms`
                  : '--'}
              </div>
            </div>

            <div className="metric-item">
              <div className="metric-label">Total End-to-End</div>
              <div className={`metric-value ${systemStatus.latency.total_latency_ms ? '' : 'unavailable'}`}>
                {systemStatus.latency.total_latency_ms != null
                  ? `${systemStatus.latency.total_latency_ms} ms`
                  : '--'}
              </div>
            </div>
          </div>
        </div>

        {/* SUPPORTED WAKE WORDS TAXONOMY CARD (12 cols) */}
        <div className="card col-span-12">
          <div className="card-header">
            <div className="card-title">
              <Sparkles size={20} />
              <span>Supported Wake Words</span>
            </div>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Multi-Keyword DS-CNN Classifier Active (6 Output Classes)
            </span>
          </div>

          <div className="keywords-grid">
            {supportedKeywords.map((item, idx) => (
              <div key={idx} className={`keyword-chip ${item.validated ? 'active' : ''}`}>
                <span className="chip-name">{item.keyword}</span>
                <span className={`chip-status ${item.validated ? 'active' : 'planned'}`}>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
