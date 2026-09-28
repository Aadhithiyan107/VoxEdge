import os
import time
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf

# ============================================================
# VoxEdge - 6-Class Multi-Keyword KWS Live Microphone Classifier
# Continuous detection for: voxedge, jarvis, friday, edith, karen, unknown
# ============================================================

MODEL_PATH_BEST = r"D:\VoxEdge\models\voxedge_multikeyword_best.keras"
MODEL_PATH_FINAL = r"D:\VoxEdge\models\voxedge_multikeyword_final.keras"

MODEL_PATH = MODEL_PATH_BEST if os.path.exists(MODEL_PATH_BEST) else MODEL_PATH_FINAL

CLASS_NAMES = ["voxedge", "jarvis", "friday", "edith", "karen", "unknown"]
NUM_CLASSES = len(CLASS_NAMES)

SAMPLE_RATE = 16000

# 1-second model input window
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

# Evaluate every 250 ms
STEP_SECONDS = 0.25
STEP_SAMPLES = int(SAMPLE_RATE * STEP_SECONDS)

# KWS decision threshold
KWS_THRESHOLD = 0.65

# Basic activity gate
MIN_RMS = 0.003

# Multi-hit tracking
REQUIRED_HITS = 2
HIT_WINDOW_SECONDS = 1.5

# Cooldown after confirmed detection
COOLDOWN_SECONDS = 2.0

# MFCC settings matching exact VoxEdge pipeline
N_MFCC = 10
N_FFT = 480
HOP_LENGTH = 320


# ============================================================
# LOAD MULTI-KEYWORD MODEL
# ============================================================

print("=" * 65)
print("VOXEDGE MULTI-KEYWORD KWS - LIVE MICROPHONE TEST")
print("=" * 65)

print("\nLoading model:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
print(f"Supported Classes ({NUM_CLASSES}): {CLASS_NAMES}")


# ============================================================
# MICROPHONE DISCOVERY & SELECTION
# ============================================================

def get_input_devices():
    devices = sd.query_devices()
    inputs = []
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            inputs.append({
                "index": index,
                "name": device["name"],
                "samplerate": device["default_samplerate"]
            })
    return inputs


def test_microphone(device_index):
    try:
        stream = sd.InputStream(
            device=device_index,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=STEP_SAMPLES
        )
        stream.start()
        time.sleep(0.15)
        stream.stop()
        stream.close()
        return True
    except Exception:
        return False


def select_microphone():
    devices = get_input_devices()
    print("\nAvailable input devices:")
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")

    if not devices:
        raise RuntimeError("No microphone/input device found.")

    # Try system default first
    try:
        default_input = sd.default.device[0]
        if default_input is not None:
            default_name = sd.query_devices(default_input)["name"]
            print(f"\nTrying system default microphone: [{default_input}] {default_name}")
            if test_microphone(default_input):
                print("Using system default microphone.")
                return default_input
    except Exception:
        pass

    # Fallback search
    print("\nSearching for a working microphone...")
    for device in devices:
        if test_microphone(device["index"]):
            print(f"Using working microphone: [{device['index']}] {device['name']}")
            return device["index"]

    raise RuntimeError("No working microphone could be opened.")


# ============================================================
# ROBUST AUDIO PREPROCESSING (PRESERVED)
# ============================================================

def normalize_audio(audio):
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
            activity_threshold = max(MIN_RMS, peak_rms * 0.20)
            active = np.where(rms_frames >= activity_threshold)[0]

            if len(active) > 0:
                start_frame = active[0]
                end_frame = active[-1]

                start_sample = start_frame * hop_length
                end_sample = end_frame * hop_length + frame_length

                context = int(0.10 * SAMPLE_RATE)
                start_sample = max(0, start_sample - context)
                end_sample = min(len(audio), end_sample + context)

                active_audio = audio[start_sample:end_sample]

                if len(active_audio) > 0:
                    normalized = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
                    if len(active_audio) >= WINDOW_SAMPLES:
                        active_audio = active_audio[:WINDOW_SAMPLES]
                        normalized[:] = active_audio
                    else:
                        start = (WINDOW_SAMPLES - len(active_audio)) // 2
                        normalized[start:start + len(active_audio)] = active_audio
                    audio = normalized

    rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    if rms > 1e-6:
        audio = audio * (0.08 / rms)

    audio = np.clip(audio, -1.0, 1.0)
    return audio


def preprocess_audio(audio):
    audio = normalize_audio(audio)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc = mfcc.astype(np.float32)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc


# ============================================================
# STREAM AUDIO BUFFER
# ============================================================

audio_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)


def audio_callback(indata, frames, callback_time, status):
    global audio_buffer
    if status:
        print(f"\nAudio stream status: {status}")

    samples = indata[:, 0].astype(np.float32)

    if len(samples) >= WINDOW_SAMPLES:
        audio_buffer = samples[-WINDOW_SAMPLES:]
    else:
        audio_buffer = np.concatenate((audio_buffer[len(samples):], samples))


# ============================================================
# INITIALIZE MICROPHONE & READY
# ============================================================

try:
    microphone = select_microphone()
except Exception as e:
    print(f"\nERROR selecting microphone: {e}")
    raise SystemExit(1)

print("\n" + "=" * 65)
print("READY FOR LIVE MULTI-KEYWORD DETECTION")
print("=" * 65)

print("\nSupported Keywords:")
print("  • VoxEdge")
print("  • Jarvis")
print("  • Friday")
print("  • Edith")
print("  • Karen")
print("  • Unknown (background noise / silence / general speech)")

print("\nPress Ctrl+C to stop.\n" + "=" * 65)

# ============================================================
# CONTINUOUS LIVE EVALUATION LOOP
# ============================================================

recent_hits = []
last_display_time = 0.0
cooldown_until = 0.0

try:
    with sd.InputStream(
        device=microphone,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=STEP_SAMPLES,
        callback=audio_callback
    ):
        print("\n🎤 Listening continuously...\n")
        time.sleep(WINDOW_SECONDS)

        while True:
            time.sleep(STEP_SECONDS)
            now = time.time()

            if now < cooldown_until:
                continue

            audio = audio_buffer.copy()

            # RMS Audio Activity Check
            raw_rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
            if raw_rms < MIN_RMS:
                recent_hits.clear()
                if now - last_display_time >= 0.5:
                    print("\r[Listening...] Silence / Ambient Noise                        ", end="", flush=True)
                    last_display_time = now
                continue

            # Model Inference
            features = preprocess_audio(audio)
            prediction = model.predict(features, verbose=0)
            probs = prediction[0]  # Array of 6 float probabilities

            top_idx = int(np.argmax(probs))
            top_class = CLASS_NAMES[top_idx]
            top_prob = float(probs[top_idx])

            # Hit tracking for keyword classes (0 to 4)
            if top_idx in [0, 1, 2, 3, 4] and top_prob >= KWS_THRESHOLD:
                recent_hits.append((now, top_idx))

            recent_hits = [h for h in recent_hits if now - h[0] <= HIT_WINDOW_SECONDS]

            # Count hits per class
            hit_counts = {}
            for _, c_idx in recent_hits:
                hit_counts[c_idx] = hit_counts.get(c_idx, 0) + 1

            confirmed_class_idx = None
            for c_idx, count in hit_counts.items():
                if count >= REQUIRED_HITS:
                    confirmed_class_idx = c_idx
                    break

            # Format 6-class probabilities for continuous output
            # Format: voxedge:XX% | jarvis:XX% | friday:XX% | edith:XX% | karen:XX% | unknown:XX%
            prob_str = " | ".join([f"{CLASS_NAMES[i][:6]}:{probs[i]*100:5.1f}%" for i in range(6)])

            if now - last_display_time >= 0.25:
                print(f"\r[{prob_str}] -> DETECTED: {top_class:<7s} ({top_prob*100:5.1f}%)   ", end="", flush=True)
                last_display_time = now

            # Confirmed Trigger Output
            if confirmed_class_idx is not None:
                detected_keyword = CLASS_NAMES[confirmed_class_idx]
                confidence = probs[confirmed_class_idx]

                print("\n\n" + "=" * 65)
                print(f"✅ CONFIRMED KEYWORD DETECTED: {detected_keyword.upper()}")
                print("=" * 65)
                print(f"  • Keyword         : {detected_keyword}")
                print(f"  • Confidence      : {confidence * 100:.2f}%")
                print(f"  • Class Index     : {confirmed_class_idx}")
                print(f"  • All Probabilities: {prob_str}")
                print("=" * 65)

                recent_hits.clear()
                cooldown_until = time.time() + COOLDOWN_SECONDS
                print("\n🎤 Listening again...\n")

except KeyboardInterrupt:
    print("\n\nStopped by user.")
except Exception as e:
    print(f"\n\nERROR: {e}")
finally:
    print("\nVoxEdge multi-keyword live test ended.")