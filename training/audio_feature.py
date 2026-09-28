import pathlib
import numpy as np
import librosa


# Audio configuration
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# MFCC configuration
N_MFCC = 10
N_FFT = 480
HOP_LENGTH = 320


def load_audio(file_path):
    """
    Load audio as mono 16 kHz and make it exactly 1 second.
    """

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Make exactly 1 second
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )
    else:
        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


def extract_mfcc(file_path):
    """
    Convert WAV audio into MFCC features.
    """

    audio = load_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Normalize features
    mfcc = (mfcc - np.mean(mfcc)) / (
        np.std(mfcc) + 1e-8
    )

    return mfcc.astype(np.float32)


if __name__ == "__main__":

    dataset_path = pathlib.Path(
        r"D:\VoxEdge\data\downloads"
        r"\mini_speech_commands_extracted"
        r"\mini_speech_commands"
    )

    # Automatically find a real WAV file
    wav_files = list(
        (dataset_path / "yes").glob("*.wav")
    )

    if not wav_files:
        raise FileNotFoundError(
            "No WAV files found in the yes folder."
        )

    test_file = wav_files[0]

    print("Testing audio preprocessing...")
    print("File:", test_file)

    features = extract_mfcc(test_file)

    print("\nPreprocessing successful!")
    print("Feature shape:", features.shape)
    print("Feature dtype:", features.dtype)
    print("Minimum:", features.min())
    print("Maximum:", features.max())