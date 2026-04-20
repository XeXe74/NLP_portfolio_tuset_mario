import librosa
import numpy as np
from local_stt import record_audio

def analyze_voice_sentiment(audio_path: str) -> str:
    """
    Analyzes the emotional tone of a voice recording based on acoustic features (energy, pitch, tempo).
    """

    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract acoustic features
    energy = np.mean(librosa.feature.rms(y=y))
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.mean(tempo))

    print(f"  Energy: {energy:.4f} | Pitch: {pitch:.1f} Hz | Tempo: {tempo:.1f} BPM")

    # Simple rule-based classification
    if energy > 0.05 and tempo > 120:
        return "angry"
    elif energy > 0.03 and pitch > 200:
        return "happy"
    elif energy < 0.02:
        return "sad"
    else:
        return "calm"

# Test
if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    sentiment = analyze_voice_sentiment(audio_file)
    print(f"🎭 Detected sentiment: {sentiment}")