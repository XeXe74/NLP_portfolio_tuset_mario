import sounddevice as sd
import soundfile as sf
import whisper
import tempfile
import os

MODEL = whisper.load_model("base")  # Load Whisper model

def record_audio(duration: int = 5, sample_rate: int = 16000) -> str:
    """
    Records audio from the microphone and saves it to a temporary file.
    """
    print(f"Recording for {duration} seconds...")

    # Record audio for the given duration and sample rate
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait() # Wait until recording is finished
    print("Recording finished.")

    # Save to a temporary .wav file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate) # Write the recorded audio in the temporary file

    return tmp.name

def transcribe_local(audio_path: str) -> str:
    """
    Transcribes an audio file using Whisper running fully locally.
    """
    print("Transcribing...")
    result = MODEL.transcribe(audio_path, language="es") # Transcribe with Whisper, specifying language
    return result["text"]

# Test
if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    text = transcribe_local(audio_file)
    print(f"\nTranscription: {text}")
    os.remove(audio_file)  # Clean up temporary file