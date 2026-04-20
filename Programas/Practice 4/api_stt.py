import assemblyai as aai
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from config import ASSEMBLYAI_API_KEY

# Set up Assembly
aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber() # Create a transcriber instance to interact with the API

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
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name

def transcribe_api(audio_path: str) -> str:
    """
    Transcribes an audio file by sending it to the AssemblyAI API.
    """
    print("Sending audio to AssemblyAI API...")

    # Configure transcription settings
    config = aai.TranscriptionConfig(
        language_code="es",
        speech_model=aai.SpeechModel.universal  # Universal mode for better accuracy across languages
    )
    result = transcriber.transcribe(audio_path, config=config) # Transcribe the audio file with the specified configuration
    return result.text

if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    text = transcribe_api(audio_file)
    print(f"\nTranscription: {text}")
    os.remove(audio_file) # Clean up temporary file