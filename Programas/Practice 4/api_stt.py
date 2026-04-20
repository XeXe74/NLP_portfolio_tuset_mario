import assemblyai as aai
import sounddevice as sd
import soundfile as sf
import tempfile
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