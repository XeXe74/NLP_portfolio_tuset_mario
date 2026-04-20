import requests
import time
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from config import ASSEMBLYAI_API_KEY

# Variables for AssemblyAI API
BASE_URL = "https://api.assemblyai.com"
HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

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

def upload_audio(audio_path: str) -> str:
    """
    Uploads the audio file to AssemblyAI and returns the hosted URL.
    """
    with open(audio_path, "rb") as f:
        response = requests.post(BASE_URL + "/v2/upload", headers=HEADERS, data=f)
    return response.json()["upload_url"]

def transcribe_api(audio_path: str) -> str:
    """
    Transcribes an audio file by sending it to the AssemblyAI API.
    """
    print("Sending audio to AssemblyAI API...")

    # Upload the audio file and get the URL
    audio_url = upload_audio(audio_path)

    # Submit transcription request
    data = {
        "audio_url": audio_url,
        "speech_models": ["universal-3-pro", "universal-2"],
        "language_code": "es"
    }
    # Get the API response with the transcript ID
    response = requests.post(BASE_URL + "/v2/transcript", headers=HEADERS, json=data)
    transcript_id = response.json()["id"]

    # Poll until transcription is complete
    polling_url = f"{BASE_URL}/v2/transcript/{transcript_id}"
    while True:
        result = requests.get(polling_url, headers=HEADERS).json()
        if result["status"] == "completed":
            return result["text"]
        elif result["status"] == "error":
            raise RuntimeError(f"Transcription failed: {result['error']}")
        time.sleep(3)

if __name__ == "__main__":
    audio_file = record_audio(duration=5)
    text = transcribe_api(audio_file)
    print(f"\nTranscription: {text}")
    os.remove(audio_file) # Clean up temporary file