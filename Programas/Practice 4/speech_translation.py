from local_stt import record_audio, transcribe_local
from local_tts import speak_local
from api_tts import speak_api

def translate_and_speak(duration: int = 5, use_api_tts: bool = False) -> None:
    """
    Records speech, translates it to English using Whisper and speaks the translation using local or API TTS.
    """
    # Record audio
    audio_file = record_audio(duration)

    # Transcribe and translate to English using Whisper locally
    translation = transcribe_local(audio_file, translate=True)
    print(f"Translation: {translation}")

    # Speak the translation with either local TTS or API TTS based on user choice
    if use_api_tts:
        speak_api(translation, lang="en")
    else:
        speak_local(translation)

# Test
if __name__ == "__main__":
    use_api_tts = input("Use API for TTS? (y/n): ").strip().lower() == "y"
    translate_and_speak(use_api_tts=use_api_tts)