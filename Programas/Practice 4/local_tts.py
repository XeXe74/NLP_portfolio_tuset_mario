import pyttsx3

engine = pyttsx3.init()  # Initialize the local TTS

def configure_engine(rate: int = 150, volume: float = 1.0) -> None:
    """
    Configures the speech engine's rate and volume.
    """
    engine.setProperty("rate", rate) # Words per minute
    engine.setProperty("volume", volume) # Volume

def speak_local(text: str, voice_index: int = 0) -> None:
    """
    Speaks the given text using the local TTS engine.
    """
    voices = engine.getProperty("voices") # Get available voices
    if voices:
        engine.setProperty("voice", voices[voice_index].id) # Set the voice (0 for default, 1 for another voice, etc.)

    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()  # Block until speaking is finished

# Test
if __name__ == "__main__":
    configure_engine()
    text = input("Enter text to speak: ")
    speak_local(text)