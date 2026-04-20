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

def list_voices():
    """
    Prints all available voices on the system.
    """
    voices = engine.getProperty("voices") # Get available voices

    # Print only the index because the list is huge
    print(f"Available voices: 0 to {len(voices) - 1}")
    return len(voices)

# Test
if __name__ == "__main__":
    configure_engine()
    num_voices = list_voices() # Get the voices count

    # Allow user to select a voice index
    voice_index = int(input("Select voice index: "))
    # Validate the voice index
    if not 0 <= voice_index <= num_voices - 1:
        print("Invalid index, using default (0).")
        voice_index = 0

    text = input("Enter text to speak: ")
    speak_local(text, voice_index=voice_index)