from gtts import gTTS
import os

def speak_api(text: str, lang: str = "es") -> str:
    """
    Converts text to speech using Google TTS API and plays the audio.
    """

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Sending text to Google TTS API...")

    # Create TTS object and save the audio file
    tts = gTTS(text=text, lang=lang)
    output_path = os.path.join(output_dir, "output_api.mp3")
    tts.save(output_path)

    print(f"Audio generated in {output_path}")
    os.system(f"afplay {output_path}") # Play the audio
    return output_path

# Test
if __name__ == "__main__":
    text = input("Enter text to speak: ")
    speak_api(text)