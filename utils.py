import pyttsx3
import whisper
import logging
import os
from typing import Optional

# ---------------- Logger Setup ----------------
logger = logging.getLogger("chatbot")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

# ---------------- Initialize Text-to-Speech ----------------
def init_tts() -> Optional[pyttsx3.Engine]:
    """
    Initialize the pyttsx3 TTS engine.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)  # Normal speech rate
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)  # Default system voice
        logger.info("✅ TTS engine initialized successfully.")
        return engine
    except Exception as e:
        logger.error(f"❌ Failed to initialize TTS engine: {e}")
        return None

# Global TTS engine
engine: Optional[pyttsx3.Engine] = init_tts()

def speak(text: str) -> None:
    """
    Convert text to speech safely.
    """
    if engine is None:
        logger.warning("⚠️ TTS engine not available.")
        return
    if not text.strip():
        logger.warning("⚠️ No text provided to speak.")
        return
    try:
        logger.info(f"🔊 Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"❌ TTS failed: {e}")

# ---------------- Whisper Model Handling ----------------
_whisper_model: Optional[whisper.Whisper] = None  # Lazy-loaded global

def load_whisper_model(model_name: str = "base") -> Optional[whisper.Whisper]:
    """
    Load Whisper model safely (lazy loading).
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        logger.info(f"⏳ Loading Whisper model: {model_name} ...")
        _whisper_model = whisper.load_model(model_name)
        logger.info("✅ Whisper model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
        _whisper_model = None

    return _whisper_model

def transcribe_audio(file_path: str, model_name: str = "base") -> str:
    """
    Convert audio file to text using Whisper safely.
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ Audio file does not exist: {file_path}")
        return ""

    model = _whisper_model or load_whisper_model(model_name)
    if model is None:
        logger.error("❌ Whisper model not loaded, cannot transcribe.")
        return ""

    try:
        logger.info(f"🎙️ Transcribing audio file: {file_path}")
        result = model.transcribe(file_path)
        text = result.get("text", "")
        if not text.strip():
            logger.warning(f"⚠️ Transcription returned empty text for {file_path}")
        else:
            logger.info(f"📝 Transcribed text: {text.strip()}")
        return text.lower().strip()
    except Exception as e:
        logger.error(f"❌ Audio transcription failed: {e}")
        return ""
