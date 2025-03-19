from threading import Thread
from queue import Queue, Empty
import platform
import os
import subprocess
import logging
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SpeechThread")

class Speech_thread(Thread):
    """
    A thread-based text-to-speech service that works across different platforms.
    Handles speech synthesis requests in a queue to prevent blocking the main thread.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the speech thread with a message queue."""
        super(Speech_thread, self).__init__(*args, **kwargs)
        self.queue = Queue()
        self.index = 0  # Default voice index
        self.stop_speaking = False  # Flag to gracefully terminate the thread
        self.system = platform.system()
        self.volume = 1.0  # Default volume (0.0 to 1.0)
        self.rate = 180    # Default speaking rate (words per minute)
        self.voices_cache = None  # Cache for available voices
        self.engine = None  # TTS engine reference (for non-macOS)
        self.language_code = "en"  # Default language code

    def run(self):
        """Main thread loop that processes text-to-speech requests."""
        try:
            logger.info(f"Starting speech thread on {self.system}")
            
            # Cache available voices
            self.cache_voices()
            
            # Main processing loop
            while not self.stop_speaking:
                try:
                    # Get text from queue with timeout to allow checking stop flag
                    text = self.queue.get(timeout=0.5)
                    
                    if text:
                        # Attempt to speak the text
                        self.speak_text(text)
                        
                except Empty:
                    # Queue timeout - this is expected, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing speech request: {e}")
                    logger.debug(traceback.format_exc())
                    # Brief pause to avoid error loops
                    time.sleep(0.1)
            
            # Clean up resources when stopping
            self.cleanup()
            logger.info("Speech thread stopped")
            
        except Exception as e:
            logger.error(f"Fatal error in speech thread: {e}")
            logger.debug(traceback.format_exc())

    def cache_voices(self):
        """Cache available voices for the current platform."""
        try:
            if self.system == "Darwin":  # macOS
                result = subprocess.run(['say', '-v', '?'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    logger.error(f"Error getting voices: {result.stderr.decode('utf-8')}")
                    return
                
                output = result.stdout.decode('utf-8')
                voices = []
                for line in output.splitlines():
                    if line:
                        voices.append(line.split()[0])
                self.voices_cache = voices
                logger.info(f"Cached {len(voices)} macOS voices")
            else:  # Windows/Linux
                try:
                    import pyttsx3
                    self.engine = pyttsx3.init()
                    voices = self.engine.getProperty('voices')
                    self.voices_cache = voices
                    logger.info(f"Cached {len(voices)} pyttsx3 voices")
                except ImportError:
                    logger.error("pyttsx3 not installed. Speech functionality limited.")
                except Exception as e:
                    logger.error(f"Error initializing pyttsx3: {e}")
        except Exception as e:
            logger.error(f"Error caching voices: {e}")

    def speak_text(self, text):
        """Speak the given text using the appropriate TTS engine."""
        if not text:
            logger.warning("Empty text received, skipping")
            return

        try:
            # Check if voice index is valid
            if self.voices_cache and self.index >= len(self.voices_cache):
                logger.warning(f"Voice index {self.index} out of range. Using default voice.")
                self.index = 0

            # Platform-specific speech synthesis
            if self.system == "Darwin":  # macOS
                self.speak_macos(text)
            else:  # Windows/Linux
                self.speak_pyttsx3(text)
                
            logger.debug(f"Successfully spoke: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        
        except Exception as e:
            logger.error(f"Error speaking text: {e}")

    def speak_macos(self, text):
        """Speak text using macOS say command."""
        try:
            if not self.voices_cache:
                # Fallback if voices weren't cached successfully
                os.system(f'say "{text}"')
                return
                
            voice_name = self.voices_cache[self.index]
            
            # Create command with all parameters
            cmd = [
                'say',
                '-v', voice_name,
                '-r', str(self.rate),
                text
            ]
            
            # Run speech synthesis
            subprocess.run(cmd, check=True, capture_output=True)
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error using 'say' command: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
            # Fallback to basic command
            try:
                os.system(f'say "{text}"')
            except Exception as fallback_e:
                logger.error(f"Fallback speech failed: {fallback_e}")
                
        except Exception as e:
            logger.error(f"Error in macOS speech: {e}")

    def speak_pyttsx3(self, text):
        """Speak text using pyttsx3 engine (Windows/Linux)."""
        try:
            import pyttsx3
            
            # Create a new engine if needed
            if not self.engine:
                self.engine = pyttsx3.init()
            
            # Set properties
            if self.voices_cache:
                self.engine.setProperty('voice', self.voices_cache[self.index].id)
            
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Speak and wait for completion
            self.engine.say(text)
            self.engine.runAndWait()
            
        except ImportError:
            logger.error("pyttsx3 is not installed. Cannot speak.")
        except Exception as e:
            logger.error(f"Error in pyttsx3 speech: {e}")
            
            # Try to reinitialize engine if it fails
            try:
                logger.info("Attempting to reinitialize speech engine")
                self.engine = pyttsx3.init()
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as reinit_e:
                logger.error(f"Engine reinitialization failed: {reinit_e}")

    def put_text(self, text):
        """Add text to the speech queue."""
        try:
            if text and isinstance(text, str):
                self.queue.put(text)
                logger.debug(f"Added to speech queue: '{text[:30]}{'...' if len(text) > 30 else ''}'")
            else:
                logger.warning(f"Invalid text received: {type(text)}")
        except Exception as e:
            logger.error(f"Error adding text to queue: {e}")

    def set_voice(self, index):
        """Change the voice to the specified index."""
        if index < 0:
            logger.warning(f"Invalid voice index: {index}, must be non-negative")
            return False
            
        self.index = index
        logger.info(f"Voice set to index {index}")
        return True

    def set_volume(self, volume):
        """Set the speech volume (0.0 to 1.0)."""
        if 0.0 <= volume <= 1.0:
            self.volume = volume
            logger.info(f"Volume set to {volume}")
            return True
        else:
            logger.warning(f"Invalid volume: {volume}, must be between 0.0 and 1.0")
            return False

    def set_rate(self, rate):
        """Set the speech rate in words per minute."""
        if rate > 0:
            self.rate = rate
            logger.info(f"Rate set to {rate}")
            return True
        else:
            logger.warning(f"Invalid rate: {rate}, must be positive")
            return False

    def clear_queue(self):
        """Clear all pending speech requests."""
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
            logger.info("Speech queue cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing queue: {e}")
            return False

    def cleanup(self):
        """Clean up resources when stopping the thread."""
        try:
            # Clear the queue
            self.clear_queue()
            
            # Clean up pyttsx3 engine if it exists
            if self.engine:
                try:
                    self.engine.stop()
                except:
                    pass
                self.engine = None
                
            logger.info("Speech resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up speech resources: {e}")

    def set_language(self, language_code):
        """Set the speech language.
        
        Args:
            language_code: Two-letter language code (e.g., 'en', 'fr', 'de')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.language_code = language_code
            
            # Platform-specific language setup
            if self.system == "Darwin":  # macOS
                # macOS uses specific voice names for different languages
                if language_code == "en":
                    voice_index = 0  # Default English voice
                elif language_code == "fr":
                    # Find a French voice
                    french_voices = [i for i, v in enumerate(self.voices_cache) 
                                   if v.lower().startswith(("fr_", "fre")) or "french" in v.lower()]
                    voice_index = french_voices[0] if french_voices else 0
                elif language_code == "de":
                    # Find a German voice
                    german_voices = [i for i, v in enumerate(self.voices_cache) 
                                   if v.lower().startswith(("de_", "ger")) or "german" in v.lower()]
                    voice_index = german_voices[0] if german_voices else 0
                elif language_code == "es":
                    # Find a Spanish voice
                    spanish_voices = [i for i, v in enumerate(self.voices_cache) 
                                    if v.lower().startswith(("es_", "spa")) or "spanish" in v.lower()]
                    voice_index = spanish_voices[0] if spanish_voices else 0
                elif language_code == "it":
                    # Find an Italian voice
                    italian_voices = [i for i, v in enumerate(self.voices_cache) 
                                    if v.lower().startswith(("it_", "ita")) or "italian" in v.lower()]
                    voice_index = italian_voices[0] if italian_voices else 0
                elif language_code == "pt":
                    # Find a Portuguese voice
                    portuguese_voices = [i for i, v in enumerate(self.voices_cache) 
                                      if v.lower().startswith(("pt_", "por")) or "portuguese" in v.lower()]
                    voice_index = portuguese_voices[0] if portuguese_voices else 0
                elif language_code == "nl":
                    # Find a Dutch voice
                    dutch_voices = [i for i, v in enumerate(self.voices_cache) 
                                  if v.lower().startswith(("nl_", "dut")) or "dutch" in v.lower()]
                    voice_index = dutch_voices[0] if dutch_voices else 0
                elif language_code == "ru":
                    # Find a Russian voice
                    russian_voices = [i for i, v in enumerate(self.voices_cache) 
                                   if v.lower().startswith(("ru_", "rus")) or "russian" in v.lower()]
                    voice_index = russian_voices[0] if russian_voices else 0
                elif language_code == "zh":
                    # Find a Chinese voice
                    chinese_voices = [i for i, v in enumerate(self.voices_cache) 
                                   if v.lower().startswith(("zh_", "chi")) or "chinese" in v.lower()]
                    voice_index = chinese_voices[0] if chinese_voices else 0
                elif language_code == "ja":
                    # Find a Japanese voice
                    japanese_voices = [i for i, v in enumerate(self.voices_cache) 
                                    if v.lower().startswith(("ja_", "jpn")) or "japanese" in v.lower()]
                    voice_index = japanese_voices[0] if japanese_voices else 0
                elif language_code == "ko":
                    # Find a Korean voice
                    korean_voices = [i for i, v in enumerate(self.voices_cache) 
                                  if v.lower().startswith(("ko_", "kor")) or "korean" in v.lower()]
                    voice_index = korean_voices[0] if korean_voices else 0
                elif language_code == "ar":
                    # Find an Arabic voice
                    arabic_voices = [i for i, v in enumerate(self.voices_cache) 
                                  if v.lower().startswith(("ar_", "ara")) or "arabic" in v.lower()]
                    voice_index = arabic_voices[0] if arabic_voices else 0
                else:
                    # Default to first voice for unsupported languages
                    voice_index = 0
                    
                # Set the voice index
                self.set_voice(voice_index)
                
            else:  # Windows/Linux with pyttsx3
                if self.engine:
                    # Map language code to pyttsx3 language code
                    language_mapping = {
                        "en": "en_US",
                        "fr": "fr_FR",
                        "de": "de_DE",
                        "es": "es_ES",
                        "it": "it_IT",
                        "pt": "pt_BR",
                        "nl": "nl_NL",
                        "ru": "ru_RU",
                        "zh": "zh_CN",
                        "ja": "ja_JP",
                        "ko": "ko_KR",
                        "ar": "ar_SA"
                    }
                    
                    # Find appropriate voice by language
                    target_lang = language_mapping.get(language_code, "en_US")
                    
                    if self.voices_cache:
                        for i, voice in enumerate(self.voices_cache):
                            if target_lang in voice.id.lower() or language_code in voice.id.lower():
                                self.set_voice(i)
                                break
            
            logger.info(f"Speech language set to: {language_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting language to {language_code}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def stop(self):
        """Gracefully stop the speech thread."""
        self.stop_speaking = True
        logger.info("Speech thread stop requested")


# Test function to verify functionality
def test_speech():
    """Test speech synthesis with various phrases."""
    print("Testing speech functionality...")
    
    speech_thread = Speech_thread()
    speech_thread.daemon = True
    speech_thread.start()
    
    print("Speech thread started")
    
    test_phrases = [
        "Hello, this is a test of the speech system.",
        "Chess piece moved from E2 to E4.",
        "Check mate in two moves."
    ]
    
    # Queue up test phrases
    for phrase in test_phrases:
        print(f"Speaking: '{phrase}'")
        speech_thread.put_text(phrase)
        time.sleep(3)  # Wait for phrase to be spoken
    
    print("Speech test complete")
    speech_thread.stop()

# Run test if script is executed directly
if __name__ == "__main__":
    test_speech()
