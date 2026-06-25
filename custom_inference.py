"""Custom inference wrappers for direct API calls (no LiveKit Inference quota)"""
import os
import asyncio
import logging
from typing import AsyncGenerator
from livekit.agents import inference, llm, tts, stt
from livekit.agents.llm import ChatMessage, ChatContext
from openai import AsyncOpenAI
from deepgram import DeepgramClient, PrerecordedOptions
import piper
import wave

logger = logging.getLogger("custom-inference")

# ============================================================================
# Custom STT using Deepgram directly
# ============================================================================

class DeepgramSTT(stt.STT):
    def __init__(self, api_key: str = None, language: str = "en"):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.language = language
        self.client = DeepgramClient(api_key=self.api_key)
        self.sample_rate = 16000
        self.num_channels = 1

    async def recognize(self, audio: bytes) -> str:
        """Transcribe audio using Deepgram"""
        try:
            options = PrerecordedOptions(
                model="nova-2-phonecall",
                language=self.language,
            )
            response = self.client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio},
                options,
            )
            return response.results.channels[0].alternatives[0].transcript
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            return ""


# ============================================================================
# Custom LLM using OpenAI directly
# ============================================================================

class OpenAILLM(llm.LLM):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.5):
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def chat(self, context: ChatContext) -> AsyncGenerator[str, None]:
        """Stream LLM response using OpenAI"""
        try:
            messages = [{"role": m.role, "content": m.content} for m in context.messages]
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"OpenAI LLM failed: {e}")
            yield ""


# ============================================================================
# Custom TTS using Piper
# ============================================================================

class PiperTTS(tts.TTS):
    def __init__(self, voice: str = "en_US-amy-medium"):
        self.voice = voice
        self.sample_rate = 22050
        self.num_channels = 1

    async def synthesize(self, text: str) -> bytes:
        """Synthesize speech using Piper"""
        try:
            # Run piper in executor to avoid blocking
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None,
                self._piper_synthesize,
                text
            )
            return audio_data
        except Exception as e:
            logger.error(f"Piper TTS failed: {e}")
            return b""

    def _piper_synthesize(self, text: str) -> bytes:
        """Blocking Piper synthesis (runs in executor)"""
        try:
            import io

            # Use piper to synthesize
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.num_channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)

                # Generate audio using piper
                for audio_chunk in piper.synthesize(text, self.voice):
                    wav_file.writeframes(audio_chunk)

            return audio_buffer.getvalue()
        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            return b""
