import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings
from app.audio import (
    mulaw_to_pcm16,
    calculate_rms,
    resample_linear,
    create_wav_buffer,
    wav_to_mulaw_8k,
    chunk_mulaw_frames,
)
from app.sarvam_client import sarvam_client
from app.llm_client import llm_client

logger = logging.getLogger(__name__)


class CallSession:
    def __init__(self, websocket: WebSocket):
        self.ws: WebSocket = websocket
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.is_speaking: bool = False
        self.is_processing: bool = False

        # Inbound audio tracking (PCM 8kHz)
        self.pcm_audio_buffer: bytearray = bytearray()
        self.has_spoken: bool = False
        self.silence_accumulated_ms: float = 0.0

    def save_transcript(self):
        """Save conversation transcript to transcripts/ directory."""
        if not self.conversation_history or len(self.conversation_history) <= 1:
            return

        try:
            transcripts_dir = os.path.join(os.getcwd(), "transcripts")
            os.makedirs(transcripts_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename_base = f"call_{timestamp}_{self.call_sid or 'unknown'}"

            # 1. Save JSON
            json_path = os.path.join(transcripts_dir, f"{filename_base}.json")
            data = {
                "call_sid": self.call_sid,
                "stream_sid": self.stream_sid,
                "timestamp": timestamp,
                "turns_count": len(self.conversation_history),
                "history": self.conversation_history
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 2. Save readable text file
            txt_path = os.path.join(transcripts_dir, f"{filename_base}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== Call Transcript: {timestamp} (SID: {self.call_sid}) ===\n\n")
                for turn in self.conversation_history:
                    speaker = "प्रिया (Priya)" if turn["role"] == "assistant" else "ग्राहक (Customer)"
                    f.write(f"{speaker}: {turn['content']}\n")

            logger.info(f"Transcript saved to {txt_path} and {json_path}")
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")

    async def speak_text(self, text: str):
        """Synthesize text via Sarvam TTS and stream audio frames to Twilio."""
        if not text or not self.stream_sid:
            return

        self.is_speaking = True
        logger.info(f"Bot speaking: '{text}'")

        try:
            # 1. Generate audio with Sarvam TTS
            tts_wav_bytes = await sarvam_client.text_to_speech(text)
            if not tts_wav_bytes:
                logger.error("TTS returned empty audio.")
                return

            # 2. Convert to 8kHz mu-law audio
            mulaw_audio = wav_to_mulaw_8k(tts_wav_bytes)
            if not mulaw_audio:
                logger.error("Failed to convert TTS audio to 8kHz mu-law.")
                return

            # 3. Stream in 20ms frames (160 bytes)
            frames = chunk_mulaw_frames(mulaw_audio, chunk_size=160)
            for frame in frames:
                payload = base64.b64encode(frame).decode("utf-8")
                media_message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": payload
                    }
                }
                await self.ws.send_text(json.dumps(media_message))
                await asyncio.sleep(0.018)

            # Send a mark event indicating completion of speech
            mark_message = {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {
                    "name": "speech_done"
                }
            }
            await self.ws.send_text(json.dumps(mark_message))

        except Exception as e:
            logger.error(f"Error while speaking text: {e}")
        finally:
            self.is_speaking = False

    async def handle_customer_speech_end(self):
        """Triggered when Voice Activity Detection detects the customer has stopped speaking."""
        if len(self.pcm_audio_buffer) == 0:
            self.is_processing = False
            return
        raw_pcm = bytes(self.pcm_audio_buffer)

        # Reset audio accumulation
        self.pcm_audio_buffer.clear()
        self.has_spoken = False
        self.silence_accumulated_ms = 0.0

        try:
            logger.info(f"Processing customer speech ({len(raw_pcm)} bytes 8kHz PCM)...")

            # 1. Resample 8kHz PCM -> 16kHz PCM for Sarvam STT
            pcm16k = resample_linear(raw_pcm, orig_rate=8000, target_rate=16000)

            # 2. Pack into standard WAV
            wav_buf = create_wav_buffer(pcm16k, sample_rate=16000)
            wav_bytes = wav_buf.getvalue()

            # 3. Call Sarvam STT
            user_text = await sarvam_client.speech_to_text(wav_bytes)
            if not user_text or len(user_text.strip()) == 0:
                logger.info("STT returned empty transcript. Resuming listening.")
                return

            logger.info(f"Customer said: '{user_text}'")
            self.conversation_history.append({"role": "user", "content": user_text})

            # 4. Get LLM response
            reply_text = await llm_client.get_response(self.conversation_history)
            logger.info(f"AI response: '{reply_text}'")
            self.conversation_history.append({"role": "assistant", "content": reply_text})

            # 5. Speak reply back to customer
            await self.speak_text(reply_text)

        except Exception as e:
            logger.error(f"Error handling customer turn: {e}")
        finally:
            self.is_processing = False

    async def process_inbound_chunk(self, mulaw_payload_b64: str):
        """Process incoming 20ms mulaw chunk from Twilio."""
        if self.is_speaking or self.is_processing:
            return

        mulaw_bytes = base64.b64decode(mulaw_payload_b64)
        pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
        rms = calculate_rms(pcm_bytes)

        frame_duration_ms = (len(mulaw_bytes) / 8000.0) * 1000.0

        if rms > settings.SILENCE_RMS_THRESHOLD:
            # Voice detected
            if not self.has_spoken:
                logger.info(f"User voice started (RMS: {rms:.0f} > {settings.SILENCE_RMS_THRESHOLD})")
            self.has_spoken = True
            self.silence_accumulated_ms = 0.0
            self.pcm_audio_buffer.extend(pcm_bytes)
        else:
            # Silence detected
            if self.has_spoken:
                self.silence_accumulated_ms += frame_duration_ms
                self.pcm_audio_buffer.extend(pcm_bytes)

                total_duration_ms = (len(self.pcm_audio_buffer) / 2 / 8000.0) * 1000.0
                if (
                    self.silence_accumulated_ms >= settings.SILENCE_DURATION_MS
                    and total_duration_ms >= settings.MIN_SPEECH_DURATION_MS
                ) or total_duration_ms >= 15000:
                    logger.info(
                        f"User voice ended (duration: {total_duration_ms:.0f}ms, silence: {self.silence_accumulated_ms:.0f}ms). Sending to STT..."
                    )
                    self.is_processing = True
                    asyncio.create_task(self.handle_customer_speech_end())


async def twilio_media_stream_handler(websocket: WebSocket):
    """WebSocket connection handler for Twilio bidirectional Media Streams."""
    await websocket.accept()
    logger.info("Twilio WebSocket connection established.")

    session = CallSession(websocket)

    try:
        while True:
            raw_message = await websocket.receive_text()
            data = json.loads(raw_message)
            event = data.get("event")

            if event == "connected":
                logger.info(f"Twilio Media Stream protocol connected: {data}")

            elif event == "start":
                start_data = data.get("start", {})
                session.stream_sid = data.get("streamSid") or start_data.get("streamSid")
                session.call_sid = start_data.get("callSid")
                logger.info(f"Call started. StreamSid={session.stream_sid}, CallSid={session.call_sid}")

                # Speak greeting message
                greeting = settings.GREETING_MESSAGE
                session.conversation_history.append({"role": "assistant", "content": greeting})
                asyncio.create_task(session.speak_text(greeting))

            elif event == "media":
                media_info = data.get("media", {})
                payload = media_info.get("payload")
                if payload:
                    await session.process_inbound_chunk(payload)

            elif event == "mark":
                mark_name = data.get("mark", {}).get("name")
                logger.debug(f"Received mark event: {mark_name}")

            elif event == "stop":
                logger.info(f"Call stopped by Twilio. StreamSid={session.stream_sid}")
                break

    except WebSocketDisconnect:
        logger.info(f"Twilio WebSocket disconnected. StreamSid={session.stream_sid}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up session.")
        session.save_transcript()
