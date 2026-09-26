import asyncio
import base64
import json
import logging
import os
import time
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

# Module-level tracking for active calls
active_sessions: Dict[str, "CallSession"] = {}
MAX_CONCURRENT_CALLS = getattr(settings, "MAX_CONCURRENT_CALLS", 100)
MAX_CALL_DURATION_SECONDS = getattr(settings, "MAX_CALL_DURATION_SECONDS", 600)
FALLBACK_MESSAGE = getattr(settings, "FALLBACK_MESSAGE", "I'm having trouble right now, give me a moment.")

class CallSession:
    def __init__(self, websocket: WebSocket):
        self.ws: WebSocket = websocket
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        self.call_start_time = datetime.now()

        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.is_speaking: bool = False
        self.is_processing: bool = False
        self.interrupted: bool = False

        # Inbound audio tracking (PCM 8kHz)
        self.pcm_audio_buffer: bytearray = bytearray()
        self.has_spoken: bool = False
        self.silence_accumulated_ms: float = 0.0

        # Metrics & Management
        self.metrics: Dict[str, List[float]] = {
            "stt_latency": [],
            "llm_latency": [],
            "tts_latency": []
        }
        self._hangup_task: Optional[asyncio.Task] = None

    def log_with_context(self, level, message):
        extra = {"call_sid": self.call_sid, "stream_sid": self.stream_sid}
        logger.log(level, f"[Call {self.call_sid}] {message}", extra=extra)

    def save_transcript(self):
        """Save conversation transcript to transcripts/ directory."""
        if not self.conversation_history or len(self.conversation_history) <= 1:
            return

        try:
            transcripts_dir = os.path.join(os.getcwd(), "transcripts")
            os.makedirs(transcripts_dir, exist_ok=True)

            timestamp = self.call_start_time.strftime("%Y-%m-%d_%H-%M-%S")
            filename_base = f"call_{timestamp}_{self.call_sid or 'unknown'}"

            # 1. Save JSON
            json_path = os.path.join(transcripts_dir, f"{filename_base}.json")
            data = {
                "call_sid": self.call_sid,
                "stream_sid": self.stream_sid,
                "timestamp": timestamp,
                "turns_count": len(self.conversation_history),
                "history": self.conversation_history,
                "metrics": self.metrics
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

            self.log_with_context(logging.INFO, f"Transcript saved to {txt_path} and {json_path}")
        except Exception as e:
            self.log_with_context(logging.ERROR, f"Failed to save transcript: {e}")

    async def _auto_hangup(self):
        """Timer to enforce maximum call duration."""
        try:
            await asyncio.sleep(MAX_CALL_DURATION_SECONDS)
            self.log_with_context(logging.WARNING, f"Max call duration ({MAX_CALL_DURATION_SECONDS}s) reached. Hanging up.")
            if self.ws.client_state.name == 'CONNECTED':
                await self.ws.close()
        except asyncio.CancelledError:
            pass

    async def _send_clear_event(self):
        """Send clear event to Twilio to stop playback during barge-in."""
        if not self.stream_sid:
            return
        try:
            clear_message = {
                "event": "clear",
                "streamSid": self.stream_sid
            }
            await self.ws.send_text(json.dumps(clear_message))
            self.log_with_context(logging.INFO, "Sent clear event for barge-in")
        except Exception as e:
            self.log_with_context(logging.ERROR, f"Error sending clear event: {e}")

    async def speak_text(self, text: str):
        """Synthesize text via Sarvam TTS and stream audio frames to Twilio."""
        if not text or not self.stream_sid:
            return

        self.is_speaking = True
        self.interrupted = False
        self.log_with_context(logging.INFO, f"Bot speaking: '{text}'")

        try:
            start_tts = time.time()
            # 1. Generate audio with Sarvam TTS
            tts_wav_bytes = await sarvam_client.text_to_speech(text)
            self.metrics["tts_latency"].append(time.time() - start_tts)

            if not tts_wav_bytes:
                self.log_with_context(logging.ERROR, "TTS returned empty audio.")
                return

            # 2. Convert to 8kHz mu-law audio
            mulaw_audio = wav_to_mulaw_8k(tts_wav_bytes)
            if not mulaw_audio:
                self.log_with_context(logging.ERROR, "Failed to convert TTS audio to 8kHz mu-law.")
                return

            # 3. Stream in 20ms frames (160 bytes)
            frames = chunk_mulaw_frames(mulaw_audio, chunk_size=160)
            for frame in frames:
                if self.interrupted:
                    self.log_with_context(logging.INFO, "Playback interrupted by barge-in")
                    break

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

            if not self.interrupted:
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
            self.log_with_context(logging.ERROR, f"Error while speaking text: {e}")
        finally:
            self.is_speaking = False
            self.interrupted = False

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
            self.log_with_context(logging.INFO, f"Processing customer speech ({len(raw_pcm)} bytes 8kHz PCM)...")

            # 1. Resample 8kHz PCM -> 16kHz PCM for Sarvam STT
            pcm16k = resample_linear(raw_pcm, orig_rate=8000, target_rate=16000)

            # 2. Pack into standard WAV
            wav_buf = create_wav_buffer(pcm16k, sample_rate=16000)
            wav_bytes = wav_buf.getvalue()

            # 3. Call Sarvam STT
            start_stt = time.time()
            user_text = await sarvam_client.speech_to_text(wav_bytes)
            self.metrics["stt_latency"].append(time.time() - start_stt)

            if not user_text or len(user_text.strip()) == 0:
                self.log_with_context(logging.INFO, "STT returned empty transcript. Resuming listening.")
                return

            self.log_with_context(logging.INFO, f"Customer said: '{user_text}'")
            self.conversation_history.append({"role": "user", "content": user_text})

            # 4. Get LLM response
            start_llm = time.time()
            reply_text = await llm_client.get_response(self.conversation_history)
            self.metrics["llm_latency"].append(time.time() - start_llm)
            self.log_with_context(logging.INFO, f"AI response: '{reply_text}'")

            if not reply_text:
                raise ValueError("LLM returned empty reply")

            self.conversation_history.append({"role": "assistant", "content": reply_text})

            # 5. Speak reply back to customer
            await self.speak_text(reply_text)

        except Exception as e:
            self.log_with_context(logging.ERROR, f"Error handling customer turn: {e}")
            await self.speak_text(FALLBACK_MESSAGE)
        finally:
            self.is_processing = False

    async def process_inbound_chunk(self, mulaw_payload_b64: str):
        """Process incoming 20ms mulaw chunk from Twilio."""
        mulaw_bytes = base64.b64decode(mulaw_payload_b64)
        pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
        rms = calculate_rms(pcm_bytes)

        frame_duration_ms = (len(mulaw_bytes) / 8000.0) * 1000.0

        if rms > settings.SILENCE_RMS_THRESHOLD:
            # Voice detected
            if self.is_speaking and not self.interrupted:
                self.log_with_context(logging.INFO, "Barge-in detected!")
                self.interrupted = True
                await self._send_clear_event()

            if not self.has_spoken:
                self.log_with_context(logging.INFO, f"User voice started (RMS: {rms:.0f} > {settings.SILENCE_RMS_THRESHOLD})")
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
                    self.log_with_context(logging.INFO,
                        f"User voice ended (duration: {total_duration_ms:.0f}ms, silence: {self.silence_accumulated_ms:.0f}ms). Sending to STT..."
                    )
                    
                    if not self.is_processing:
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

                if len(active_sessions) >= MAX_CONCURRENT_CALLS:
                    session.log_with_context(logging.WARNING, "Max concurrent calls reached, rejecting call.")
                    await websocket.close(code=1008)
                    return

                active_sessions[session.stream_sid] = session
                session._hangup_task = asyncio.create_task(session._auto_hangup())
                session.log_with_context(logging.INFO, f"Call started. StreamSid={session.stream_sid}, CallSid={session.call_sid}")

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
                session.log_with_context(logging.DEBUG, f"Received mark event: {mark_name}")

            elif event == "stop":
                session.log_with_context(logging.INFO, f"Call stopped by Twilio. StreamSid={session.stream_sid}")
                break

    except WebSocketDisconnect:
        session.log_with_context(logging.INFO, f"Twilio WebSocket disconnected. StreamSid={session.stream_sid}")
    except Exception as e:
        session.log_with_context(logging.ERROR, f"WebSocket error: {e}")
    finally:
        session.log_with_context(logging.INFO, "Cleaning up session.")
        if session._hangup_task:
            session._hangup_task.cancel()
        if session.stream_sid in active_sessions:
            del active_sessions[session.stream_sid]
        
        def avg(l): return sum(l)/len(l) if l else 0
        summary = {
            "duration_secs": (datetime.now() - session.call_start_time).total_seconds(),
            "avg_stt_ms": avg(session.metrics["stt_latency"]) * 1000,
            "avg_llm_ms": avg(session.metrics["llm_latency"]) * 1000,
            "avg_tts_ms": avg(session.metrics["tts_latency"]) * 1000,
        }
        session.log_with_context(logging.INFO, f"Call summary metrics: {json.dumps(summary)}")
        
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, session.save_transcript)
