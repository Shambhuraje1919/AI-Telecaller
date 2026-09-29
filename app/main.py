import logging
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client

from app.config import settings
from app.telecaller import twilio_media_stream_handler, active_sessions
from app.middleware import RequestIDMiddleware, RateLimitMiddleware, is_within_calling_hours, get_next_calling_window
from app.sarvam_client import sarvam_client
from app.cartesia_client import cartesia_client
from app.llm_client import llm_client

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "call_sid") and record.call_sid:
            log_obj["call_sid"] = record.call_sid
        if hasattr(record, "stream_sid") and record.stream_sid:
            log_obj["stream_sid"] = record.stream_sid
        return json.dumps(log_obj)

log_level = logging.INFO
if hasattr(settings, 'LOG_LEVEL'):
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=log_level, handlers=[handler])
logger = logging.getLogger("ai_telecaller")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    # Validate critical config
    if not settings.PUBLIC_HOSTNAME:
        logger.warning("PUBLIC_HOSTNAME not set!")

    # Pre-warm Cartesia HTTP connection so first call greeting is instant
    try:
        await cartesia_client._get_client()
        logger.info("Cartesia HTTP connection pre-warmed ✅")
    except Exception as e:
        logger.warning(f"Cartesia pre-warm failed (will retry on first call): {e}")

    yield
    logger.info("Application shutting down...")
    # Cleanup clients if needed
    if hasattr(sarvam_client, "close"):
        await sarvam_client.close()
    if hasattr(cartesia_client, "close"):
        await cartesia_client.close()
    if hasattr(llm_client, "close"):
        await llm_client.close()


app = FastAPI(title="AI Telecaller Prototype", version="1.0.0", lifespan=lifespan)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "llm_provider": settings.LLM_PROVIDER,
        "tts_provider": getattr(settings, "TTS_PROVIDER", "cartesia"),
        "cartesia_voice_id": settings.CARTESIA_VOICE_ID,
        "public_hostname": settings.PUBLIC_HOSTNAME,
        "active_calls": len(active_sessions)
    }

@app.get("/metrics")
async def metrics():
    return {
        "active_calls": len(active_sessions),
        "cartesia_metrics": cartesia_client.get_metrics(),
        "sarvam_metrics": sarvam_client.get_metrics(),
        "llm_metrics": llm_client.get_metrics()
    }


@app.post("/twiml")
@app.get("/twiml")
async def twiml_endpoint(request: Request):
    """Twilio Webhook endpoint returning TwiML that connects to the WebSocket media stream."""
    public_host = settings.PUBLIC_HOSTNAME.strip()
    if not public_host or "localhost" in public_host:
        logger.warning(
            "PUBLIC_HOSTNAME is currently set to localhost. Twilio needs a public domain (e.g. ngrok)!"
        )

    clean_host = (
        public_host.replace("https://", "")
        .replace("http://", "")
        .replace("wss://", "")
        .replace("ws://", "")
    ).strip().rstrip("/")

    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{clean_host}/media-stream" />
    </Connect>
</Response>"""

    return Response(content=twiml_response, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream_endpoint(websocket: WebSocket):
    """Bidirectional WebSocket media stream for live Twilio audio."""
    await twilio_media_stream_handler(websocket)


@app.post("/call")
@app.get("/call")
async def trigger_outbound_call(to: str):
    """
    Triggers an outbound phone call using Twilio REST API.
    Example: POST or GET /call?to=+919876543210
    """
    if not is_within_calling_hours():
        next_window = get_next_calling_window()
        raise HTTPException(status_code=403, detail=f"Outside TRAI calling hours. Next window at {next_window}.")

    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        raise HTTPException(
            status_code=400,
            detail="Twilio credentials (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN) are not configured."
        )

    if not settings.TWILIO_PHONE_NUMBER:
        raise HTTPException(
            status_code=400,
            detail="TWILIO_PHONE_NUMBER is not configured."
        )

    clean_host = (
        settings.PUBLIC_HOSTNAME.replace("https://", "")
        .replace("http://", "")
        .replace("wss://", "")
        .replace("ws://", "")
    ).strip().rstrip("/")
    twiml_url = f"https://{clean_host}/twiml"

    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            to=to,
            from_=settings.TWILIO_PHONE_NUMBER,
            url=twiml_url
        )
        logger.info(f"Initiated outbound call to {to}. Call SID: {call.sid}")
        return {
            "success": True,
            "message": f"Calling {to}...",
            "call_sid": call.sid,
            "status": call.status
        }
    except Exception as e:
        logger.error(f"Failed to place call to {to}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to place call: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
