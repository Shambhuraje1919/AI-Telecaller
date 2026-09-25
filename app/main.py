import logging
from fastapi import FastAPI, WebSocket, Request, Response, HTTPException
from twilio.rest import Client

from app.config import settings
from app.telecaller import twilio_media_stream_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_telecaller")

app = FastAPI(title="AI Telecaller Prototype", version="1.0.0")


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "llm_provider": settings.LLM_PROVIDER,
        "stt_speaker": settings.SARVAM_TTS_SPEAKER,
        "public_hostname": settings.PUBLIC_HOSTNAME
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
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
