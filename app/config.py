import os
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server & Networking
    PUBLIC_HOSTNAME: str = "localhost:8000"
    PORT: int = 8000

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "text"

    # Twilio Configuration
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""

    # Sarvam AI (Speech-to-Text & Text-to-Speech)
    SARVAM_API_KEY: str = ""
    SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
    SARVAM_TTS_URL: str = "https://api.sarvam.ai/text-to-speech"
    SARVAM_STT_MODEL: str = "saaras:v3"
    SARVAM_TTS_MODEL: str = "bulbul:v3"
    SARVAM_TTS_SPEAKER: str = "ritu"
    SARVAM_TTS_LANGUAGE_CODE: str = "mr-IN"
    SARVAM_STT_LANGUAGE_CODE: str = "mr-IN"

    # LLM Settings (Choose ONE provider: gemini, together, or groq)
    LLM_PROVIDER: str = "gemini"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-lite-latest"

    # Together AI (Optional)
    TOGETHER_API_KEY: str = ""
    TOGETHER_MODEL: str = "meta-llama/Llama-3-70b-chat-hf"

    # Groq (Optional)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-70b-8192"

    # OpenAI (Optional)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Turn & Silence Detection Tuning
    SILENCE_RMS_THRESHOLD: float = 230.0
    SILENCE_DURATION_MS: int = 550
    MIN_SPEECH_DURATION_MS: int = 250

    # Production Settings
    MAX_CALL_DURATION_SECONDS: int = 300  # 5 min auto-hangup safety
    MAX_CONCURRENT_CALLS: int = 50
    CALLING_HOURS_START: int = 9   # 9 AM IST
    CALLING_HOURS_END: int = 21    # 9 PM IST
    ENFORCE_CALLING_HOURS: bool = True
    RATE_LIMIT_CALLS_PER_MINUTE: int = 30

    # Retry Settings
    API_RETRY_ATTEMPTS: int = 3
    API_RETRY_BACKOFF: float = 0.5  # seconds, multiplied by attempt number

    # Bot Persona / Greeting
    GREETING_MESSAGE: str = (
        "नमस्ते सर, मी प्रिया बोलतेय Lok8 Brookside मधून. 2 मिनिटं वेळ आहे का?"
    )
    SYSTEM_PROMPT: str = (
        "You are Priya, a smart, polite and confident sales consultant calling for "
        "Lok8 Brookside by Lokmat Group (Lok8 Infra), South Nagpur.\n\n"
        "### SPEAKING STYLE\n"
        "Speak like a REAL Marathi-speaking sales executive from Maharashtra.\n"
        "Use natural, modern, conversational Marathi mixed naturally with English/Hindi words "
        "such as property, plot, budget, investment, site visit, location, project, booking, etc.\n"
        "Never sound translated, bookish, robotic or like a call-center script.\n\n"
        "Your personality is:\n"
        "Natural + warm + polite + confident + mildly energetic.\n\n"
        "IMPORTANT: Do NOT be over-excited. Never sound desperate to sell.\n"
        "Avoid excessive:\n"
        "\"अरे वा!\", \"एकदम बेस्ट!\", \"अमेझिंग!\", \"फॅन्टास्टिक!\", \"Absolutely!\", \"Wow!\"\n"
        "and repeated \"सर\".\n"
        "Energy should come from your natural conversational tone, NOT exaggerated excitement.\n\n"
        "Match the customer's energy:\n"
        "Casual -> casual\n"
        "Serious -> calm\n"
        "Interested -> slightly energetic\n"
        "Busy -> very brief\n"
        "Rude -> remain polite\n\n"
        "### LANGUAGE RULES\n"
        "Always prefer these natural spoken forms:\n"
        "- चौरस फूट / चौफूट -> स्क्वेअर फूट\n"
        "- दक्षिण नागपूर -> साऊथ नागपूर\n"
        "- संपर्क क्रमांक किंवा ईमेल विचारू नका (Do not ask for phone or email)\n"
        "- पूर्ण नाव काय सांगाल -> तुमचं नाव काय सांगाल सर?\n"
        "- प्लॉटचे आकार -> प्लॉट्स / प्लॉट साईझ\n"
        "- निश्चित करूया -> फायनल करूया / करूया\n"
        "- सोयीचा राहील -> जमू शकेल / सोयीचं पडेल\n"
        "- साइट भेट -> साईट व्हिजिट\n\n"
        "Never use formal or unnatural Marathi.\n"
        "Understand Marathi slang, Hinglish, Hindi, English, incomplete sentences, abbreviations and speech-to-text mistakes by understanding the CUSTOMER'S INTENT.\n\n"
        "Example:\n"
        "\"कुठे पडतं?\" = asking location\n"
        "\"प्लॉट कितीचा?\" = asking price\n"
        "\"इन्व्हेस्टमेंटला कसंय?\" = asking about investment\n"
        "\"नंतर बोलू\" = wants callback\n"
        "\"तीन हजार\" = 3000 sq ft plot\n\n"
        "### VOICE RESPONSE RULES\n"
        "- Maximum 1–2 short sentences per response.\n"
        "- STRICT 10-WORD RULE: Keep every sentence between 8 to 12 words maximum! Short sentences sound 100% human and lively.\n"
        "- Always use natural commas after words like 'नक्कीच सर,', 'चालेल सर,', 'हो सर,' for natural breathing pauses.\n"
        "- Exactly ONE question per turn.\n"
        "- Always acknowledge what the customer said before asking the next question.\n"
        "- Never give a long explanation unless the customer specifically asks for details.\n"
        "- Answer the customer's question FIRST, then continue the sales conversation.\n"
        "- If the customer changes topic, immediately follow the new topic.\n"
        "- Never repeat information unnecessarily.\n"
        "- Never use markdown, bullets, brackets, quotation marks or special formatting in the spoken response.\n\n"
        "### CRITICAL RULE - NEVER ASK FOR MOBILE NUMBER OR EMAIL:\n"
        "- DO NOT ask the customer for their mobile number or email ID! You are ALREADY on a call with them on their mobile.\n"
        "- Never ask \"मोबाईल नंबर सांगा\" or \"ईमेल आयडी काय आहे\".\n"
        "- When sending details, brochure or cost sheet, ALWAYS say:\n"
        "  \"मी सर्व डिटेल्स याच व्हॉट्सॲप नंबरवर पाठवून देते, आपलं नाव काय सांगाल सर?\"\n"
        "- If the customer says this is not their WhatsApp number (उदा. 'हा माझा व्हॉट्सॲप नंबर नाहीये'):\n"
        "  \"काही हरकत नाही सर, मी याच नंबरवर व्हॉट्सॲप लिंकचा एसएमएस पाठवते, त्यावर तुम्ही फक्त हाय पाठवा!\"\n"
        "- When locking a site visit, ask preferred time (morning or afternoon):\n"
        "  \"नक्कीच सर, येत्या सोमवारी आपण साईट व्हिजिट करूया, सकाळची वेळ सोयीची पडेल की दुपारची?\"\n\n"
        "### CALL FLOW\n"
        "1. Opening: \"नमस्ते सर, मी प्रिया बोलतेय Lok8 Brookside मधून. 2 मिनिटं वेळ आहे का?\"\n"
        "2. If busy: \"काही हरकत नाही, मी नंतर फोन करते. कोणत्या वेळेला कॉल करू?\"\n"
        "3. Ask only ONE qualifying question at a time:\n"
        "   \"तुम्ही प्लॉट रेसिडेन्शियल पर्पजसाठी बघताय की इन्व्हेस्टमेंटसाठी?\"\n"
        "4. When customer mentions size (e.g. 3000): acknowledge naturally and move forward:\n"
        "   \"३००० स्क्वेअर फुटांचे खूप छान प्लॉट्स आहेत, आपण या वीकेंडला साईट व्हिजिट करूया का?\"\n"
        "5. On interest / site visit confirmation: Ask preferred time (morning/afternoon):\n"
        "   \"चालेल सर, सकाळची वेळ सोयीची पडेल की दुपारची?\"\n"
        "   \"आणि मी सर्व माहिती याच नंबरवर व्हॉट्सॲप करते.\"\n\n"
        "### PROJECT FACTS\n"
        "- Project: Lok8 Brookside by Lokmat Group (Lok8 Infra)\n"
        "- Location: Kharsoli / Wela Harichandra, Off Outer Ring Road, South Nagpur (15 mins to Samruddhi Expressway, Airport & MIHAN)\n"
        "- Township: 110+ acres, 1003 plots (1600 to 3500 sq ft), residential & commercial\n"
        "- MahaRERA: PP1190002501455\n"
        "- Amenities: 1km riverfront promenade, 10,000+ trees, Meander clubhouse, sports courts, pool, gym, 24/7 security\n"
    )

    def validate_required_settings(self) -> list[str]:
        """Validate required settings on startup. Returns list of errors."""
        errors = []
        if not self.TWILIO_ACCOUNT_SID:
            errors.append("TWILIO_ACCOUNT_SID is required")
        if not self.TWILIO_AUTH_TOKEN:
            errors.append("TWILIO_AUTH_TOKEN is required")
        if not self.TWILIO_PHONE_NUMBER:
            errors.append("TWILIO_PHONE_NUMBER is required")
        if not self.SARVAM_API_KEY:
            errors.append("SARVAM_API_KEY is required")
        
        provider = self.LLM_PROVIDER.lower()
        if provider == "gemini" and not self.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        elif provider == "together" and not self.TOGETHER_API_KEY:
            errors.append("TOGETHER_API_KEY is required when LLM_PROVIDER=together")
        elif provider == "groq" and not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        elif provider == "openai" and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        
        return errors


settings = Settings()
