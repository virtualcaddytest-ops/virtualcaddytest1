import os
import time
from fastapi import FastAPI, UploadFile, File
import httpx

app = FastAPI(title="CaddieAI Backend Core")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

@app.get("/")
def read_root():
    return {"status": "CaddieAI Backend attivo e pronto"}

@app.post("/v1/voice-rules")
async def process_voice_rule(file: UploadFile = File(...)):
    start_time = time.time()
    audio_bytes = await file.read()
    
    if not audio_bytes:
        return {"response_text": "Errore: File audio vuoto ricevuto dall'orologio."}
    
    if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
        return {"response_text": "Errore: Chiavi API non configurate su Render."}

    # 1. Speech-to-Text tramite Deepgram
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            stt_response = await client.post(
                "https://api.deepgram.com/v1/listen?language=it&model=nova-2",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/m4a"
                },
                content=audio_bytes
            )
            
            if stt_response.status_code != 200:
                return {"response_text": f"Errore Deepgram ({stt_response.status_code}): {stt_response.text}"}

            stt_data = stt_response.json()
            user_transcript = stt_data['results']['channels'][0]['alternatives'][0]['transcript']

            if not user_transcript.strip():
                return {"response_text": "Non ho sentito nulla, riprova a parlare."}

            # 2. Elaborazione LLM tramite Groq
          # Elaborazione LLM tramite Groq
            llm_response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {"role": "system", "content": "Sei un caddie esperto di golf. Rispondi in massimo 2 frasi in italiano."},
                        {"role": "user", "content": user_transcript}
                    ],
                    "max_tokens": 100
                }
            )

            if llm_response.status_code != 200:
                return {"response_text": f"Errore Groq ({llm_response.status_code}): {llm_response.text}"}

            ai_text = llm_response.json()['choices'][0]['message']['content']
            return {"response_text": ai_text}

    except Exception as e:
        return {"response_text": f"Errore server: {str(e)}"}
