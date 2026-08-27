import os
import time
from fastapi import FastAPI, UploadFile, File
import httpx

app = FastAPI(title="CaddieAI Backend Core")

# Configurazione API Keys (da impostare nei secrets di Render)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.get("/")
def read_root():
    return {"status": "CaddieAI Backend attivo e pronto"}

@app.post("/v1/voice-rules")
async def process_voice_rule(file: UploadFile = File(...)):
    start_time = time.time()
    audio_bytes = await file.read()

    # 1. Speech-to-Text tramite Deepgram (Latenza ~0.8s)
    async with httpx.AsyncClient() as client:
        stt_response = await client.post(
            "https://api.deepgram.com/v1/listen?language=it&model=nova-2",
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
            content=audio_bytes
        )
        transcript = stt_response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]

    # 2. RAG + Inferenza LLM tramite Groq (Latenza ~1.2s)
    system_prompt = (
        "Sei un assistente esperto e diretto per le Regole del Golf R&A/USGA. "
        "Rispondi al giocatore in massimo 3 o 4 punti elenco diretti, imperativi e operativi. "
        "Nessun convenevole, nessuna spiegazione teorica o legalese. "
        "Indica subito se c'è penalità o meno."
    )
    
    async with httpx.AsyncClient() as client:
        llm_response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript}
                ],
                "temperature": 0.2
            }
        )
        ai_text = llm_response.json()["choices"][0]["message"]["content"]

    total_latency = time.time() - start_time

    return {
        "transcript": transcript,
        "response_text": ai_text,
        "latency_seconds": round(total_latency, 2)
    }
