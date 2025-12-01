import os
import time
import math
import logging
from typing import Optional, Dict, Any

try:
    import google.generativeai as genai
except:
    genai = None

try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    from groq import Groq
except:
    Groq = None

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY and genai:
    genai.configure(api_key=GOOGLE_API_KEY)
    _gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
else:
    _gemini_model = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client_openai = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_client_groq = Groq(api_key=GROQ_API_KEY) if (GROQ_API_KEY and Groq) else None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
_client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if (DEEPSEEK_API_KEY and OpenAI) else None



def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("rate" in msg or "429" in msg or "quota" in msg or "too many" in msg)



def call_llm_api(model_id: str, prompt_completo: str, max_retries=4, base_backoff=1.0) -> Dict[str, Any]:
    start = time.time()
    attempt = 0
    last_exc = None

    while attempt <= max_retries:
        attempt += 1
        try:

            # Gemini
            if model_id.startswith("gemini"):
                if not _gemini_model:
                    raise ValueError("Cliente Gemini não inicializado.")
                response = _gemini_model.generate_content(prompt_completo)
                raw = response.text

            # OpenAI GPT
            elif model_id.startswith("gpt"):
                if not _client_openai:
                    raise ValueError("Cliente OpenAI não inicializado.")
                resp = _client_openai.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_completo}]
                )
                raw = resp.choices[0].message.content

            # Groq (LLaMA / Mixtral)
            elif model_id.startswith("llama") or model_id.startswith("mixtral"):
                if not _client_groq:
                    raise ValueError("Cliente Groq não inicializado.")
                resp = _client_groq.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_completo}]
                )
                raw = resp.choices[0].message.content

            # DeepSeek
            elif model_id.startswith("deepseek"):
                if not _client_deepseek:
                    raise ValueError("Cliente DeepSeek não inicializado.")
                resp = _client_deepseek.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_completo}]
                )
                raw = resp.choices[0].message.content

            else:
                raise ValueError(f"Modelo não reconhecido: {model_id}")

            elapsed = time.time() - start
            return {
                "raw_text": raw.strip() if raw else None,
                "tempo_s": elapsed,
                "erro": None,
                "try_count": attempt
            }

        except Exception as e:
            last_exc = e
            elapsed = time.time() - start
            logger.warning(f"Erro API [{model_id}] tentativa {attempt}: {e}")

            if isinstance(e, ValueError) and "inicializado" in str(e):
                return {"raw_text": None, "tempo_s": elapsed, "erro": str(e), "try_count": attempt}

            if _is_rate_limit_error(e) and attempt <= max_retries:
                backoff = base_backoff * (2 ** (attempt - 1))
                time.sleep(backoff)
                continue

            if attempt <= max_retries:
                time.sleep(base_backoff * attempt)
                continue

            return {"raw_text": None, "tempo_s": elapsed, "erro": str(e), "try_count": attempt}

    return {"raw_text": None, "tempo_s": time.time() - start, "erro": str(last_exc), "try_count": attempt}
