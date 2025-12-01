import time
import random
from datetime import datetime

from .connectors import call_llm_api
from .prompts import PROMPT_STRATEGIES
from .dataset_adapter import carregar_demandas
from .classification import limpar_resposta
from .results_writer import save_results_incremental

TEST_MODE = True
NUM_TEST = 5

MODELOS = [
    "gemini-2.5-flash-lite",
    "gpt-5-nano",
    "llama-3.1-8b-instant",
]

SLEEP = 1.0


def run_benchmark(path_csv: str):
    demandas = carregar_demandas(path_csv)
    if TEST_MODE:
        demandas = random.sample(demandas, min(NUM_TEST, len(demandas)))

    total = len(demandas) * len(PROMPT_STRATEGIES) * len(MODELOS)
    count = 0

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    buffer = []

    for d in demandas:
        for p in PROMPT_STRATEGIES:
            prompt = p["template"].format(demanda=d["texto"])

            for modelo in MODELOS:
                count += 1
                print(f"[{count}/{total}] {modelo} | {p['id']} | ID {d['id']}")

                resp = call_llm_api(modelo, prompt)
                cat = limpar_resposta(resp["raw_text"])

                row = {
                    "demanda_id": d["id"],
                    "classificacao_real": d["classificacao_real"],
                    "prompt_id": p["id"],
                    "modelo_id": modelo,
                    "classificacao_limpa": cat,
                    "acertou": int(cat == d["classificacao_real"]),
                    "demanda_texto": d["texto"],
                    "classificacao_raw": resp["raw_text"],
                    "tempo_resposta_s": resp["tempo_s"],
                    "erro": resp["erro"],
                    "try_count": resp["try_count"]
                }

                buffer.append(row)

                if len(buffer) >= 20:
                    save_results_incremental(buffer, run_name)
                    buffer = []

                time.sleep(SLEEP)

    if buffer:
        save_results_incremental(buffer, run_name)

    print("✔ Benchmark concluído!")
