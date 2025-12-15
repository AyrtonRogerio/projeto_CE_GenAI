import time
import random
from datetime import datetime
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(current_dir)


try:

    from src.genai.connectors import call_llm_api
    from src.genai.prompts import PROMPT_STRATEGIES
    from src.genai.dataset_adapter import carregar_demandas
    from src.genai.classification import limpar_resposta
    from src.genai.results_writer import save_results_incremental
except ImportError:

    from connectors import call_llm_api
    from prompts import PROMPT_STRATEGIES
    from dataset_adapter import carregar_demandas
    from classification import limpar_resposta
    from results_writer import save_results_incremental


TEST_MODE = False
NUM_TEST = 5

MODELOS = [
    "gemini-2.5-flash-lite",
    "gpt-5-nano",
    "llama-3.1-8b-instant",
]

SLEEP = 1.0

def run_benchmark(path_csv: str):
    print(f"Iniciando Benchmark GenAI")
    print(f"Modo Teste: {TEST_MODE}")
    print(f"Modelos Selecionados: {MODELOS}")

    demandas = carregar_demandas(path_csv)

    if TEST_MODE:
        print(f"AVISO: Rodando apenas {NUM_TEST} amostras aleatórias!")
        demandas = random.sample(demandas, min(NUM_TEST, len(demandas)))

    total_steps = len(demandas) * len(PROMPT_STRATEGIES) * len(MODELOS)
    print(f"Total de chamadas previstas à API: {total_steps}")

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    buffer = []
    count = 0

    for d in demandas:
        for p in PROMPT_STRATEGIES:
            # Preenche o template
            prompt = p["template"].format(demanda=d["texto"])

            for modelo in MODELOS:
                count += 1
                if count % 10 == 0:
                    print(f"Progresso: {count}/{total_steps} ({(count/total_steps)*100:.1f}%)")

                # Chamada API
                resp = call_llm_api(modelo, prompt)

                # Pós-processamento
                cat_limpa = limpar_resposta(resp["raw_text"])

                # Validação (Ground Truth)
                acertou = 1 if cat_limpa.lower() == str(d["classificacao_real"]).lower().strip() else 0

                row = {
                    "demanda_id": d["id"],
                    "classificacao_real": d["classificacao_real"],
                    "prompt_id": p["id"],
                    "modelo_id": modelo,
                    "classificacao_limpa": cat_limpa,
                    "acertou": acertou,
                    "demanda_texto": d["texto"],
                    "classificacao_raw": resp["raw_text"],
                    "tempo_resposta_s": resp["tempo_s"],
                    "erro": resp["erro"],
                    "try_count": resp["try_count"]
                }

                buffer.append(row)

                # Salva parcial a cada 10 requisições
                if len(buffer) >= 10:
                    save_results_incremental(buffer, run_name)
                    buffer = []

                time.sleep(SLEEP)

    # Salva o restante
    if buffer:
        save_results_incremental(buffer, run_name)

    print(f"✔ Benchmark GenAI concluído! Dados em outputs/genai_benchmarks/{run_name}")

if __name__ == "__main__":

    if os.path.exists("data/raw/ouvidoria_sintetico.csv"):
        csv_path = "data/raw/ouvidoria_sintetico.csv"
    else:

        csv_path = os.path.join(project_root, "data", "raw", "ouvidoria_sintetico.csv")

    if os.path.exists(csv_path):
        run_benchmark(csv_path)
    else:
        print(f"ERRO: Dataset não encontrado em {csv_path}")