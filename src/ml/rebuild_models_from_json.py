import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVOLUTIONARY_DIR = PROJECT_ROOT / "outputs" / "evolutionary"
GRID_DIR = PROJECT_ROOT / "outputs" / "gridsearch_optuna"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "reconstructed_models.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_model_json(json_path):
    data = load_json(json_path)

    # CASO Evolucionário
    if "Parametros" in data:

        score = data.get("Holdout_F1", -1.0)
        return data["Modelo"], data["Parametros"], score

    # CASO Baseline (GridSearch/Optuna)
    if "best_params" in data and "model" in data:

        score = data.get("f1_holdout", -1.0)
        return data["model"], data["best_params"], score

    return None

def aggregate_all_models():
    print(f"Varrendo resultados em: {PROJECT_ROOT}")

    # Lista pastas
    evo_exps = [d for d in EVOLUTIONARY_DIR.iterdir() if d.is_dir()] if EVOLUTIONARY_DIR.exists() else []
    grid_exps = [d for d in GRID_DIR.iterdir() if d.is_dir()] if GRID_DIR.exists() else []

    all_models_data = {}

    # Coleta EVOLUTIONARY
    for exp in evo_exps:
        for model in ["SVM", "LogisticRegression", "NaiveBayes", "FastText"]:
            json_path = exp / f"final_evaluation_{model}.json"
            if json_path.exists():
                parsed = parse_model_json(json_path)
                if parsed:
                    model_type, params, score = parsed

                    key = f"{exp.name}__{model_type}__evolutionary"
                    all_models_data[key] = {
                        "params": params,
                        "score": float(score),
                        "origin": "Evolutionary",
                        "model_type": model_type
                    }

    # Coleta GRID/OPTUNA
    for exp in grid_exps:
        for f in exp.iterdir():
            if f.name.startswith("final_") and f.name.endswith(".json"):
                parsed = parse_model_json(f)
                if parsed:
                    model_type, params, score = parsed

                    # Normalizar nomes (LinearSVC -> SVM, MultinomialNB -> NaiveBayes)

                    if model_type == "LinearSVC": model_type = "SVM"
                    if model_type == "MultinomialNB": model_type = "NaiveBayes"

                    key = f"{exp.name}__{model_type}__grid_optuna"
                    all_models_data[key] = {
                        "params": params,
                        "score": float(score),
                        "origin": "Baseline",
                        "model_type": model_type
                    }

    # Salvar arquivo
    OUTPUT_JSON.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_models_data, f, indent=4, ensure_ascii=False)

    print(f"JSON enriquecido salvo com {len(all_models_data)} modelos.")
    return all_models_data

if __name__ == "__main__":
    aggregate_all_models()