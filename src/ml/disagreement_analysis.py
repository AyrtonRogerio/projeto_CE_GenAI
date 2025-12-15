import os
import json
import pandas as pd
from glob import glob


PROJECT_ROOT = "/home/ayrton/IdeaProjects/projeto_CE_GenAI"
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

def find_all_final_jsons():

    root_dirs = [
        os.path.join(PROJECT_ROOT, "outputs", "evolutionary"),
        os.path.join(PROJECT_ROOT, "outputs", "gridsearch_optuna")
    ]

    json_files = []
    for root_dir in root_dirs:
        print(f"Verificando diretório: {root_dir}")
        if not os.path.exists(root_dir):
            print(f"Diretório não encontrado: {root_dir}")
            continue
        for subdir in glob(os.path.join(root_dir, "*")):
            if not os.path.isdir(subdir):
                continue
            for file in os.listdir(subdir):
                if file.startswith("final_") and file.endswith(".json"):
                    file_path = os.path.join(subdir, file)
                    json_files.append(file_path)
                    print(f"Arquivo JSON encontrado: {file_path}")

    return json_files

def extract_best_params(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    elif not isinstance(data, dict):
        raise ValueError(f"Formato inesperado no arquivo {json_path}")

    model_name = data.get("model") or data.get("Modelo")
    best_params = data.get("best_params") or data.get("Parametros")
    f1 = data.get("f1_holdout") or data.get("Holdout_F1", -1)
    accuracy = data.get("accuracy_holdout") or data.get("Holdout_accuracy", -1)
    precision = data.get("precision_holdout") or data.get("Holdout_precision", -1)
    recall = data.get("recall_holdout") or data.get("Holdout_recall", -1)

    return {
        "model": model_name,
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "params": best_params,
        "origin": "evolutionary" if "evolutionary" in json_path else "gridsearch_optuna",
        "source_file": json_path
    }

def generate_hyperparameters_summary():

    json_files = find_all_final_jsons()
    best_results = {}

    if not json_files:
        print("Nenhum arquivo JSON encontrado. Verifique os caminhos.")
        return

    for json_file in json_files:
        try:
            result = extract_best_params(json_file)
            model = result["model"]
            f1 = result["f1"]

            print(f"Processando {json_file}: Modelo={model}, F1={f1}")

            if model not in best_results or f1 > best_results[model]["f1"]:
                best_results[model] = result
        except Exception as e:
            print(f"Erro ao processar {json_file}: {e}")

    if not best_results:
        print("Nenhum resultado válido encontrado.")
        return

    df = pd.DataFrame(best_results.values())
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "best_hyperparameters_summary.csv")
    json_path = os.path.join(output_dir, "best_hyperparameters_summary.json")

    df.to_csv(csv_path, index=False, sep=";")
    df.to_json(json_path, orient="records", indent=2)

    print("Resumo dos melhores hiperparâmetros salvo em:")
    print(f"- {csv_path}")
    print(f"- {json_path}")

if __name__ == "__main__":
    generate_hyperparameters_summary()
