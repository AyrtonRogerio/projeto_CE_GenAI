import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from train_and_test_local_models import (
    build_sklearn_model, train_sklearn, train_fasttext
)
from src.utils.data_loader import load_data

ROOT = Path(".")
BEST_PARAMS_FILE = ROOT / "outputs/local_retrained/best_hyperparams_all.json"
OUTPUT_DIR = ROOT / "outputs/local_retrained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Carregando dataset...")
    df = load_data("data/raw/ouvidoria_sintetico.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.label, stratify=df.label,
        test_size=0.2, random_state=42
    )

    df_train = pd.DataFrame({"text": X_train, "label": y_train})
    df_test = pd.DataFrame({"text": X_test, "label": y_test})

    print("Lendo hiperparâmetros...")
    best = json.load(open(BEST_PARAMS_FILE))

    summary_rows = []

    for model_name, runs in best.items():
        for entry in runs:
            params = entry["params"]
            run_id = Path(entry["run_folder"]).name

            out = OUTPUT_DIR / model_name / run_id
            out.mkdir(parents=True, exist_ok=True)

            print(f"\n[TRAIN] {model_name} | run {run_id}")

            if model_name == "FastText":
                model, preds, metrics = train_fasttext(params, df_train, df_test, out)
            else:
                model = build_sklearn_model(model_name, params)
                model, preds, metrics = train_sklearn(
                    model, df_train.text, df_train.label,
                    df_test.text, df_test.label
                )

                import joblib
                joblib.dump(model, out / "model.pkl")

            pd.DataFrame({
                "text": df_test.text,
                "true": df_test.label,
                "pred": preds
            }).to_csv(out / "predictions.csv", index=False)

            with open(out / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            summary_rows.append({
                "model": model_name,
                "run": run_id,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "train_time": metrics["train_time"],
                "infer_time": metrics["infer_time"],
            })

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print("\nTreinamento concluído — summary.csv gerado.")

if __name__ == "__main__":
    main()
