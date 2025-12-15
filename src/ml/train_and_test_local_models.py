import os
import json
import time
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import fasttext
from sklearn.model_selection import train_test_split

from src.utils.data_loader import load_data


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RECON_PATH = os.path.join(PROJECT_ROOT, "outputs", "reconstructed_models.json")

LOCAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "local_models")
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(LOCAL_MODELS_DIR, "summary.csv")
ALL_PRED_PATH = os.path.join(LOCAL_MODELS_DIR, "all_predictions_long.csv")


def build_sklearn_model(model_name, params):
    

    vectorizer = TfidfVectorizer()

    if model_name == "SVM":
        clf = LinearSVC(C=params.get("C", 1.0))
    elif model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=2000, C=params.get("C", 1.0))
    elif model_name == "NaiveBayes":
        clf = MultinomialNB(alpha=params.get("alpha", 1.0))
    else:
        return None

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])


def train_fasttext_model(df_train, df_test, params, save_dir):
    

    os.makedirs(save_dir, exist_ok=True)

    train_txt = os.path.join(save_dir, "train.txt")
    test_txt = os.path.join(save_dir, "test.txt")

    
    df_train.apply(lambda x: f"__label__{x['label']} {x['text']}", axis=1).to_csv(train_txt, sep="\n", index=False)
    df_test.apply(lambda x: f"__label__{x['label']} {x['text']}", axis=1).to_csv(test_txt, sep="\n", index=False)

    model = fasttext.train_supervised(
        input=train_txt,
        lr=float(params.get("lr", 1.0)),
        epoch=int(params.get("epoch", 50)),
        wordNgrams=int(params.get("wordNgrams", 1)),
        dim=int(params.get("dim", 100))
    )

    model_path = os.path.join(save_dir, "fasttext_model.bin")
    model.save_model(model_path)

    # Avaliação
    preds = []
    for t in df_test["text"].tolist():
        p = model.predict(t)[0][0].replace("__label__", "")
        preds.append(p)

    f1 = f1_score(df_test["label"], preds, average="macro")
    acc = accuracy_score(df_test["label"], preds)

    return f1, acc, preds


def run_training():

    print("Modelos reconstruídos carregados...", flush=True)

    if not os.path.exists(RECON_PATH):
        raise FileNotFoundError("reconstructed_models.json não encontrado. Rode rebuild_models_from_json.py.")

    with open(RECON_PATH, "r") as f:
        reconstructed = json.load(f)

    all_models = reconstructed.get("evolutionary_models", []) + reconstructed.get("grid_optuna_models", [])

    print(f"Modelos encontrados: {len(all_models)}")

    
    df = load_data(os.path.join(PROJECT_ROOT, "data", "raw", "ouvidoria_sintetico.csv"))
    print("Dataset carregado:", len(df), "linhas\n")

    # Divisão do dataset
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])

    summary_rows = []
    pred_rows = []

    
    for model_dict in all_models:

        run_id = model_dict.get("run_id")
        model_name = model_dict.get("model")
        origin = model_dict.get("origin")
        params = model_dict.get("params", {})

        print(f"{run_id} | modelo: {model_name}")

        save_dir = os.path.join(LOCAL_MODELS_DIR, model_name, f"{run_id}__{model_name}__{origin}")
        os.makedirs(save_dir, exist_ok=True)

        start = time.time()

        
        if model_name in ["SVM", "LogisticRegression", "NaiveBayes", "LinearSVC"]:

            model = build_sklearn_model(model_name, params)
            if model is None:
                print("Modelo inválido:", model_name)
                continue

            model.fit(df_train["text"], df_train["label"])
            preds = model.predict(df_test["text"])

            f1 = f1_score(df_test["label"], preds, average="macro")
            acc = accuracy_score(df_test["label"], preds)

            
            import joblib
            joblib.dump(model, os.path.join(save_dir, f"{model_name}.pkl"))

        
        elif model_name == "FastText":

            f1, acc, preds = train_fasttext_model(df_train, df_test, params, save_dir)

        else:
            print("Modelo desconhecido:", model_name)
            continue

        end = time.time()

        print(f"{model_name} treinado. F1_holdout={f1:.4f}, ACC_holdout={acc:.4f}\n")

        # Resumo
        summary_rows.append({
            "run_id": run_id,
            "modelo": model_name,
            "origem": origin,
            "f1_holdout": f1,
            "acc_holdout": acc,
            "treino_segundos": round(end - start, 3)
        })

        # Predições detalhadas
        for idx, (true, pred, text) in enumerate(zip(df_test["label"], preds, df_test["text"])):
            pred_rows.append({
                "run_id": run_id,
                "model": model_name,
                "origin": origin,
                "index": idx,
                "text": text,
                "true_label": true,
                "pred_label": pred
            })


    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(pred_rows).to_csv(ALL_PRED_PATH, index=False)

    print(f"[FIN] Resumo salvo em: {SUMMARY_PATH}")
    print(f"[FIN] Predições detalhadas salvas em: {ALL_PRED_PATH}")


if __name__ == "__main__":
    run_training()
