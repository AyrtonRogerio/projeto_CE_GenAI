import os
import sys
import json
import time
import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC

try:
    import fasttext
except ImportError:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(ROOT)

from src.utils.data_loader import load_data
from src.ml.rebuild_models_from_json import aggregate_all_models

OUTPUT_DIR = os.path.join(ROOT, "outputs", "ml_benchmark_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)



def train_predict_sklearn(model_name, params, X_train, y_train, X_test):

    tfidf_p = {}
    clf_p_raw = {}

    tfidf_keys = ['ngram_range', 'min_df', 'max_df', 'use_idf', 'binary', 'stop_words', 'norm']

    for k, v in params.items():
        clean_key = k.split('__')[-1]
        if clean_key in tfidf_keys:
            tfidf_p[clean_key] = v
        else:
            clf_p_raw[clean_key] = v


    if 'ngram_range' in tfidf_p:
        if isinstance(tfidf_p['ngram_range'], list):
            tfidf_p['ngram_range'] = tuple(tfidf_p['ngram_range'])
    else:
        tfidf_p['ngram_range'] = (1, 1)



    clf = None

    if model_name == "LogisticRegression":
        valid_lr = {'C', 'solver', 'penalty', 'max_iter', 'tol', 'class_weight', 'random_state', 'n_jobs'}
        lr_kwargs = {k: v for k, v in clf_p_raw.items() if k in valid_lr}
        clf = LogisticRegression(**lr_kwargs, max_iter=1000)

    elif "NaiveBayes" in model_name or model_name == "MultinomialNB":
        valid_nb = {'alpha', 'fit_prior', 'class_prior'}
        nb_kwargs = {k: v for k, v in clf_p_raw.items() if k in valid_nb}
        clf = MultinomialNB(**nb_kwargs)

    elif "SVM" in model_name or model_name == "LinearSVC":

        use_svc_class = False
        if 'kernel' in clf_p_raw:
            if clf_p_raw['kernel'] != 'linear':
                use_svc_class = True

        if use_svc_class:
            valid_svc = {'C', 'kernel', 'degree', 'gamma', 'coef0', 'probability', 'tol', 'class_weight', 'random_state'}
            svc_kwargs = {k: v for k, v in clf_p_raw.items() if k in valid_svc}
            clf = SVC(**svc_kwargs)
        else:

            valid_linear = {'C', 'penalty', 'loss', 'dual', 'tol', 'multi_class', 'fit_intercept', 'class_weight', 'random_state', 'max_iter'}
            linear_kwargs = {k: v for k, v in clf_p_raw.items() if k in valid_linear}
            clf = LinearSVC(**linear_kwargs, dual='auto')

    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")


    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_p)),
        ('clf', clf)
    ])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    t_train = time.time() - t0

    t1 = time.time()
    preds = pipe.predict(X_test)
    t_inf_total = time.time() - t1

    return preds, t_train, (t_inf_total / len(X_test))

def train_predict_fasttext(params, X_train, y_train, X_test, model_dir):
    ft_train = os.path.join(model_dir, "ft_final_train.txt")
    with open(ft_train, 'w', encoding='utf-8') as f:
        for t, l in zip(X_train, y_train):
            f.write(f"__label__{l} {t}\n")


    valid_ft = {'lr', 'dim', 'ws', 'epoch', 'minCount', 'minCountLabel', 'minn', 'maxn', 'neg', 'wordNgrams', 'loss', 'bucket', 'thread', 'lrUpdateRate', 't'}
    ft_kwargs = {k: v for k, v in params.items() if k in valid_ft}

    t0 = time.time()
    model = fasttext.train_supervised(input=ft_train, verbose=0, **ft_kwargs)
    t_train = time.time() - t0

    preds = []
    t1 = time.time()
    for txt in X_test:
        lbl = model.predict(txt)[0][0].replace('__label__', '')
        preds.append(lbl)
    t_inf_total = time.time() - t1

    if os.path.exists(ft_train): os.remove(ft_train)
    return preds, t_train, (t_inf_total / len(X_test))


def get_best_models_per_type(all_models_data):
    best_candidates = {}

    print("\nDisputa de Estratégias (Torneio)")
    for key, data in all_models_data.items():
        m_type = data['model_type']
        score = float(data['score'])
        origin = data['origin']

        # print(f"Candidato: {m_type} | Origem: {origin} | F1: {score:.4f}")

        if m_type not in best_candidates or score > best_candidates[m_type]['score']:
            best_candidates[m_type] = {
                "params": data['params'],
                "origin": origin,
                "score": score,
                "source_key": key
            }
    return best_candidates

def main():
    print("Iniciando Benchmark Final de ML")
    data_path = os.path.join(ROOT, "data/raw/ouvidoria_sintetico.csv")
    df = load_data(data_path)

    df['demanda_id'] = [f"{i+1:04d}" for i in range(len(df))]

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    le = LabelEncoder()
    le.fit(df['label'])
    y_train_enc = le.transform(train_df['label'])
    y_train_ft = [str(x).replace(' ', '_') for x in train_df['label']]

    # 1. Carregar dados
    all_data = aggregate_all_models()

    # 2. Selecionar campeões
    champions = get_best_models_per_type(all_data)

    print("-" * 40)
    print("VENCEDORES FINAIS SELECIONADOS:")
    for m, info in champions.items():
        print(f"-> {m}: F1={info['score']:.4f} (Vencedor: {info['origin']})")
    print("-" * 40)

    final_results = []

    for model_name, info in champions.items():
        print(f"\nTreinando: {model_name}...")
        params = info['params']

        try:
            if model_name == "FastText":
                preds, t_train, t_inf = train_predict_fasttext(params, train_df['text'].values, y_train_ft, test_df['text'].values, OUTPUT_DIR)
                preds_decoded = preds
            else:
                preds, t_train, t_inf = train_predict_sklearn(model_name, params, train_df['text'], y_train_enc, test_df['text'])
                preds_decoded = le.inverse_transform(preds)

            for i, (idx, row) in enumerate(test_df.iterrows()):
                pred = str(preds_decoded[i]).replace('_', ' ')
                real = str(row['label'])
                final_results.append({
                    "demanda_id": row['demanda_id'],
                    "classificacao_real": real,
                    "prompt_id": f"ML_{info['origin']}",
                    "modelo_id": model_name,
                    "classificacao_limpa": pred,
                    "acertou": 1 if pred.lower() == real.lower() else 0,
                    "demanda_texto": row['text'],
                    "classificacao_raw": pred,
                    "tempo_resposta_s": t_inf,
                    "erro": None,
                    "try_count": 1
                })
        except Exception as e:
            print(f"ERRO em {model_name}: {e}")
            traceback.print_exc()

    out_csv = os.path.join(OUTPUT_DIR, "ml_benchmark_results.csv")
    pd.DataFrame(final_results).to_csv(out_csv, sep='|', index=False, quoting=1)
    print(f"\nBenchmark salvo em: {out_csv}")

if __name__ == "__main__":
    main()