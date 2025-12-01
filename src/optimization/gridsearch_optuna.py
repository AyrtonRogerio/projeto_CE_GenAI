import sys
import os
import time
import json
import re
import random
import numpy as np
import pandas as pd
import nltk
import spacy
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

from multiprocessing import Pool, cpu_count

import optuna


try:
    import fasttext
    HAS_FASTTEXT = True
except Exception:
    HAS_FASTTEXT = False


try:
    from src.utils.data_loader import load_data
except Exception:
    try:
        from utils.data_loader import load_data
    except Exception:
        load_data = None

warnings.filterwarnings("ignore")


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


try:
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
except Exception:
    nlp = None


class GridSearchOptunaRunner:
    def __init__(self, data_path, output_dir="outputs/gridsearch_optuna", seed=42, n_folds=5):
        self.data_path = data_path
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.n_folds = n_folds
        self.results = []
        self.results_test = []
        self.error_log = []


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isabs(output_dir):

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            self.run_dir = os.path.join(project_root, output_dir, f"run_{timestamp}")
        else:
            self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)


        self.ft_train_file = os.path.join(self.run_dir, "fasttext_train.txt")
        self.ft_val_file = os.path.join(self.run_dir, "fasttext_val.txt")

        # stopwords
        try:
            self.stopwords_pt = nltk.corpus.stopwords.words('portuguese')
        except Exception:
            self.stopwords_pt = []

        # load data and prepare
        self._load_and_prep_data()
        self._make_train_holdout_split()

    # preprocessamento
    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.strip().lower()
        if nlp:
            doc = nlp(text)
            lemmas = [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in self.stopwords_pt and len(t.lemma_) > 2]
            return " ".join(lemmas)
        return re.sub(r'[^\w\s]', '', text)

    # carregando dados
    def _load_and_prep_data(self):
        if load_data is None:
            raise RuntimeError("load_data function not found. Adicione src/utils/data_loader.py ou utils/data_loader.py ao projeto.")
        print(f"Carregando dados de: {self.data_path}")
        df = load_data(self.data_path)
        if df is None or df.empty:
            raise ValueError("Dataset vazio ou não encontrado.")

        print("Pré-processando textos...")
        df['text_processed'] = df['text'].astype(str).apply(self._preprocess_text)
        df = df[df['text_processed'].str.len() > 0].copy()
        df.reset_index(drop=True, inplace=True)

        # save processed snapshot
        try:
            df.to_csv(os.path.join(self.run_dir, "processed_data.csv"), sep=';', index=False)
        except Exception:
            pass

        self.X_raw = df['text_processed']
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(df['label'])

        # record meta
        self._save_json({
            "n_samples": int(len(df)),
            "n_classes": int(len(self.le.classes_)),
            "classes": list(self.le.classes_)
        }, "data_snapshot.json")


    def _make_train_holdout_split(self):
        self.X_train_val, self.X_holdout, self.y_train_val, self.y_holdout = train_test_split(
            self.X_raw, self.y, test_size=0.2, stratify=self.y, random_state=self.seed
        )


    def _save_json(self, data, fname):
        try:
            with open(os.path.join(self.run_dir, fname), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.error_log.append(f"Erro salvando json {fname}: {e}")

    def _append_train_result(self, rec):
        self.results.append(rec)
        try:
            pd.DataFrame(self.results).to_csv(os.path.join(self.run_dir, "performance_train.csv"), sep=';', index=False)
        except Exception:
            pass

    def _append_test_result(self, rec):
        self.results_test.append(rec)
        try:
            pd.DataFrame(self.results_test).to_csv(os.path.join(self.run_dir, "performance_test.csv"), sep=';', index=False)
        except Exception:
            pass


    def run_classic_models_gridsearch(self, n_jobs=-1):
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        models_config = [
            {
                "name": "LogisticRegression",
                "pipeline": Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression(max_iter=1000, random_state=self.seed))]),
                "params": {'tfidf__ngram_range': [(1,1),(1,2)], 'clf__C': [0.1,1,10]}
            },
            {
                "name": "MultinomialNB",
                "pipeline": Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())]),
                "params": {'tfidf__ngram_range': [(1,1),(1,2)], 'clf__alpha': [0.1,0.5,1.0]}
            },
            {
                "name": "LinearSVC",
                "pipeline": Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC(dual='auto', random_state=self.seed))]),
                "params": {'tfidf__ngram_range': [(1,1),(1,2)], 'clf__C': [0.1,1,10]}
            }
        ]

        for cfg in models_config:
            model_name = cfg['name']
            t0 = time.time()
            print(f"[GridSearch] Iniciando {model_name} ...")
            try:
                grid = GridSearchCV(cfg['pipeline'], cfg['params'], cv=cv, scoring='f1_macro', n_jobs=n_jobs, verbose=1)
                grid.fit(self.X_train_val, self.y_train_val)

                rec_train = {
                    "model": model_name,
                    "algorithm": "GridSearchCV",
                    "best_f1_train": float(grid.best_score_),
                    "best_params": grid.best_params_,
                    "runtime_seconds": round(time.time()-t0, 2)
                }
                self._append_train_result(rec_train)


                best_model = grid.best_estimator_
                best_model.fit(self.X_train_val, self.y_train_val)

                preds = best_model.predict(self.X_holdout)
                f1 = f1_score(self.y_holdout, preds, average='macro')
                acc = accuracy_score(self.y_holdout, preds)
                try:
                    cr = classification_report(self.y_holdout, preds, target_names=list(self.le.classes_), output_dict=True)
                except Exception:
                    cr = None

                rec_test = {
                    "model": model_name,
                    "algorithm": "GridSearchCV",
                    "best_f1_train": float(grid.best_score_),
                    "f1_holdout": float(f1),
                    "accuracy_holdout": float(acc),
                    "best_params": grid.best_params_,
                    "runtime_seconds": round(time.time()-t0, 2)
                }
                self._append_test_result(rec_test)


                try:
                    self._save_json(rec_test, f"final_{model_name}_gridsearch.json")
                    if cr is not None:
                        self._save_json(cr, f"classification_report_{model_name}.json")
                except Exception as e:
                    self.error_log.append(f"Erro salvando final json/report {model_name}: {e}")

                print(f"[GridSearch] {model_name} done. Holdout F1: {f1:.4f}")

            except Exception as e:
                self.error_log.append(f"Erro GridSearch {model_name}: {e}")
                print(f"[GridSearch] Erro em {model_name}: {e}")

    # FastText + Optuna
    def run_fasttext_optuna(self, n_trials=20):
        if not HAS_FASTTEXT:
            print("[FastText] fasttext não encontrado, pulando FastText.")
            return

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        X_arr = np.array(self.X_train_val)
        y_arr = np.array(self.y_train_val)

        def write_fasttext_file(path, texts, labels):
            with open(path, 'w', encoding='utf-8') as f:
                for t, l in zip(texts, labels):
                    t = str(t).replace('\n',' ')
                    l = str(l).replace(' ','_')
                    f.write(f"__label__{l} {t}\n")

        def eval_params(params):
            fold_scores = []
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_arr)):
                tr_texts = X_arr[tr_idx]
                tr_labels = self.le.inverse_transform(y_arr[tr_idx])
                va_texts = X_arr[va_idx]
                va_labels = self.le.inverse_transform(y_arr[va_idx])

                train_path = os.path.join(self.run_dir, f"ft_fold{fold}_train.txt")
                val_path = os.path.join(self.run_dir, f"ft_fold{fold}_val.txt")
                write_fasttext_file(train_path, tr_texts, tr_labels)
                write_fasttext_file(val_path, va_texts, va_labels)

                try:
                    model = fasttext.train_supervised(input=train_path, verbose=0, **params)
                except Exception as e:
                    self.error_log.append(f"FastText train error fold {fold}: {e} -- params: {params}")
                    try:
                        os.remove(train_path)
                        os.remove(val_path)
                    except Exception:
                        pass
                    return 0.0

                preds = []
                for txt in va_texts:
                    try:
                        lbl = model.predict(txt)[0][0]
                        preds.append(lbl.replace('__label__', '').replace('_',' '))
                    except Exception:
                        preds.append(self.le.classes_[0])

                try:
                    y_pred_enc = self.le.transform(preds)
                    y_true_enc = y_arr[va_idx]
                    fold_scores.append(f1_score(y_true_enc, y_pred_enc, average='macro'))
                except Exception as e:
                    self.error_log.append(f"FastText eval error fold {fold}: {e}")
                    fold_scores.append(0.0)

                # cleanup fold files
                try:
                    os.remove(train_path)
                    os.remove(val_path)
                except Exception:
                    pass

            return float(np.mean(fold_scores)) if fold_scores else 0.0

        def objective(trial):
            params = {
                "lr": trial.suggest_float('lr', 0.01, 0.5, log=True),
                "dim": trial.suggest_categorical('dim', [50,100,200,300]),
                "epoch": trial.suggest_int('epoch', 5, 30),
                "wordNgrams": trial.suggest_int('wordNgrams', 1, 3)
            }
            return eval_params(params)

        print("[Optuna] Iniciando Optuna para FastText...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_score = float(study.best_value)

        rec_train = {
            "model": "FastText",
            "algorithm": "Optuna",
            "best_f1_train": best_score,
            "best_params": best_params
        }
        self._append_train_result(rec_train)


        write_fasttext_file(self.ft_train_file, self.X_train_val, self.le.inverse_transform(self.y_train_val))
        try:
            final_model = fasttext.train_supervised(input=self.ft_train_file, verbose=0, **best_params)
        except Exception as e:
            self.error_log.append(f"FastText final train error: {e}")
            return

        preds = []
        for txt in self.X_holdout:
            try:
                lbl = final_model.predict(txt)[0][0]
                preds.append(lbl.replace('__label__','').replace('_',' '))
            except Exception:
                preds.append(self.le.classes_[0])

        try:
            y_pred_enc = self.le.transform(preds)
        except Exception:
            y_pred_enc = np.array([0]*len(preds))

        f1 = f1_score(self.y_holdout, y_pred_enc, average='macro')
        acc = accuracy_score(self.y_holdout, y_pred_enc)
        try:
            cr = classification_report(self.y_holdout, y_pred_enc, target_names=list(self.le.classes_), output_dict=True)
        except Exception:
            cr = None

        rec_test = {
            "model": "FastText",
            "algorithm": "Optuna",
            "best_f1_train": best_score,
            "f1_holdout": float(f1),
            "accuracy_holdout": float(acc),
            "best_params": best_params
        }
        self._append_test_result(rec_test)


        try:
            self._save_json(rec_test, "final_FastText_optuna.json")
            if cr is not None:
                self._save_json(cr, "classification_report_FastText.json")
        except Exception as e:
            self.error_log.append(f"Erro salvando FastText final json/report: {e}")


        try:
            os.remove(self.ft_train_file)
        except Exception:
            pass

        print(f"[Optuna] FastText final holdout F1: {f1:.4f}")

    # save summary
    def save_results(self):

        try:
            if self.results_test:
                df = pd.DataFrame(self.results_test)
                fcol = 'f1_holdout' if 'f1_holdout' in df.columns else 'best_f1_train'
                best_idx = df[fcol].astype(float).idxmax()
                best_row = df.loc[best_idx].to_dict()
                self._save_json(best_row, 'best_model.json')
        except Exception as e:
            self.error_log.append(f"Erro salvando best_model.json: {e}")


        if self.error_log:
            try:
                with open(os.path.join(self.run_dir, "error_log.txt"), 'w', encoding='utf-8') as f:
                    for e in self.error_log:
                        f.write(str(e) + "\n")
            except Exception:
                pass


    def run_all(self, grid_n_jobs=-1, fasttext_n_trials=20):
        self.run_classic_models_gridsearch(n_jobs=grid_n_jobs)
        self.run_fasttext_optuna(n_trials=fasttext_n_trials)
        self.save_results()


if __name__ == "__main__":

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATASET_PATH_DEFAULT = os.path.join(PROJECT_ROOT, "data", "raw", "ouvidoria_sintetico.csv")


    dataset_arg = None
    if len(sys.argv) > 1:
        dataset_arg = sys.argv[1]
    DATASET_PATH = dataset_arg or os.environ.get("OUVIDORIA_DATASET_PATH") or DATASET_PATH_DEFAULT

    if os.path.exists(DATASET_PATH):
        runner = GridSearchOptunaRunner(data_path=DATASET_PATH, seed=None, n_folds=5)
        runner.run_all(grid_n_jobs=max(1, cpu_count() - 2), fasttext_n_trials=100)
        print("Run finished. Results in:", runner.run_dir)
    else:
        print(f"Arquivo nao encontrado: {DATASET_PATH}")
        sys.exit(0)
