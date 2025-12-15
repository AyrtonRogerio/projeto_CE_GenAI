import sys
import os
import time
import random
import numpy as np
import pandas as pd
import functools
import warnings
import json
import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count, current_process

# ML Libs
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# DEAP Lib
from deap import base, creator, tools

try:
    import fasttext
    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

try:
    from src.utils.data_loader import load_data
except ImportError:
    print("ERRO: Não foi possível importar src.utils.data_loader. Verifique o PYTHONPATH.")

warnings.filterwarnings("ignore")

class EvolutionaryOptimizer:
    def __init__(self, data_path, output_dir="outputs/evolutionary", seed=None):
        self.data_path = data_path


        if seed is None:
            self.seed = int(time.time())
        else:
            self.seed = seed

        print(f"Inicializando Otimizador com Seed do Algoritmo: {self.seed}")


        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join(project_root, output_dir)
        else:
            self.output_dir = output_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(self.output_dir, f"exp_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)


        random.seed(self.seed)
        np.random.seed(self.seed)

        self._load_and_prep_data()
        self._setup_deap()

    def _load_and_prep_data(self):
        print(f"Carregando dados: {self.data_path}")
        self.df_original = load_data(self.data_path)

        texts = self.df_original['text'].astype(str).str.lower().values
        labels = self.df_original['label'].values

        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(labels)


        print("Dividindo dados com random_state=42 (Fixo para Comparabilidade)...")
        self.X_train_val, self.X_holdout, self.y_train_val, self.y_holdout = train_test_split(
            texts, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

    def _setup_deap(self):
        if hasattr(creator, "FitnessMax"): del creator.FitnessMax
        if hasattr(creator, "Individual"): del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def _get_configs(self):
        configs = [
            {
                "name": "SVM", "type": "sklearn", "cls": SVC,
                "map": {
                    'C': {0: 0.1, 1: 1.0, 2: 10.0, 3: 50.0, 4: 100.0},
                    'kernel': {0: 'linear', 1: 'rbf', 2: 'sigmoid'},
                    'ngram_range': {0: (1, 1), 1: (1, 2)}
                },
                "genes": [("C", 0, 4), ("kernel", 0, 2), ("ngram_range", 0, 1)]
            },
            {
                "name": "NaiveBayes", "type": "sklearn", "cls": MultinomialNB,
                "map": {
                    'alpha': {0: 0.01, 1: 0.1, 2: 0.5, 3: 1.0},
                    'ngram_range': {0: (1, 1), 1: (1, 2)}
                },
                "genes": [("alpha", 0, 3), ("ngram_range", 0, 1)]
            },
            {
                "name": "LogisticRegression", "type": "sklearn", "cls": LogisticRegression,
                "map": {
                    'C': {0: 0.1, 1: 1.0, 2: 10.0, 3: 100.0},
                    'solver': {0: 'liblinear', 1: 'lbfgs'},
                    'ngram_range': {0: (1, 1), 1: (1, 2)}
                },
                "genes": [("C", 0, 3), ("solver", 0, 1), ("ngram_range", 0, 1)]
            }
        ]
        if HAS_FASTTEXT:
            configs.append({
                "name": "FastText", "type": "fasttext",
                "map": {
                    'lr': {0: 0.05, 1: 0.1, 2: 0.25, 3: 0.5, 4: 1.0},
                    'epoch': {0: 5, 1: 15, 2: 25, 3: 50},
                    'wordNgrams': {0: 1, 1: 2, 2: 3},
                    'dim': {0: 50, 1: 100, 2: 300}
                },
                "genes": [("lr", 0, 4), ("epoch", 0, 3), ("wordNgrams", 0, 2), ("dim", 0, 2)]
            })
        return configs

    #AVALIADORES
    def eval_sklearn(self, individual, cls, p_map, cv_splits=5):
        try:
            params, tfidf_params = {}, {}
            for i, val in enumerate(individual):
                p_name = list(p_map.keys())[i]
                if p_name == 'ngram_range':
                    tfidf_params[p_name] = p_map[p_name][val]
                else:
                    params[p_name] = p_map[p_name][val]

            if 'ngram_range' not in tfidf_params: tfidf_params['ngram_range'] = (1, 1)

            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_params)),
                ('clf', cls(**params))
            ])


            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.seed)
            scores = cross_val_score(pipe, self.X_train_val, self.y_train_val, cv=cv, scoring='f1_macro', n_jobs=1)
            return (float(np.mean(scores)),)
        except Exception:
            return (0.0,)

    def eval_fasttext_kfold(self, individual, p_map, cv_splits=5):
        try:
            params = {k: p_map[k][individual[i]] for i, k in enumerate(p_map)}
            skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.seed)
            scores = []
            proc_id = current_process().pid

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train_val, self.y_train_val)):
                ft_train = os.path.join(self.exp_dir, f"tmp_train_{proc_id}_{fold_idx}.txt")
                ft_val = os.path.join(self.exp_dir, f"tmp_val_{proc_id}_{fold_idx}.txt")

                try:
                    self._write_fasttext(self.X_train_val[train_idx], self.y_train_val[train_idx], ft_train)
                    self._write_fasttext(self.X_train_val[val_idx], self.y_train_val[val_idx], ft_val)

                    model = fasttext.train_supervised(input=ft_train, verbose=0, **params)
                    result = model.test(ft_val)

                    p, r = result[1], result[2]
                    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                    scores.append(f1)
                finally:
                    if os.path.exists(ft_train): os.remove(ft_train)
                    if os.path.exists(ft_val): os.remove(ft_val)

            return (float(np.mean(scores)),)
        except Exception:
            return (0.0,)

    def _write_fasttext(self, texts, labels, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for text, label in zip(texts, labels):
                clean_text = text.replace('\n', ' ')
                clean_label = str(label).replace(' ', '_')
                f.write(f"__label__{clean_label} {clean_text}\n")

    def run(self, n_gen=100, pop_size=100, cx_prob=0.8, mut_prob=0.2):
        configs = self._get_configs()
        summary_results = []

        start_global = time.time()


        total_cores = max(1, cpu_count() - 2)

        for cfg in configs:
            model_name = cfg['name']


            if cfg['type'] == 'fasttext':

                n_workers = min(6, total_cores)
                print(f"\nOtimizando {model_name} (Modo RAM-SAFE: {n_workers} processos)")
            else:

                n_workers = total_cores
                print(f"\nOtimizando {model_name} (Modo TURBO: {n_workers} processos)")

            pool = Pool(processes=n_workers)

            try:
                model_dir = os.path.join(self.exp_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)

                toolbox = base.Toolbox()
                toolbox.register("map", pool.map)

                gene_funcs = []
                for _, min_v, max_v in cfg['genes']:
                    gene_funcs.append(functools.partial(random.randint, min_v, max_v))

                toolbox.register("individual", tools.initCycle, creator.Individual, gene_funcs, n=1)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)

                if cfg['type'] == 'sklearn':
                    toolbox.register("evaluate", functools.partial(self.eval_sklearn, cls=cfg['cls'], p_map=cfg['map']))
                else:
                    toolbox.register("evaluate", functools.partial(self.eval_fasttext_kfold, p_map=cfg['map']))

                toolbox.register("mate", tools.cxTwoPoint)
                toolbox.register("mutate", functools.partial(tools.mutUniformInt, low=[g[1] for g in cfg['genes']], up=[g[2] for g in cfg['genes']], indpb=0.2))
                toolbox.register("select", tools.selTournament, tournsize=3)

                pop = toolbox.population(n=pop_size)
                hof = tools.HallOfFame(1)
                stats = tools.Statistics(lambda ind: ind.fitness.values)
                stats.register("avg", np.mean)
                stats.register("max", np.max)

                logbook = tools.Logbook()

                # Inicialização
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                hof.update(pop)

                # Loop Evolutivo
                for gen in range(1, n_gen + 1):
                    offspring = toolbox.select(pop, len(pop))
                    offspring = list(map(toolbox.clone, offspring))

                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < cx_prob:
                            toolbox.mate(child1, child2)
                            del child1.fitness.values, child2.fitness.values

                    for mutant in offspring:
                        if random.random() < mut_prob:
                            toolbox.mutate(mutant)
                            del mutant.fitness.values

                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                    pop[:] = offspring
                    pop.extend(list(map(toolbox.clone, hof)))
                    pop = tools.selBest(pop, pop_size)
                    hof.update(pop)

                    record = stats.compile(pop)
                    logbook.record(gen=gen, **record)

                    best_fit = hof[0].fitness.values[0]
                    print(f"[{model_name}] Gen {gen}: Max F1: {best_fit:.4f}")

                    # Checkpoint
                    pd.DataFrame(logbook).to_csv(os.path.join(model_dir, "log.csv"), index=False)

                # Salvar final
                best_ind = hof[0]
                best_params = {k: cfg['map'][k][best_ind[i]] for i, k in enumerate(cfg['map'])}
                holdout_score = self._final_evaluate_holdout(cfg, best_params)

                result_entry = {
                    "Modelo": model_name,
                    "Parametros": best_params,
                    "Holdout_F1": holdout_score,
                    "runtime_seconds": time.time() - start_global
                }
                summary_results.append(result_entry)

                with open(os.path.join(self.exp_dir, f"final_evaluation_{model_name}.json"), 'w') as f:
                    json.dump(result_entry, f, indent=2)

            finally:
                pool.close()
                pool.join()

        df_results = pd.DataFrame(summary_results)
        df_results.to_csv(os.path.join(self.exp_dir, "final_results_all_models.csv"), index=False, sep=';')

        with open(os.path.join(self.exp_dir, "final_results_all_models.json"), 'w') as f:
            json.dump(summary_results, f, indent=2)

        print(f"Resultados salvos em: {self.exp_dir}")

    def _final_evaluate_holdout(self, cfg, params):
        if cfg['type'] == 'sklearn':
            tfidf_params = {k: v for k, v in params.items() if k == 'ngram_range'}
            clf_params = {k: v for k, v in params.items() if k != 'ngram_range'}
            if 'ngram_range' not in tfidf_params: tfidf_params['ngram_range'] = (1,1)

            pipe = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)), ('clf', cfg['cls'](**clf_params))])
            pipe.fit(self.X_train_val, self.y_train_val)
            preds = pipe.predict(self.X_holdout)
            return f1_score(self.y_holdout, preds, average='macro')
        else:
            ft_train_full = os.path.join(self.exp_dir, "full_train.txt")
            self._write_fasttext(self.X_train_val, self.y_train_val, ft_train_full)
            model = fasttext.train_supervised(input=ft_train_full, verbose=0, **params)
            preds = []
            for txt in self.X_holdout:
                lbl = model.predict(txt)[0][0].replace('__label__', '')
                preds.append(int(lbl) if lbl.isdigit() else 0)
            if os.path.exists(ft_train_full): os.remove(ft_train_full)
            return f1_score(self.y_holdout, preds, average='macro')

if __name__ == "__main__":
    DATASET = os.path.join(project_root, "data", "raw", "ouvidoria_sintetico.csv")
    if os.path.exists(DATASET):

        opt = EvolutionaryOptimizer(DATASET, seed=None)

        opt.run(
            n_gen=100,
            pop_size=100,
            cx_prob=0.8,
            mut_prob=0.2
        )
    else:
        print(f"Dataset não encontrado: {DATASET}")