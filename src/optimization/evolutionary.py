import sys
import os

#CONFIGURACAO DE CAMINHO
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


import time
import random
import numpy as np
import pandas as pd
import functools
import warnings
import shutil
import json
import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ML Lib
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

try:
    import fasttext
    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False

from deap import base, creator, tools, algorithms
# importa seletores do DEAP
from deap.tools import selTournament, selStochasticUniversalSampling, selRoulette, selBest

# Utils
try:
    from src.utils.data_loader import load_data
except ImportError:
    from utils.data_loader import load_data

warnings.filterwarnings("ignore")


def selRank(individuals, k):

    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness.values[0], reverse=True)
    ranks = np.arange(len(sorted_inds), 0, -1)
    probs = ranks / ranks.sum()
    return random.choices(sorted_inds, weights=probs, k=k)


def selTournament3(individuals, k):
    return selTournament(individuals, k, tournsize=3)


def selTournament5(individuals, k):
    return selTournament(individuals, k, tournsize=5)


SELECTION_METHODS = {
    "tournament": selTournament3,
    "tournament5": selTournament5,
    "sus": selStochasticUniversalSampling,
    "roulette": selRoulette,
    "best": selBest,
    "rank": selRank
}


class EvolutionaryOptimizer:


    def __init__(self, data_path, output_dir="outputs/evolutionary", seed=None):
        self.data_path = data_path

        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join(project_root, output_dir)
        else:
            self.output_dir = output_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(self.output_dir, f"exp_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        self.df_original = None
        self.X_train_val = None
        self.y_train_val = None
        self.X_holdout = None
        self.y_holdout = None

        self.ft_train_file = os.path.join(self.exp_dir, "fasttext_train.txt")
        self.ft_val_file = os.path.join(self.exp_dir, "fasttext_val.txt")
        self.ft_test_file = os.path.join(self.exp_dir, "fasttext_test.txt")

        self.le = LabelEncoder()
        self._setup_deap()

        self.error_log = []
        self.run_config = {}
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._load_and_prep_data()

    def _load_and_prep_data(self):
        print(f"Carregando dados de: {self.data_path}")
        self.df_original = load_data(self.data_path)

        if self.df_original is None or self.df_original.empty:
            raise ValueError("Erro: Dataset vazio ou nao encontrado.")

        texts = self.df_original['text'].astype(str).str.lower().values
        labels = self.df_original['label'].values

        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(labels)

        # Split Estratificado (80% Treino/Validação - 20% Holdout Final)
        X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
            texts, y_encoded, test_size=0.2, stratify=y_encoded, random_state=self.seed
        )

        self.X_train_val = X_train_val
        self.y_train_val = y_train_val
        self.X_holdout = X_holdout
        self.y_holdout = y_holdout

        if HAS_FASTTEXT:
            X_ft_train, X_ft_val, y_ft_train, y_ft_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=self.seed
            )
            self._save_fasttext_file(X_ft_train, y_ft_train, self.ft_train_file)
            self._save_fasttext_file(X_ft_val, y_ft_val, self.ft_val_file)
            self._save_fasttext_file(X_holdout, y_holdout, self.ft_test_file)

    def _save_fasttext_file(self, texts, labels, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for text, label in zip(texts, labels):
                clean_text = text.replace('\n', ' ')
                clean_label = str(label).replace(' ', '_')
                f.write(f"__label__{clean_label} {clean_text}\n")

    def _setup_deap(self):
        # Limpa classes criadas anteriormente para evitar conflito em loops
        for name in ("FitnessMax", "Individual"):
            if hasattr(creator, name):
                try:
                    delattr(creator, name)
                except Exception:
                    pass

        # 1. Define o objetivo: Maximizar (peso 1.0)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # 2. Define o Indivíduo: Uma lista de genes com o atributo fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def _get_configs(self, include_fasttext=True):
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
        if HAS_FASTTEXT and include_fasttext:
            configs.append({
                "name": "FastText", "type": "fasttext",
                "map": {
                    'lr': {0: 0.01, 1: 0.05, 2: 0.1, 3: 0.5, 4: 1.0},
                    'epoch': {0: 5, 1: 10, 2: 25, 3: 50},
                    'wordNgrams': {0: 1, 1: 2, 2: 3},
                    'dim': {0: 50, 1: 100, 2: 300}
                },
                "genes": [("lr", 0, 4), ("epoch", 0, 3), ("wordNgrams", 0, 2), ("dim", 0, 2)]
            })
        return configs

    def eval_sklearn(self, individual, cls, p_map, cv_splits=5):
        try:
            params, tfidf_params = {}, {}
            for i, val in enumerate(individual):
                p_name = list(p_map.keys())[i]
                if p_name in ['ngram_range', 'max_df', 'min_df']:
                    tfidf_params[p_name] = p_map[p_name][val]
                else:
                    params[p_name] = p_map[p_name][val]

            tfidf_params_effective = tfidf_params.copy()
            if 'ngram_range' not in tfidf_params_effective:
                tfidf_params_effective['ngram_range'] = (1, 1)

            pipe = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params_effective)), ('clf', cls(**params))])

            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.seed)
            scores = cross_val_score(pipe, self.X_train_val, self.y_train_val, cv=cv, scoring='f1_macro', n_jobs=1)
            mean_score = float(np.mean(scores))
            return (mean_score,)
        except Exception as e:
            msg = f"Erro eval_sklearn: {e} -- individual: {individual}"
            self.error_log.append(msg)
            return (-1.0,)

    def eval_fasttext(self, individual, p_map):
        try:
            params = {k: p_map[k][individual[i]] for i, k in enumerate(p_map)}
            model = fasttext.train_supervised(input=self.ft_train_file, verbose=0, **params)
            result = model.test(self.ft_val_file)

            precision = result[1]
            recall = result[2]
            if (precision + recall) == 0:
                return (0.0,)
            f1 = 2 * (precision * recall) / (precision + recall)
            return (f1,)
        except Exception as e:
            msg = f"Erro eval_fasttext: {e} -- individual: {individual}"
            self.error_log.append(msg)
            return (-1.0,)

    def _individual_to_params(self, individual, p_map):
        return {k: p_map[k][individual[i]] for i, k in enumerate(p_map)}

    def _population_diversity(self, population):
        try:
            arr = np.array(population, dtype=float)
            gene_std = np.std(arr, axis=0)
            return float(np.mean(gene_std))
        except Exception:
            return 0.0

    def _save_json(self, data, filename):
        with open(os.path.join(self.exp_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def run(self,
            n_gen=10,
            pop_size=20,
            cx_op="two_point",
            mut_op="uniform_int",
            sel_op="tournament",
            cx_prob=0.5,
            mut_prob=0.2,
            elitism_size=1,
            max_time=None,
            cv_splits=5,
            n_jobs=None):


        # Salva configurações
        self.run_config = {
            "n_gen": n_gen,
            "pop_size": pop_size,
            "cx_op": cx_op,
            "mut_op": mut_op,
            "sel_op": sel_op,
            "cx_prob": cx_prob,
            "mut_prob": mut_prob,
            "elitism_size": elitism_size,
            "max_time": max_time,
            "cv_splits": cv_splits,
            "seed": self.seed,
            "has_fasttext": HAS_FASTTEXT
        }
        self._save_json(self.run_config, "run_config.json")

        #  Configuração dos Operadores DEAP 
        crossover_ops = {
            "one_point": tools.cxOnePoint,
            "two_point": tools.cxTwoPoint,
            "uniform": functools.partial(tools.cxUniform, indpb=0.5)
        }
        mutation_ops = {
            "uniform_int": tools.mutUniformInt,
            "shuffle": tools.mutShuffleIndexes,
            "gaussian": tools.mutGaussian
        }
        selection_ops = {
            "tournament": SELECTION_METHODS["tournament"],
            "tournament5": SELECTION_METHODS["tournament5"],
            "sus": SELECTION_METHODS["sus"],
            "roulette": SELECTION_METHODS["roulette"],
            "best": SELECTION_METHODS["best"],
            "rank": SELECTION_METHODS["rank"]
        }

        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)

        pool = Pool(processes=n_jobs)

        configs = self._get_configs(include_fasttext=self.run_config["has_fasttext"])
        summary_results = []

        start_experiment_time = time.time()

        for cfg in configs:
            model_name = cfg['name']
            print(f"\n Otimizando {model_name} ")
            model_dir = os.path.join(self.exp_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            toolbox_model = base.Toolbox()
            toolbox_model.register("map", pool.map)

            # Define Genes
            gene_funcs = []
            low_list, up_list = [], []
            for _, min_v, max_v in cfg['genes']:
                gene_funcs.append(functools.partial(random.randint, min_v, max_v))
                low_list.append(min_v)
                up_list.append(max_v)

            # Cria Indivíduo e População
            toolbox_model.register("individual", tools.initCycle, creator.Individual, gene_funcs, n=1)
            toolbox_model.register("population", tools.initRepeat, list, toolbox_model.individual)

            # Função de Avaliação
            if cfg['type'] == 'sklearn':
                toolbox_model.register("evaluate", functools.partial(self.eval_sklearn, cls=cfg['cls'], p_map=cfg['map'], cv_splits=cv_splits))
            else:
                toolbox_model.register("evaluate", functools.partial(self.eval_fasttext, p_map=cfg['map']))

            # Registra Operadores
            cx_func = crossover_ops.get(cx_op, tools.cxTwoPoint)
            toolbox_model.register("mate", cx_func)

            if mut_op == "uniform_int":
                toolbox_model.register("mutate", functools.partial(tools.mutUniformInt, low=low_list, up=up_list, indpb=0.2))
            else:
                # Default fallback
                toolbox_model.register("mutate", functools.partial(tools.mutUniformInt, low=low_list, up=up_list, indpb=0.2))

            sel_func = selection_ops.get(sel_op, SELECTION_METHODS["tournament"])
            toolbox_model.register("select", sel_func)

            # Estatísticas
            pop = toolbox_model.population(n=pop_size)
            hof = tools.HallOfFame(1) # Guarda o melhor de todos os tempos

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            #  Avaliação Inicial (Gen 0) 
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = list(toolbox_model.map(toolbox_model.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=0, nevals=len(invalid_ind), **record)
            print(f"{model_name} Gen 0: {record}")

            log_filepath = os.path.join(model_dir, f"log_{model_name}.csv")
            start_time = time.time()


            for gen in range(1, n_gen + 1):
                # Controle de Tempo
                if max_time is not None and (time.time() - start_experiment_time) > max_time:
                    print(f"[{model_name}] Tempo máximo atingido. Parando.")
                    break

                # 1. Seleção (Escolhe pais)
                offspring = toolbox_model.select(pop, len(pop))
                offspring = list(map(base.Toolbox().clone, offspring))

                # 2. Cruzamento (Mistura genes)
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < cx_prob:
                        toolbox_model.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # 3. Mutação (Altera genes aleatoriamente)
                for mutant in offspring:
                    if random.random() < mut_prob:
                        toolbox_model.mutate(mutant)
                        del mutant.fitness.values

                # 4. Elitismo Manual (Salva os melhores da geração anterior)

                if elitism_size > 0:
                    elites = tools.selBest(pop, k=elitism_size)
                    # Preserva os elites
                    elites = list(map(base.Toolbox().clone, elites))
                else:
                    elites = []

                # 5. Avaliação dos novos indivíduos
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                if invalid_ind:
                    fitnesses = list(toolbox_model.map(toolbox_model.evaluate, invalid_ind))
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                # 6. Substituição Populacional (Com Elitismo)
                # Junta filhos + elites e escolhe os melhores para manter o tamanho fixo
                if elitism_size > 0:
                    combined = offspring + elites
                    # Seleção determinística dos melhores para a próxima geração
                    # (Garante que a população não cresça e os elites fiquem)
                    pop = tools.selBest(combined, k=pop_size)
                else:
                    pop = offspring

                hof.update(pop)

                # 7. Logs e Checkpoints
                record = stats.compile(pop)
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                print(f"{model_name} Gen {gen}: {record}")

                # Salva log CSV
                df_log = pd.DataFrame(logbook)
                df_log.to_csv(log_filepath, index=False, sep=';')

                # Salva checkpoint
                with open(os.path.join(model_dir, f"hof_gen{gen}.pkl"), 'wb') as f:
                    pickle.dump(hof, f)

            # Fim do Loop
            with open(os.path.join(model_dir, f"final_population.pkl"), 'wb') as f:
                pickle.dump(pop, f)
            with open(os.path.join(model_dir, f"final_hof.pkl"), 'wb') as f:
                pickle.dump(hof, f)

            # Avaliação Final no Holdout (Conjunto nunca visto)
            best_ind = hof[0]
            best_params = self._individual_to_params(best_ind, cfg['map'])
            final_eval = self._final_evaluate(cfg, best_ind, best_params)
            final_eval.update({
                "Modelo": model_name,
                "Parametros": best_params
            })

            summary_results.append(final_eval)
            self._save_json(final_eval, f"final_evaluation_{model_name}.json")

        pool.close()
        pool.join()

        final_df = pd.DataFrame(summary_results)
        final_df.to_csv(os.path.join(self.exp_dir, "final_results_all_models.csv"), index=False, sep=';')
        self._save_json(final_df.fillna('').to_dict(orient='records'), "final_results_all_models.json")

        print(f"\nExperimento finalizado. Dados em: {self.exp_dir}")
        return final_df

    def _final_evaluate(self, cfg, best_ind, best_params):

        try:
            if cfg['type'] == 'sklearn':
                tfidf_params = {}
                clf_params = {}
                for i, val in enumerate(best_ind):
                    p_name = list(cfg['map'].keys())[i]
                    if p_name in ['ngram_range', 'max_df', 'min_df']:
                        tfidf_params[p_name] = cfg['map'][p_name][val]
                    else:
                        clf_params[p_name] = cfg['map'][p_name][val]

                if 'ngram_range' not in tfidf_params:
                    tfidf_params['ngram_range'] = (1, 1)

                pipe = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)), ('clf', cfg['cls'](**clf_params))])
                pipe.fit(self.X_train_val, self.y_train_val)
                preds = pipe.predict(self.X_holdout)

                f1 = f1_score(self.y_holdout, preds, average='macro')
                precision = precision_score(self.y_holdout, preds, average='macro', zero_division=0)
                recall = recall_score(self.y_holdout, preds, average='macro', zero_division=0)
                acc = accuracy_score(self.y_holdout, preds)

                return {
                    "Holdout_F1": float(f1),
                    "Holdout_precision": float(precision),
                    "Holdout_recall": float(recall),
                    "Holdout_accuracy": float(acc)
                }
            else:
                # FastText Logic
                if not HAS_FASTTEXT: return {}
                ft_train_all = os.path.join(self.exp_dir, "fasttext_train_all.txt")
                self._save_fasttext_file(self.X_train_val, self.y_train_val, ft_train_all)

                params = {k: cfg['map'][k][best_ind[i]] for i, k in enumerate(cfg['map'])}
                model = fasttext.train_supervised(input=ft_train_all, verbose=0, **params)
                result = model.test(self.ft_test_file)

                precision, recall = result[1], result[2]
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                return {
                    "Holdout_F1": float(f1),
                    "Holdout_precision": float(precision),
                    "Holdout_recall": float(recall),
                    "Holdout_accuracy": None
                }

        except Exception as e:
            self.error_log.append(f"Erro final evaluation: {e}")
            return {"Holdout_F1": -1.0}


if __name__ == "__main__":
    DATASET_PATH = os.path.join(project_root, "data", "raw", "ouvidoria_sintetico.csv")

    if os.path.exists(DATASET_PATH):
        opt = EvolutionaryOptimizer(data_path=DATASET_PATH, seed=42)

        print("Iniciando Otimização Evolutiva (Sem Early Stopping)...")


        results = opt.run(
            # 1. GERAÇÕES: Quantas vezes o ciclo evolutivo roda. Mais = melhor convergência, mas demora mais.
            n_gen=100,

            # 2. POPULAÇÃO: Número de indivíduos (modelos) por geração. Mais = maior diversidade inicial.
            pop_size=200,

            # 3. OPERADORES GENÉTICOS
            cx_op="two_point",        # Crossover: Troca genes em 2 pontos preservando blocos de genes bons
            mut_op="uniform_int",     # Mutação: Altera valor para um novo inteiro
            sel_op="tournament",      # Seleção por Torneio: escolhe o melhor de 3 aleatórios.

            # 4. PROBABILIDADES
            cx_prob=0.7,              # % de chance de dois pais cruzarem.
            mut_prob=0.3,             # % de chance de um filho sofrer mutação evita estagnar.

            # 5. ELITISMO: Mantém os N melhores da geração anterior intocados para n piorar
            elitism_size=1,

            # 6. LIMITES:
            max_time=None,            # define limite de tempo de execucao

            # 7. AVALIAÇÃO TÉCNICA:
            cv_splits=5,              # 5-Fold Cross Validation
            n_jobs=max(1, cpu_count() - 2) # uso de cpu, config para deixar 2 nucleos sem uso
        )
        print("\nResultados Finais:")
        print(results)
    else:
        print(f"Arquivo nao encontrado: {DATASET_PATH}")