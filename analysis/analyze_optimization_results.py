import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import glob
from pathlib import Path


sns.set_theme(style="ticks", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
COLORS = {"Evolutionary": "#2ca02c", "Baseline": "#1f77b4"}


CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR.parent / "outputs").exists(): ROOT = CURRENT_DIR.parent
elif (CURRENT_DIR.parent.parent / "outputs").exists(): ROOT = CURRENT_DIR.parent.parent
else: ROOT = Path(os.getcwd())

OUTPUT_DIR = os.path.join(ROOT, "outputs", "analysis_optimization")
os.makedirs(OUTPUT_DIR, exist_ok=True)
EVO_DIR = os.path.join(ROOT, "outputs", "evolutionary")
GRID_DIR = os.path.join(ROOT, "outputs", "gridsearch_optuna")
RECONSTRUCTED_JSON = os.path.join(ROOT, "outputs", "reconstructed_models.json")

def load_json(path):
    try:
        with open(path, 'r') as f: return json.load(f)
    except: return {}

def plot_convergence_final():
    print("\nGerando Convergência")
    log_files = glob.glob(os.path.join(EVO_DIR, "**", "log.csv"), recursive=True)
    if not log_files: return

    model_logs = {}
    for log_path in log_files:
        path_parts = Path(log_path).parts
        model_name = path_parts[-2]
        mod_time = os.path.getmtime(log_path)
        if model_name not in model_logs or mod_time > model_logs[model_name]['time']:
            model_logs[model_name] = {'path': log_path, 'time': mod_time}

    for model, info in model_logs.items():
        try:
            df = pd.read_csv(info['path'])
            if 'gen' not in df.columns: continue

            fig, ax = plt.subplots(figsize=(8, 5))

            # Diversidade
            if 'min' in df.columns and 'max' in df.columns:
                ax.fill_between(df['gen'], df['min'], df['max'], color='#dddddd', alpha=0.5, label='Range (Min-Max)')

            #  Média
            if 'avg' in df.columns:
                ax.plot(df['gen'], df['avg'], '--', color='#1f77b4', linewidth=1.5, label='Média Pop.')

            #  Melhor (Elite)
            if 'max' in df.columns:
                ax.plot(df['gen'], df['max'], '-', color='#2ca02c', linewidth=2.5, label='Elite (Max)')

                # PONTO DE CONVERGÊNCIA
                max_val = df['max'].max()
                convergence_gen = df[df['max'] >= max_val - 1e-9]['gen'].iloc[0]

                # Ponto Vermelho
                ax.scatter(convergence_gen, max_val, color='red', s=80, zorder=10,
                           label=f'Conv. (Gen {int(convergence_gen)})')

                # Linha Vertical
                ax.axvline(x=convergence_gen, color='red', linestyle=':', alpha=0.6)



                text_offset_y = 0.0005 if max_val > 0.99 else (max_val * 0.005)
                ax.text(convergence_gen + 2, max_val + text_offset_y,
                        f"{max_val:.4f}",
                        color='red', va='bottom', fontweight='bold', fontsize=12)


            y_min_data = df['min'].min() if 'min' in df.columns else 0.9
            y_max_data = df['max'].max()

            if y_max_data > 0.995:

                ax.set_ylim(0.990, 1.005)
            elif y_max_data > 0.98:
                ax.set_ylim(0.970, 1.005)
            else:
                ax.set_ylim(max(0, y_min_data - 0.02), y_max_data + 0.02)

            ax.set_title(f"Dinâmica Evolutiva: {model}", pad=15)
            ax.set_xlabel("Gerações")
            ax.set_ylabel("F1-Score")
            sns.despine()
            ax.grid(True, axis='y', linestyle=':', alpha=0.4)


            ax.legend(loc='lower right', frameon=True, framealpha=1.0, fontsize=10, edgecolor='gray')

            plt.savefig(os.path.join(OUTPUT_DIR, f"convergencia_{model}_final.png"), bbox_inches='tight')
            plt.close()
            print(f"Salvo: convergencia_{model}_final.png")

        except Exception as e:
            print(f"Erro {model}: {e}")

def plot_comparison_bar_final():
    print("\nGerando Barras Comparativas")
    if not os.path.exists(RECONSTRUCTED_JSON): return
    with open(RECONSTRUCTED_JSON, 'r') as f: data = json.load(f)

    records = []
    for key, info in data.items():
        if 'score' not in info or info['score'] == -1: continue
        origin = "Evolutionary" if "Evolutionary" in info['origin'] else "Baseline"
        m_type = info['model_type']
        if m_type == "LinearSVC": m_type = "SVM"
        if m_type == "MultinomialNB": m_type = "NaiveBayes"
        records.append({"Modelo": m_type, "Estratégia": origin, "F1-Score": info['score']})

    if not records: return
    df = pd.DataFrame(records)
    df_best = df.groupby(['Modelo', 'Estratégia'], as_index=False)['F1-Score'].max()

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=df_best, x='Modelo', y='F1-Score', hue='Estratégia', palette=COLORS, edgecolor='black', linewidth=0.5)

    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Teto Teórico (1.0)')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=2, fontsize=10)

    plt.ylim(0.94, 1.025)
    plt.title("Eficácia: Evolutivo vs Baseline", pad=15)
    plt.ylabel("Melhor F1-Score")
    plt.xlabel("")
    sns.despine(left=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    line = Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    handles.append(line)
    labels.append("Teto Teórico (1.0)")

    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.0, 1), loc='upper left', frameon=False)

    plt.savefig(os.path.join(OUTPUT_DIR, "comparativo_barras_final.png"), bbox_inches='tight')
    plt.close()
    print("Salvo: comparativo_barras_final.png")

def plot_pareto_final():
    print("\nGerando Pareto Final")
    data_points = []
    evo_files = glob.glob(os.path.join(EVO_DIR, "**", "final_evaluation_*.json"), recursive=True)
    for fpath in evo_files:
        d = load_json(fpath)
        if 'runtime_seconds' in d:
            data_points.append({"Modelo": d.get('Modelo'), "Estratégia": "Evolutionary", "Tempo (s)": d['runtime_seconds'], "F1-Score": d['Holdout_F1']})

    grid_files = glob.glob(os.path.join(GRID_DIR, "**", "final_*_*.json"), recursive=True)
    for fpath in grid_files:
        d = load_json(fpath)
        if 'runtime_seconds' in d:
            score = d.get('f1_holdout', d.get('best_f1_train', 0))
            model = d.get('model')
            if model == "LinearSVC": model = "SVM"
            if model == "MultinomialNB": model = "NaiveBayes"
            data_points.append({"Modelo": model, "Estratégia": "Baseline", "Tempo (s)": d['runtime_seconds'], "F1-Score": score})

    if not data_points: return
    df = pd.DataFrame(data_points)
    df = df[df['Tempo (s)'] > 0.01]

    plt.figure(figsize=(9, 6))
    markers = {"SVM": "s", "FastText": "o", "NaiveBayes": "P", "LogisticRegression": "X"}

    sns.scatterplot(data=df, x='Tempo (s)', y='F1-Score', hue='Estratégia', style='Modelo', markers=markers, s=250, palette=COLORS, alpha=0.85, edgecolor='k')

    plt.xscale('log')
    y_min = df['F1-Score'].min()
    plt.ylim(max(0.9, y_min - 0.02), 1.015)

    plt.title("Fronteira de Pareto: Custo vs. Resultado")
    plt.xlabel("Tempo de Otimização (s) - Log")
    sns.despine()
    plt.grid(True, which="major", linestyle='--', alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.savefig(os.path.join(OUTPUT_DIR, "pareto_final.png"), bbox_inches='tight')
    plt.close()
    print("Salvo: pareto_final.png")

if __name__ == "__main__":
    plot_convergence_final()
    plot_comparison_bar_final()
    plot_pareto_final()