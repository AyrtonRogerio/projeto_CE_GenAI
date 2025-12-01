import os
import glob
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

#  Configurações Visuais 
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 12
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.labelpad'] = 15
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLOR_MAP = {
    'Evolutionary': '#2ca02c',      # Verde
    'GridSearchCV': '#1f77b4',      # Azul
    'Optuna': '#ff7f0e',            # Laranja
    'Grid/Optuna': '#1f77b4'        # Azul
}


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
    return path

def get_pareto_frontier(df, x_col='runtime_seconds', y_col='best_f1_holdout'):
    if df.empty: return df, []
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[True, False])

    pareto_front = []
    current_max_y = -np.inf
    indices_pareto = []

    for idx, row in sorted_df.iterrows():
        if row[y_col] >= current_max_y:
            pareto_front.append(row)
            indices_pareto.append(idx)
            current_max_y = row[y_col]

    return pd.DataFrame(pareto_front), indices_pareto

#Coleta de Dados 

def collect_grid_results(grid_dir):
    results = []
    if not grid_dir or not os.path.exists(grid_dir): return pd.DataFrame()
    json_files = glob.glob(os.path.join(grid_dir, "final_*.json"))
    for jf in json_files:
        try:
            with open(jf, 'r') as f: data = json.load(f)
            results.append({
                'model': data.get('model', 'Unknown'),
                'approach': data.get('algorithm', 'Grid/Optuna'),
                'best_f1_holdout': float(data.get('f1_holdout', 0.0)),
                'runtime_seconds': float(data.get('runtime_seconds', 0.0))
            })
        except: pass
    return pd.DataFrame(results)

def collect_evo_results(evo_dir):
    results = []
    if not evo_dir or not os.path.exists(evo_dir): return pd.DataFrame()
    main_res_path = os.path.join(evo_dir, "final_results_all_models.json")
    if not os.path.exists(main_res_path): return pd.DataFrame()
    try:
        with open(main_res_path, 'r') as f: data_list = json.load(f)
        for item in data_list:
            model_name = item.get('Modelo')
            f1 = float(item.get('Holdout_F1', 0.0))
            runtime = 0.0

            eval_path = os.path.join(evo_dir, f"final_evaluation_{model_name}.json")
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f_eval:
                    runtime = float(json.load(f_eval).get('runtime_seconds', 0.0))
            else:
                log_path = os.path.join(evo_dir, model_name, f"log_{model_name}.csv")
                if os.path.exists(log_path):
                    df_log = pd.read_csv(log_path, sep=';')
                    if 'time' in df_log.columns: runtime = df_log['time'].sum()

            results.append({
                'model': model_name,
                'approach': 'Evolutionary',
                'best_f1_holdout': f1,
                'runtime_seconds': runtime
            })
    except: pass
    return pd.DataFrame(results)

def aggregate_and_save(evo_dir, grid_dir, out_dir):
    print(f"\nColetando dados brutos...")
    df_grid = collect_grid_results(grid_dir)
    df_evo = collect_evo_results(evo_dir)
    df_final = pd.concat([df_grid, df_evo], ignore_index=True)

    if df_final.empty:
        print("ERRO: Nenhum dado encontrado.")
        return pd.DataFrame()

    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, "aggregated_comparison.csv")
    df_final.to_csv(out_file, sep=';', index=False)
    print(f"Dados salvos em: {out_file}")
    return df_final

def load_or_create_data(comp_dir, evo_dir, grid_dir):
    csv_path = os.path.join(comp_dir, "aggregated_comparison.csv")
    if os.path.exists(csv_path):
        print(f"Carregando dados existentes: {csv_path}")
        df = pd.read_csv(csv_path, sep=';')
    else:
        df = aggregate_and_save(evo_dir, grid_dir, comp_dir)

    if not df.empty:
        df['runtime_seconds'] = pd.to_numeric(df['runtime_seconds'], errors='coerce').fillna(0)
        df['best_f1_holdout'] = pd.to_numeric(df['best_f1_holdout'], errors='coerce').fillna(0)
    return df

def load_evo_logs(evo_path):
    if not evo_path or not os.path.exists(evo_path): return pd.DataFrame()
    trajectory_data = []
    log_files = glob.glob(os.path.join(evo_path, "**", "log_*.csv"), recursive=True)
    for log_f in log_files:
        try:
            df = pd.read_csv(log_f, sep=';')
            df['model'] = os.path.basename(log_f).replace('log_', '').replace('.csv', '')
            trajectory_data.append(df)
        except: pass
    return pd.concat(trajectory_data, ignore_index=True) if trajectory_data else pd.DataFrame()


def format_func(x, pos):
    if x < 1: return f"{x:.1f}s"
    return f"{int(x)}s"

#GRÁFICO DE PARETO
def plot_pareto_landscape(df, out_dir):
    if df.empty: return
    print("Gerando Pareto...")

    df_plot = df.copy()
    df_plot['strategy_label'] = df_plot['approach'].apply(lambda x: 'Evolutionary' if 'evolutionary' in str(x).lower() else str(x))

    pareto_df, pareto_indices = get_pareto_frontier(df_plot)
    df_plot['is_pareto'] = False
    if pareto_indices:
        df_plot.loc[pareto_indices, 'is_pareto'] = True

    plt.figure(figsize=(16, 10))

    sns.scatterplot(
        data=df_plot, x='runtime_seconds', y='best_f1_holdout',
        hue='strategy_label', style='model', palette=COLOR_MAP,
        s=180, alpha=0.75, edgecolor='k', zorder=5
    )

    if not pareto_df.empty:
        pareto_df = pareto_df.sort_values('runtime_seconds')
        plt.plot(pareto_df['runtime_seconds'], pareto_df['best_f1_holdout'],
                 color='grey', linestyle='--', alpha=0.6, lw=1.5, zorder=1, label='Fronteira Pareto')

    texts = []

    for _, row in pareto_df.iterrows():
        label = f"{row['model']}\n{row['best_f1_holdout']:.4f}"

        if HAS_ADJUST_TEXT:
            texts.append(plt.text(row['runtime_seconds'], row['best_f1_holdout'], label, fontsize=10, fontweight='bold', color='#444444'))
        else:
            plt.annotate(label, (row['runtime_seconds'], row['best_f1_holdout']), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    if HAS_ADJUST_TEXT and texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.xscale('symlog', linthresh=1.0)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    y_min = df_plot['best_f1_holdout'].min()
    y_max = df_plot['best_f1_holdout'].max()
    margin = (y_max - y_min) * 0.15
    if margin < 0.001: margin = 0.01
    plt.ylim(max(0, y_min - margin), y_max + margin * 1.2)

    plt.title("Trade-off: Tempo vs Performance (Pareto)", fontsize=22, weight='bold')
    plt.xlabel("Tempo de Execução (Log)", fontsize=16)
    plt.ylabel("F1-Score (Holdout)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Legenda", frameon=False)
    plt.grid(True, which="major", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_efficiency_frontier.png"), dpi=150)
    plt.close()

#GRÁFICO DE BARRAS
def plot_strategy_comparison_bar(df, out_dir):
    if df.empty: return
    print("Gerando Barras (Comparação Simples)...")

    df_plot = df.copy()
    df_plot['strategy_label'] = df_plot['approach'].apply(lambda x: 'Evolutionary' if 'evolutionary' in str(x).lower() else 'Grid/Optuna')
    ordered_models = sorted(df_plot['model'].unique())

    fig, ax = plt.subplots(figsize=(16, 10))
    plt.subplots_adjust(bottom=0.2)

    sns.barplot(
        data=df_plot, x='model', y='best_f1_holdout', hue='strategy_label',
        palette=COLOR_MAP, order=ordered_models, edgecolor='white', ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=-25, fontsize=11, color='white', fontweight='bold')

    y_min = df_plot['best_f1_holdout'].min()
    y_max = df_plot['best_f1_holdout'].max()
    ax.set_ylim(y_min - 0.05 if y_min > 0.8 else 0, y_max * 1.05)

    plt.title("Comparativo de Performance", fontsize=22, weight='bold')
    plt.ylabel("Melhor F1-Score", fontsize=16)
    plt.xlabel("")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_MAP['Evolutionary'], label='Evolucionário'),
        Patch(facecolor=COLOR_MAP['Grid/Optuna'], label='Grid/Optuna')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=12)

    sns.despine(left=True)
    plt.savefig(os.path.join(out_dir, "strategy_comparison_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()

#GRÁFICO DE CONVERGÊNCIA
def plot_evo_convergence(df_logs, out_dir):
    if df_logs.empty: return
    print("Gerando Convergência...")
    for model in df_logs['model'].unique():
        df_m = df_logs[df_logs['model'] == model]
        plt.figure(figsize=(12, 7))

        plt.fill_between(df_m['gen'], df_m['min'], df_m['max'], color=COLOR_MAP['Evolutionary'], alpha=0.1)
        plt.plot(df_m['gen'], df_m['avg'], '--', color=COLOR_MAP['Evolutionary'], label='Média Populacional', alpha=0.7)
        plt.plot(df_m['gen'], df_m['max'], '-', color='#1a5c1a', linewidth=3, label='Melhor (Max)')

        idx_max = df_m['max'].idxmax()
        row_max = df_m.loc[idx_max]

        plt.scatter(row_max['gen'], row_max['max'], color='red', s=120, zorder=10,
                    edgecolors='white', linewidth=2, label='Melhor Absoluto')

        plt.annotate(
            f"F1: {row_max['max']:.4f}\nGen: {int(row_max['gen'])}",
            xy=(row_max['gen'], row_max['max']),
            xytext=(0, 20), textcoords='offset points', ha='center',
            color='red', fontweight='bold'
        )

        plt.title(f"Dinâmica Evolutiva: {model}", fontsize=18, weight='bold')
        plt.xlabel("Geração", fontsize=14)
        plt.ylabel("Fitness (F1-Score)", fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(loc='lower right', frameon=True)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"evo_convergence_{model}.png"), dpi=150)
        plt.close()


# def plot_evo_convergence(df_logs, out_dir):
#     if df_logs.empty: return
#     print("Gerando Convergência...")
#     for model in df_logs['model'].unique():
#         df_m = df_logs[df_logs['model'] == model]
#         plt.figure(figsize=(12, 7))
#
#         plt.fill_between(df_m['gen'], df_m['min'], df_m['max'], color=COLOR_MAP['Evolutionary'], alpha=0.1)
#         plt.plot(df_m['gen'], df_m['avg'], '--', color=COLOR_MAP['Evolutionary'], label='Média Populacional', alpha=0.7)
#         plt.plot(df_m['gen'], df_m['max'], '-', color='#1a5c1a', linewidth=3, label='Melhor (Max)')
#
#         idx_max = df_m['max'].idxmax()
#         row_max = df_m.loc[idx_max]
#
#         plt.scatter(row_max['gen'], row_max['max'], color='red', s=120, zorder=10,
#                     edgecolors='white', linewidth=2, label='Melhor Absoluto')
#
#         plt.annotate(
#             f"F1: {row_max['max']:.4f}\nGen: {int(row_max['gen'])}",
#             xy=(row_max['gen'], row_max['max']),
#             xytext=(0, 10), textcoords='offset points', ha='center',
#             color='red', fontweight='bold'
#         )
#
#         plt.title(f"Dinâmica Evolutiva: {model}", fontsize=18, weight='bold')
#         plt.xlabel("Geração", fontsize=14)
#         plt.ylabel("Fitness (F1-Score)", fontsize=14)
#         plt.grid(True, axis='y', alpha=0.3)
#         plt.legend(loc='lower right', frameon=True)
#         sns.despine()
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, f"evo_convergence_{model}.png"), dpi=150)
#         plt.close()


def find_latest_run(base_path):
    if not os.path.exists(base_path): return None
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None

def run_analysis(comp_dir, evo_dir, grid_dir):
    print(f"\n=== Análise Comparativa (Final) ===")
    print(f"Output: {comp_dir}")
    df_agg = load_or_create_data(comp_dir, evo_dir, grid_dir)

    if not df_agg.empty:
        plot_pareto_landscape(df_agg, comp_dir)
        plot_strategy_comparison_bar(df_agg, comp_dir)
    else:
        print("ERRO: Falha ao obter dados.")

    if evo_dir:
        df_logs = load_evo_logs(evo_dir)
        if not df_logs.empty:
            evo_out = ensure_dir(os.path.join(comp_dir, "evolutionary_details"))
            plot_evo_convergence(df_logs, evo_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp_dir", type=str)
    parser.add_argument("--evo_dir", type=str)
    parser.add_argument("--grid_dir", type=str)
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()

    target_comp = args.comp_dir
    target_evo = args.evo_dir
    target_grid = args.grid_dir
    is_auto = (not target_comp) and (not args.manual)

    if is_auto:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_root = os.path.abspath(os.path.join(script_dir, "../../outputs"))
        print(f"MODO AUTO: Detectando inputs")
        target_evo = find_latest_run(os.path.join(outputs_root, "evolutionary"))
        target_grid = find_latest_run(os.path.join(outputs_root, "gridsearch_optuna"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_comp = os.path.join(outputs_root, "comparison", f"analysis_{timestamp}")

        if target_evo: print(f"Evo: {os.path.basename(target_evo)}")
        if target_grid: print(f"Grid: {os.path.basename(target_grid)}")

    if not target_evo or not target_grid:
        print("\nERRO: Inputs não encontrados.")
        exit(1)

    run_analysis(target_comp, target_evo, target_grid)