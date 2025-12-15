import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "comparison_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Gerando Relatório Final Detalhado")

    # Carregar ML
    ml_path = os.path.join(ROOT, "outputs", "ml_benchmark_final", "ml_benchmark_results.csv")
    df_ml = pd.DataFrame()
    if os.path.exists(ml_path):
        df_ml = pd.read_csv(ml_path, sep='|', quoting=1)
        df_ml['Tipo'] = 'ML Supervisionado'

        df_ml['Estratégia'] = 'Otimizado (AutoML)'
    else:
        print("[AVISO] CSV de ML não encontrado.")

    # Carregar GenAI
    genai_root = os.path.join(ROOT, "src", "genai", "outputs", "genai_benchmarks")
    df_genai = pd.DataFrame()
    if os.path.exists(genai_root):
        # run mais recente
        runs = [os.path.join(genai_root, d) for d in os.listdir(genai_root) if os.path.isdir(os.path.join(genai_root, d))]
        if runs:
            latest = max(runs, key=os.path.getmtime)
            g_path = os.path.join(latest, "results.csv")
            if os.path.exists(g_path):
                df_genai = pd.read_csv(g_path, sep='|', quoting=1)
                df_genai['Tipo'] = 'LLM (GenAI)'

                if 'prompt_id' in df_genai.columns:
                    df_genai['Estratégia'] = df_genai['prompt_id']
                else:
                    df_genai['Estratégia'] = 'Desconhecida'

    # Unificar
    df_full = pd.concat([df_ml, df_genai], ignore_index=True)

    if df_full.empty:
        print("Nenhum dado para analisar.")
        return

    # Métricas Agrupadas por MODELO + ESTRATÉGIA
    results = []

    df_full['classificacao_real'] = df_full['classificacao_real'].astype(str)
    df_full['classificacao_limpa'] = df_full['classificacao_limpa'].astype(str)


    groups = ['Tipo', 'modelo_id', 'Estratégia']

    for (tipo, model, strategy), group in df_full.groupby(groups):
        f1 = f1_score(group['classificacao_real'], group['classificacao_limpa'], average='macro')
        acc = accuracy_score(group['classificacao_real'], group['classificacao_limpa'])
        time_avg = group['tempo_resposta_s'].mean()

        results.append({
            "Tipo": tipo,
            "Modelo": model,
            "Estratégia": strategy,
            "F1-Macro": f1,
            "Acurácia": acc,
            "Tempo Médio (s)": time_avg,
            "Amostras": len(group)
        })

    metrics = pd.DataFrame(results).sort_values(by='F1-Macro', ascending=False)

    # Salvar CSV
    csv_out = os.path.join(OUTPUT_DIR, "tabela_metricas_final_detalhada.csv")
    metrics.to_csv(csv_out, index=False)
    print(f"\nTabela detalhada salva em: {csv_out}")
    print(metrics[['Modelo', 'Estratégia', 'F1-Macro', 'Tempo Médio (s)']])

    # Gráfico de Barras Comparação de Prompts nas LLMs
    # Filtrando só LLMs para ver o impacto do prompt
    df_llm_metrics = metrics[metrics['Tipo'] == 'LLM (GenAI)']

    if not df_llm_metrics.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_llm_metrics,
            x='Modelo',
            y='F1-Macro',
            hue='Estratégia',
            palette='viridis'
        )
        plt.ylim(0.0, 1.05)
        plt.title("Impacto da Estratégia de Prompt na Performance das LLMs")
        plt.ylabel("F1-Score (Macro)")
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "impacto_prompts.png"))
        print("Gráfico de impacto de prompts salvo.")

    # Gráfico de Pareto Melhor Prompt de cada Modelo vs ML
    # Para o Pareto não ficar poluído, pegamos o melhor F1 de cada modelo
    best_indices = metrics.groupby('Modelo')['F1-Macro'].idxmax()
    best_metrics = metrics.loc[best_indices]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=best_metrics,
        x='Tempo Médio (s)',
        y='F1-Macro',
        hue='Tipo',
        style='Tipo',
        s=200,
        palette='deep'
    )
    plt.xscale('log')
    plt.title("Fronteira de Eficiência")


    for i, row in best_metrics.iterrows():
        plt.text(
            row['Tempo Médio (s)'],
            row['F1-Macro']+0.01,
            f"{row['Modelo']}",
            fontsize=8,
            ha='center'
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pareto_final.png"))
    print("Gráfico Pareto final salvo.")

if __name__ == "__main__":
    main()