import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from sklearn.metrics import f1_score


sns.set_theme(style="whitegrid")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 19,
    'axes.titlesize': 19,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,
    'figure.dpi': 300,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "comparison_final_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)



def get_clean_model_name(name):
    mapping = {
        'LogisticRegression': 'Log. Reg.',
        'NaiveBayes': 'Naive Bayes',
        'SVM': 'SVM',
        'FastText': 'FastText',
        'gemini-2.5-flash-lite': 'Gemini 2.5',
        'gpt-5-nano': 'GPT-5',
        'llama-3.1-8b-instant': 'Llama 3'
    }
    return mapping.get(name, name)


def get_clean_strategy(s):
    s = str(s).lower()
    if 'cot' in s: return 'CoT'
    if 'meta' in s: return 'Meta-Prompt'
    if 'few' in s: return 'Few-Shot'
    if 'zero' in s: return 'Zero-Shot'
    if 'otimizado' in s: return 'Evolucionária'
    return 'Outra'



def load_data():
    ml_path = os.path.join(ROOT, "outputs", "ml_benchmark_final", "ml_benchmark_results.csv")
    genai_root = os.path.join(ROOT, "src", "genai", "outputs", "genai_benchmarks")

    df_ml, df_llm = pd.DataFrame(), pd.DataFrame()

    if os.path.exists(ml_path):
        df_ml = pd.read_csv(ml_path, sep="|", quoting=1)
        df_ml["Tipo"] = "ML"
        df_ml["Estratégia"] = "Otimizado"

    if os.path.exists(genai_root):
        subdirs = [os.path.join(genai_root, d) for d in os.listdir(genai_root)]
        if subdirs:
            latest = max(subdirs, key=os.path.getmtime)
            path = os.path.join(latest, "results.csv")

            if os.path.exists(path):
                df_llm = pd.read_csv(path, sep="|", quoting=1)
                df_llm["Tipo"] = "LLM"

                prompt_map = {
                    'zero_shot_direto': 'Zero-Shot',
                    'zero_shot_cot': 'CoT',
                    'few_shot_completo': 'Few-Shot',
                    'meta_prompt_classificacao': 'Meta-Prompt'
                }

                col = "prompt_id" if "prompt_id" in df_llm.columns else "prompt"
                df_llm["Estratégia"] = df_llm[col].map(prompt_map).fillna("Outra")

    df = pd.concat([df_ml, df_llm], ignore_index=True)

    df["classificacao_real"] = df["classificacao_real"].astype(str)
    df["classificacao_limpa"] = df["classificacao_limpa"].astype(str)

    return df


def plot_scientific(df, filename, title, use_log=False, time_unit="s"):

    # Cores discretas, científicas
    strategy_colors = {
        'Evolucionária': '#1f77b4',
        'Zero-Shot': '#d62728',
        'CoT': '#2ca02c',
        'Few-Shot': '#ff7f0e',
        'Meta-Prompt': '#9467bd',
        'Outra': '#7f7f7f'
    }


    model_markers = {
        'Log. Reg.': 'o',
        'Naive Bayes': 's',
        'SVM': 'D',
        'FastText': '^',
        'Gemini 2.5': 'P',
        'GPT-5': 'X',
        'Llama 3': 'v'
    }

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    dfp = df.copy()


    if time_unit == 's':
        dfp['Tempo'] = dfp['Tempo Médio (s)']
        xlabel = "Tempo de Inferência (s)"
    elif time_unit == 'ms':
        dfp['Tempo'] = dfp['Tempo Médio (s)'] * 1000
        xlabel = "Tempo de Inferência (ms)"
    else:
        dfp['Tempo'] = dfp['Tempo Médio (s)'] * 1_000_000
        xlabel = r"Tempo de Inferência ($\mu$s)"


    sns.scatterplot(
        data=dfp,
        x="Tempo",
        y="F1-Macro",
        hue="Estratégia",
        style="Modelo",
        palette=strategy_colors,
        markers=model_markers,
        s=90,
        edgecolor='black',
        linewidth=0.6,
        ax=ax
    )


    if use_log:
        ax.set_xscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel(xlabel)
    plt.ylabel("Macro F1")
    plt.title(title)

    plt.grid(axis="y", linestyle="--", alpha=0.35)


    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    plt.close()
    print(f"[OK] {filename}")


def main():
    df = load_data()
    if df.empty:
        print("ERRO: Nenhum dado carregado.")
        return


    metrics = df.groupby(["Tipo", "modelo_id", "Estratégia"]).apply(
        lambda x: pd.Series({
            "F1-Macro": f1_score(x["classificacao_real"], x["classificacao_limpa"], average="macro"),
            "Tempo Médio (s)": x["tempo_resposta_s"].mean()
        })
    ).reset_index()

    metrics["Modelo"] = metrics["modelo_id"].apply(get_clean_model_name)
    metrics["Estratégia"] = metrics["Estratégia"].apply(get_clean_strategy)

    # ML apenas
    df_ml = metrics[metrics["Tipo"] == "ML"]
    if not df_ml.empty:
        plot_scientific(
            df_ml,
            "ml_scientific.png",
            "Eficiência dos Modelos de ML",
            time_unit="us"
        )

    # LLMs
    df_llm = metrics[metrics["Tipo"] == "LLM"]
    if not df_llm.empty:
        plot_scientific(
            df_llm,
            "llm_scientific.png",
            "LLMs – Comparação das Estratégias de Prompt",
            time_unit="s"
        )

    # Final: ML + Top LLM
    df_top = df_llm.sort_values("F1-Macro", ascending=False).groupby("Modelo").head(1)
    df_final = pd.concat([df_ml, df_top])

    plot_scientific(
        df_final,
        "final_scientific.png",
        "Comparação Entre Modelos de ML e LLMs",
        time_unit="s",
        use_log=True
    )


if __name__ == "__main__":
    main()
