import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from matplotlib.lines import Line2D
import numpy as np


sns.set_theme(style="whitegrid")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "comparison_final_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


STRATEGY_COLORS = {
    'Zero-Shot': '#d62728',
    'Zero-Shot-CoT': '#2ca02c',
    'Few-Shot': '#ff7f0e',
    'Meta-Prompt': '#9467bd',
    'Evolucionária': '#1f77b4',
    'Outra': '#7f7f7f'
}

MODEL_MARKERS = {
    'Log. Reg.': '*',
    'Naive Bayes': 's',
    'SVM': 'D',
    'FastText': '^',
    'Gemini 2.5-Flash-Lite': 'P',
    'GPT-5-NANO': 'X',
    'Llama-3.1-8b-Instant': '<'
}

MODEL_SIZES = {
    'Log. Reg.': 190,
    'Naive Bayes': 50,
    'SVM': 60,
    'FastText': 90,
    'Gemini 2.5-Flash-Lite': 90,
    'GPT-5-NANO': 90,
    'Llama-3.1-8b-Instant': 80
}

MODEL_JITTER = {
    'Log. Reg.': -0.01,
    'Naive Bayes': -0.006,
    'SVM': -0.003,
    'FastText': 0.0,
    'Gemini 2.5-Flash-Lite': 0.003,
    'GPT-5-NANO': 0.006,
    'Llama-3.1-8b-Instant': 0.01
}

def get_clean_model_name(name):
    mapping = {
        'LogisticRegression': 'Log. Reg.',
        'NaiveBayes': 'Naive Bayes',
        'SVM': 'SVM',
        'FastText': 'FastText',
        'gemini-2.5-flash-lite': 'Gemini 2.5-Flash-Lite',
        'gpt-5-nano': 'GPT-5-NANO',
        'llama-3.1-8b-instant': 'Llama-3.1-8b-Instant'
    }
    return mapping.get(name, name)

def get_clean_strategy(s):
    s = str(s).lower()
    if 'cot' in s: return 'Zero-Shot-CoT'
    if 'meta' in s: return 'Meta-Prompt'
    if 'few' in s: return 'Few-Shot'
    if 'zero' in s: return 'Zero-Shot'
    if 'otimizado' in s: return 'Evolucionária'
    return 'Evolucionária'


def load_data():
    ml_path = os.path.join(ROOT, "outputs", "ml_benchmark_final", "ml_benchmark_results.csv")
    genai_root = os.path.join(ROOT, "src", "genai", "outputs", "genai_benchmarks")

    df_ml, df_llm = pd.DataFrame(), pd.DataFrame()

    if os.path.exists(ml_path):
        df_ml = pd.read_csv(ml_path, sep="|", quoting=1)
        df_ml["Tipo"] = "ML"
        df_ml["Estratégia"] = "Evolucionária"

    if os.path.exists(genai_root):
        latest = max(
            [os.path.join(genai_root, d) for d in os.listdir(genai_root)],
            key=os.path.getmtime
        )
        df_llm = pd.read_csv(os.path.join(latest, "results.csv"), sep="|", quoting=1)
        df_llm["Tipo"] = "LLM"

        prompt_map = {
            'zero_shot_direto': 'Zero-Shot-CoT',
            'zero_shot_cot': 'Zero-Shot-CoT',
            'few_shot_completo': 'Few-Shot',
            'meta_prompt_classificacao': 'Meta-Prompt'
        }

        col = "prompt_id" if "prompt_id" in df_llm.columns else "prompt"
        df_llm["Estratégia"] = df_llm[col].map(prompt_map).fillna("Evolucionário")

    df = pd.concat([df_ml, df_llm], ignore_index=True)
    df["classificacao_real"] = df["classificacao_real"].astype(str)
    df["classificacao_limpa"] = df["classificacao_limpa"].astype(str)
    return df

def plot_scientific(df, filename, title, use_log=False, time_unit="s"):

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    dfp = df.copy()

    dfp["Tempo"] = dfp["Tempo Médio (s)"]
    xlabel = "Tempo de Inferência (s) " if time_unit == "s" else r"Tempo de Inferência ($\mu$s) "

    for model in dfp["Modelo"].unique():
        subset = dfp[dfp["Modelo"] == model]
        ax.scatter(
            subset["Tempo"] * (1 + MODEL_JITTER.get(model, 0)),
            subset["F1-Macro"],
            marker=MODEL_MARKERS[model],
            s=MODEL_SIZES[model],
            c=subset["Estratégia"].map(STRATEGY_COLORS),
            edgecolor='black',
            linewidth=0.4
        )

    if use_log:
        ax.set_xscale("log")

    ax.set_title(title)
    ax.set_xlabel(xlabel+" ")
    ax.set_ylabel("Macro F1")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    strategy_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=STRATEGY_COLORS[s],
               markeredgecolor='black',
               markersize=6, label=s)
        for s in STRATEGY_COLORS if s in dfp["Estratégia"].unique()
    ]

    leg1 = fig.legend(
        handles=strategy_handles,
        title="Estratégias",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False
    )
    leg1.get_title().set_fontsize(10)

    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[m],
               color='black', linestyle='None',
               markersize=6, label=m)
        for m in MODEL_MARKERS if m in dfp["Modelo"].unique()
    ]

    leg2 = fig.legend(
        handles=model_handles,
        title="Modelos",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.50),
        ncol=2,
        frameon=False
    )
    leg2.get_title().set_fontsize(10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    plt.close()
    print(f"[OK] {filename}")


def main():
    df = load_data()

    metrics = df.groupby(["Tipo", "modelo_id", "Estratégia"], as_index=False).apply(
        lambda x: pd.Series({
            "F1-Macro": f1_score(x["classificacao_real"], x["classificacao_limpa"], average="macro"),
            "Tempo Médio (s)": x["tempo_resposta_s"].mean()
        })
    )

    metrics["Modelo"] = metrics["modelo_id"].apply(get_clean_model_name)
    metrics["Estratégia"] = metrics["Estratégia"].apply(get_clean_strategy)

    df_ml = metrics[metrics["Tipo"] == "ML"]
    df_llm = metrics[metrics["Tipo"] == "LLM"]

    if not df_ml.empty:
        plot_scientific(df_ml, "ml_scientific.png", "Modelos de ML", time_unit="us")

    if not df_llm.empty:
        plot_scientific(df_llm, "llm_scientific.png", "LLMs – Estratégias de Prompt")

    df_top = df_llm.sort_values("F1-Macro", ascending=False).groupby("Modelo").head(1)
    df_final = pd.concat([df_ml, df_top])

    plot_scientific(df_final, "final_scientific.png", "Comparação entre Modelos de ML vs LLMs", use_log=True)

if __name__ == "__main__":
    main()
