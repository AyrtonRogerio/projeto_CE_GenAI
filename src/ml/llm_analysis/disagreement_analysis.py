import pandas as pd
from pathlib import Path

LOCAL_DIR = Path("outputs/local_retrained")
LLM_DIR = Path("src/genai/outputs")

def main():
    llm_files = list(LLM_DIR.rglob("result.csv"))
    local_files = list(LOCAL_DIR.rglob("predictions.csv"))

    out_rows = []


    for llm_f in llm_files:
        df_llm = pd.read_csv(llm_f)

        for local_f in local_files:
            df_local = pd.read_csv(local_f)


            merged = df_llm.merge(df_local, on="text", suffixes=("_llm", "_local"))

            for _, r in merged.iterrows():
                if r.pred_llm != r.pred_local:
                    out_rows.append({
                        "text": r.text,
                        "true_label": r.true_label,
                        "llm_pred": r.pred_llm,
                        "local_pred": r.pred_local,
                        "llm_run": llm_f.parent.name,
                        "local_run": local_f.parent.name
                    })

    pd.DataFrame(out_rows).to_csv("outputs/disagreement_cases.csv", index=False)
    print("[OK] Arquivo gerado: outputs/disagreement_cases.csv")

if __name__ == "__main__":
    main()
