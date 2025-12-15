

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGG = PROJECT_ROOT / "outputs" / "local_models"
ALL_PRED = AGG / "all_predictions_long.csv"
OUT = AGG / "disagreement_samples.csv"

def main():
    if not ALL_PRED.exists():
        raise FileNotFoundError(f"{ALL_PRED} não encontrado. Rode compare_local_vs_llm.py antes.")

    df = pd.read_csv(ALL_PRED, encoding="utf-8")

    pivot = df.pivot_table(index=['text','true_label'], columns='model', values='predicted_label', aggfunc=lambda x: x.iloc[0] if len(x)>0 else "")
    pivot = pivot.reset_index()

    def distinct_preds(row):
        preds = [v for k,v in row.items() if k not in ('text','true_label') and isinstance(v, str)]
        return len(set(preds)) > 1

    pivot['divergent'] = pivot.apply(distinct_preds, axis=1)
    dis = pivot[pivot['divergent']]
    dis.to_csv(OUT, index=False, encoding='utf-8')
    print(f"[OK] {len(dis)} instâncias divergentes salvas em {OUT}")

if __name__ == "__main__":
    main()
