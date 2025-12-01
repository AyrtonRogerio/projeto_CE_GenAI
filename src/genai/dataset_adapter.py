from typing import List, Dict
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    from src.utils.data_loader import load_data
except ImportError:
    from utils.data_loader import load_data


def carregar_demandas(path: str) -> List[Dict]:
    df = load_data(path)

    demandas = []
    for i, row in df.iterrows():
        demandas.append({
            "id": f"{i+1:04d}",
            "texto": str(row["text"]).strip(),
            "classificacao_real": str(row["label"]).strip()
        })

    return demandas
