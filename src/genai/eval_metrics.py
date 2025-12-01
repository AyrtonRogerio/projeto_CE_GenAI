from typing import List, Dict
from collections import defaultdict


def accuracy(records: List[Dict]) -> float:
    if not records:
        return 0.0
    acertos = sum(1 for r in records if r["acertou"] == 1)
    return acertos / len(records)


def confusion(records: List[Dict], labels: List[str]):
    m = defaultdict(int)
    for r in records:
        t = r["classificacao_real"]
        p = r["classificacao_limpa"]
        m[(t, p)] += 1
    return dict(m)
