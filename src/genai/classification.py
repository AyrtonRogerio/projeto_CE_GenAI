import re

CATEGORIAS_VALIDAS = [
    "Elogio", "Crítica", "Sugestão",
    "Pedido de LAI", "Solicitação", "Denúncia"
]

MAPA = {c.lower(): c for c in CATEGORIAS_VALIDAS}


def limpar_resposta(raw: str) -> str:
    if raw is None:
        return "ERRO_API"

    linhas = [l.strip() for l in raw.split("\n") if l.strip()]
    candidato = linhas[-1]

    candidato = re.sub(r'^categoria[:\s]*', '', candidato, flags=re.IGNORECASE)

    candidato_limpo = re.sub(r'[^\w\s]', '', candidato).lower().strip()

    for k, v in MAPA.items():
        if k == candidato_limpo:
            return v

    for k, v in MAPA.items():
        if k in candidato_limpo:
            return v

    return "NAO_MAPEADO"
