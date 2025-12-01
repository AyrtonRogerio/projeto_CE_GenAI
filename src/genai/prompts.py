CATEGORIAS_VALIDAS = [
    'Elogio', 'Crítica', 'Sugestão',
    'Pedido de LAI', 'Solicitação', 'Denúncia'
]

PROMPT_STRATEGIES = [
    {
        "id": "zero_shot_direto",
        "template": (
            "Sua tarefa é classificar demandas de ouvidoria.\n"
            "Classifique a demanda abaixo em uma das seguintes categorias: "
            "'Elogio', 'Crítica', 'Sugestão', 'Pedido de LAI', 'Solicitação', 'Denúncia'.\n"
            "Responda apenas com o nome da categoria.\n\n"
            "Demanda: \"{demanda}\"\nCategoria:"
        )
    },
    {
        "id": "zero_shot_cot",
        "template": (
            "Sua tarefa é classificar uma demanda de ouvidoria.\n"
            "Explique seu raciocínio e depois forneça a categoria final.\n"
            "Categorias possíveis: 'Elogio', 'Crítica', 'Sugestão', "
            "'Pedido de LAI', 'Solicitação', 'Denúncia'.\n\n"
            "Demanda: \"{demanda}\"\nCategoria:"
        )
    }
]
