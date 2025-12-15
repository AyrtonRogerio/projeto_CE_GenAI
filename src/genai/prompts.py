CATEGORIAS_VALIDAS = [
    "Elogio",
    "Crítica",
    "Sugestão",
    "Pedido de LAI",
    "Solicitação",
    "Denúncia"
]


# PROMPT 1 — ZERO-SHOT DIRETO

PROMPT_ZERO_SHOT_DIRETO = {
    "id": "zero_shot_direto",
    "template": (
        "Você é um classificador especializado em manifestações de ouvidoria.\n"
        "Sua tarefa é ler o texto fornecido e determinar a categoria exata.\n\n"
        "Siga rigorosamente estas regras:\n"
        "1. Escolha apenas uma categoria.\n"
        "2. Responda SOMENTE com a categoria, sem explicações.\n"
        "3. As categorias possíveis são:\n"
        "- Elogio\n"
        "- Crítica\n"
        "- Sugestão\n"
        "- Pedido de LAI\n"
        "- Solicitação\n"
        "- Denúncia\n\n"
        "Texto da manifestação:\n\"{demanda}\"\n\n"
        "Resposta (apenas a categoria):"
    )
}


# PROMPT 2 — ZERO-SHOT COM CADEIA DE RACIOCÍNIO (CoT)

PROMPT_ZERO_SHOT_COT = {
    "id": "zero_shot_cot",
    "template": (
        "Você é um classificador especializado em manifestações de ouvidoria.\n"
        "Analise o texto, apresente um raciocínio breve e determine a categoria.\n\n"
        "Categorias permitidas:\n"
        "- Elogio\n"
        "- Crítica\n"
        "- Sugestão\n"
        "- Pedido de LAI\n"
        "- Solicitação\n"
        "- Denúncia\n\n"
        "IMPORTANTE:\n"
        "1. Primeiro explique seu raciocínio em poucas linhas.\n"
        "2. Depois forneça a resposta final no formato:\n"
        "Categoria Final: <nome_da_categoria>\n"
        "3. Use somente as categorias acima.\n\n"
        "Texto:\n\"{demanda}\"\n\n"
        "Raciocínio:"
    )
}

# PROMPT 3 — FEW-SHOT


FEWSHOT_EXEMPLOS = """
[Exemplo — Elogio]
Texto: "O cidadão compareceu ao atendimento presencial desta Ouvidoria, realizou consulta de sua demanda, agradeceu pelo encaminhamento ao MPCO, 
se mostrou muito satisfeito, elogiou bastante toda equipe de Ouvidoria do ***-**, pela dedicação, acolhimento, atenção e pelo excelente 
trabalho desempenhado por todos."
Categoria: Elogio

[Exemplo — Crítica]
Texto: "Referente a manifestação de nº ****.
A cidadã entrou em contato pelo Disque Ouvidoria, pois não concordou com a resposta dada a sua manifestação, informando que existem 
deliberações do ***-**, que tratam dos fatos elencados na denúncia nº ****."
Categoria: Crítica

[Exemplo — Sugestão]
Texto: "Sugestões para a migração do Sistema **********:

1 - Ambiente de treinamento totalmente em português, há pessoas que não dominam o inglês e isso cria uma resistência adicional para 
uma atividade que já é desafiadora.
2 - A parte do API de dados, melhorar a explicação do que se trata para facilitar a compreensão de quem não domina linguagem técnica 
de tecnologia da informação.
3 - Migrações de sistemas corporativos de grande escala naturalmente geram considerável sobrecarga as áreas demandadas (contudo o prazo 
inicial de migração do sistema foi praticamente inexequível, a dilação de prazo auxiliou a questão, porém continua extremamente desafiadora, 
o que tornou o processo angustiante). Tentar viabilizar uma rotina de carga adicional de trabalho de modo progressivo: considerar uma migração 
de médio prazo e adotar um perfil de regulação responsiva no primeiro ano de migração em detrimento de uma abordagem punitiva com multas. 
Dessa forma incentivar e encorajar as pessoas que estão em processo de migração e tentando atender os prazos.
4 - criação de treinamentos específicos, para cada área, processos de contratações, instrumentos jurídicos e sobre tudo, obras, 
com casos mais próximos da realidade.

Desde já agradeço a atenção!"
Categoria: Sugestão

[Exemplo — Pedido de LAI]
Texto: "Prezados auditores do Tribunal de Contas,

Solicito informações sobre o andamento da Auditora Especial que será realizada em virtude do Processo TC nº ********-*.
Em qual fase a auditoria está atualmente? Existe alguma previsão para a publicação do Relatório de Auditoria?

Cordialmente."
Categoria: Pedido de LAI

[Exemplo — Solicitação]
Texto: "Prezados, 
Cordiais cumprimentos. 
À luz dos princípios constitucionais, bem como das leis que regem o direito à informação, sirvo-me do presente para solicitar o esclarecimento
 de uma dúvida: gostaria de saber qual a previsão de servidores aptos a se aposentar no cargo de analista de gestão - administração, 
 para os próximos dois anos.
Atenciosamente,"
Categoria: Solicitação

[Exemplo — Denúncia]
Texto: "A Prefeitura de ********, através do edital ** ***/**** - CONTRATAÇÃO DE EMPRESA DE ENGENHARIA PARA EXECUÇÃO DO CAPEAMENTO DE VIAS 
NO MUNICÍPIO DE **********/**, cuja abertura será dia **/**/**** às 11, EXIGE no item3.4 que as empresas interessadas 
em participar do processo envie a garantia de proposta para o *E-MAIL* da Comissão, podendo ser desclassificada. 
Informo também que a Habilitação foi solicitada no portal antes do momento da abertura, ou seja, os licitantes 
já terão o dever de apresentar a proposta neste momento. O envio da garantia para o E-MAIL da Comissão só servira para 
IDENTIFICAÇÃO PRÉVIA de quem vai participar da Licitação, visto que apenas o envio da garantia não assegura absolutamente 
nada como "prévia habilitação". Apenas deixa o processo mais inseguro, indo completamente de encontro ao que a nova lei prega."
Categoria: Denúncia
"""

PROMPT_FEW_SHOT = {
    "id": "few_shot_completo",
    "template": (
        "Você é um sistema especialista em classificação de manifestações de ouvidoria.\n"
        "A seguir estão exemplos reais já classificados:\n\n"
        f"{FEWSHOT_EXEMPLOS}\n"
        "Agora classifique a manifestação abaixo.\n\n"
        "Categorias possíveis:\n"
        "- Elogio\n"
        "- Crítica\n"
        "- Sugestão\n"
        "- Pedido de LAI\n"
        "- Solicitação\n"
        "- Denúncia\n\n"
        "Texto:\n\"{demanda}\"\n\n"
        "Resposta (somente a categoria):"
    )
}


# PROMPT 4 — META-PROMPT (Pedindo para a propria LLM qual é o melhor prompt)

PROMPT_META = {
    "id": "meta_prompt_classificacao",
    "template": (
        "Você é um especialista em engenharia de prompts e classificação de ouvidoria.\n"
        "Sua tarefa é TRÊS PARTES:\n\n"
        "1.  Crie o MELHOR prompt possível para classificar a manifestação abaixo (gere o prompt ideal).\n"
        "2.  Use o prompt ideal que você criou na parte 1 para analisar a Manifestação e classificar.\n"
        "3.  Forneça a categoria final da manifestação. Responda APENAS com a categoria.\n\n"
        "Categorias disponíveis: Elogio, Crítica, Sugestão, Pedido de LAI, Solicitação, Denúncia.\n\n"
        "Manifestação a ser classificada:\n\"{demanda}\"\n\n"
        "Resposta (APENAS a categoria final):"
    )
}


PROMPT_STRATEGIES = [
    PROMPT_ZERO_SHOT_DIRETO,
    PROMPT_ZERO_SHOT_COT,
    PROMPT_FEW_SHOT,
    PROMPT_META,
]
