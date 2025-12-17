import pandas as pd
import os
import csv
import re


INPUT_FILE = "/home/ayrton/IdeaProjects/projeto_CE_GenAI/data/raw/ouvidoria_sintetico.csv"
OUTPUT_DIR = "/home/ayrton/IdeaProjects/projeto_CE_GenAI/outputs/dados_validacao_humana"
OUTPUT_FILENAME = "amostra_para_validacao_humana.csv"
FULL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)


AMOSTRAS_POR_CLASSE = 8

def limpar_texto(texto):
    #Remove quebras de linha e pipes que possam quebrar o CSV de saída
    if not isinstance(texto, str):
        return str(texto)
    # Remove quebras de linha
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    # Substitui | (PIPE) por / (barra) no texto
    texto = texto.replace('|', '/')
    # Remove espaços duplos
    texto = re.sub(' +', ' ', texto)
    return texto.strip()

def main():
    print(">>> Gerando Amostra (Leitura TAB -> Saída PIPE) <<<")
    print(f"Lendo dataset: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"Arquivo não encontrado: {INPUT_FILE}")
        return


    try:
        df = pd.read_csv(
            INPUT_FILE,
            sep='\t',
            encoding='utf-8',
            engine='python',
            on_bad_lines='warn'
        )
    except Exception as e:
        print(f"[ERRO CRÍTICO] Falha na leitura: {e}")
        return

    # Validação
    if df.shape[1] < 2:
        print("O arquivo foi lido, mas continua com 1 coluna.")
        print("Verifique se o arquivo realmente usa TAB.")
        print(f"Colunas lidas: {df.columns.tolist()}")
        return

    print(f"Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas.")

    # Normalizar Nomes das Colunas
    df.columns = df.columns.str.strip()

    col_id = 'ID'
    col_cat = 'Categoria'
    col_txt = 'Texto da Demanda'

    # Fallback para encontrar a coluna de texto
    if col_txt not in df.columns:
        possiveis = [c for c in df.columns if 'texto' in c.lower()]
        if possiveis: col_txt = possiveis[0]

    if col_txt not in df.columns:
        print(f"Coluna de texto não encontrada. Colunas: {df.columns.tolist()}")
        return

    # Limpeza do Texto
    print("Normalizando textos (removendo enters e pipes internos)...")
    df[col_txt] = df[col_txt].apply(limpar_texto)

    # Amostragem Estratificada
    amostras = []
    if col_cat in df.columns:
        classes = df[col_cat].unique()
        print(f"Classes encontradas: {classes}")
        for cls in classes:
            df_cls = df[df[col_cat] == cls]
            n = min(len(df_cls), AMOSTRAS_POR_CLASSE)
            amostras.append(df_cls.sample(n=n, random_state=42))
        df_final = pd.concat(amostras)
    else:
        df_final = df.sample(n=AMOSTRAS_POR_CLASSE*6, random_state=42)

    
    # Seleciona colunas existentes
    cols_out = [c for c in [col_id, col_cat, col_txt] if c in df_final.columns]
    df_final = df_final[cols_out].copy()

    
    mapa_nomes = {col_id: 'ID_Original', col_cat: 'Categoria_Original', col_txt: 'Texto_Manifestacao'}
    df_final.rename(columns=mapa_nomes, inplace=True)

    # Adicionando Colunas de Validação
    df_final['Rotulo Correto (Sim/Não)'] = ""
    df_final['Realismo 1 a 5 \n(1-Muito Artificial 2- Artificial 3- Regular 4- Natural 5- Muito Natural)'] = ""
    df_final['Tem Vazamento (Sim/Não)\n Ex. Elogio, Denúncia '] = ""
    df_final['Complexidade (Sim/Não)'] = ""
    df_final['Obs'] = ""

    
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    #SALVAR COM SEPARADOR PIPE
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        df_final.to_csv(
            FULL_OUTPUT_PATH,
            sep='|',             
            index=False,
            encoding='utf-8-sig',
            quoting=csv.QUOTE_ALL 
        )
        print("-" * 50)
        print(f"Arquivo salvo em:\n{FULL_OUTPUT_PATH}")
        print("Separador utilizado: PIPE (|)")
        print("-" * 50)
    except Exception as e:
        print(f"Falha ao salvar: {e}")

if __name__ == "__main__":
    main()


