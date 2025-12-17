import pandas as pd
import os


FILE_PATH = "/home/ayrton/IdeaProjects/projeto_CE_GenAI/outputs/dados_validacao_humana/amostra_para_validacao_humana.csv"

def limpar_nome_coluna(col):
    #Remove quebras de linha e espaços extras dos nomes das colunas
    return col.replace('\n', ' ').replace('\r', '').strip()

def normalizar_sim_nao(valor):
    #Converte Sim/S/1 para 1 e Não/N/0 para 0
    if pd.isna(valor): return 0
    s = str(valor).strip().lower()
    if s.startswith('s') or s == '1': return 1
    return 0

def main():
    print("PROCESSANDO ESTATÍSTICAS DA VALIDAÇÃO")

    if not os.path.exists(FILE_PATH):
        print(f"Arquivo não encontrado: {FILE_PATH}")
        return

    # Carregar Dataset
    try:
        df = pd.read_csv(FILE_PATH, sep=',')
        if df.shape[1] < 2:
            df = pd.read_csv(FILE_PATH, sep='|')
        if df.shape[1] < 2:
            df = pd.read_csv(FILE_PATH, sep=';')
    except Exception as e:
        print(f"Falha ao ler CSV: {e}")
        return

    # Limpar Cabeçalho

    df.columns = [limpar_nome_coluna(c) for c in df.columns]

    print(f"Colunas detectadas: {df.columns.tolist()}")

    # Mapear Colunas
    try:

        col_rotulo = [c for c in df.columns if 'Rotulo' in c or 'Rótulo' in c][0]
        col_realismo = [c for c in df.columns if 'Realismo' in c][0]
        col_vazamento = [c for c in df.columns if 'Vazamento' in c][0]
        col_complex = [c for c in df.columns if 'Complexidade' in c][0]
    except IndexError as e:
        print(f"Não conseguiu encontrar alguma coluna")
        return


    total = len(df)

    # Consistência
    df['consist_bin'] = df[col_rotulo].apply(normalizar_sim_nao)
    acc = df['consist_bin'].mean() * 100

    # Realismo
    df['real_num'] = pd.to_numeric(df[col_realismo], errors='coerce')
    media_real = df['real_num'].mean()
    # Alta qualidade (>=4)
    alta_qual = df[df['real_num'] >= 4].shape[0]
    perc_alta = (alta_qual / total) * 100

    # Vazamento
    df['vaz_bin'] = df[col_vazamento].apply(normalizar_sim_nao)
    taxa_vaz = df['vaz_bin'].mean() * 100

    # Complexidade
    df['comp_bin'] = df[col_complex].apply(normalizar_sim_nao)
    taxa_comp = df['comp_bin'].mean() * 100


    print("-" * 50)
    print(f"RESULTADOS (n={total})")
    print("-" * 50)
    print(f"1. Consistência do Rótulo:  {acc:.2f}%")
    print(f"2. Média de Realismo:       {media_real:.2f} / 5.0")
    print(f"   -> Alta Qualidade (>=4): {perc_alta:.2f}%")
    print(f"3. Taxa de Vazamento:       {taxa_vaz:.2f}%")
    print(f"4. Taxa de Complexidade:    {taxa_comp:.2f}%")
    print("-" * 50)



if __name__ == "__main__":
    main()