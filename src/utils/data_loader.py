import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:

    if not os.path.exists(filepath):

        raise FileNotFoundError(f"Arquivo não encontrado: {os.path.abspath(filepath)}")

    try:

        df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')


        if len(df.columns) < 2:
            df = pd.read_csv(filepath, sep='|', on_bad_lines='skip')

        print(f"Dataset carregado: {len(df)} linhas.")


        mapa_colunas = {
            'Texto da Demanda': 'text',
            'Texto da Manifestação': 'text',
            'demanda_texto': 'text',
            'Categoria': 'label',
            'classificacao_real': 'label'
        }

        df.rename(columns=mapa_colunas, inplace=True)


        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"Aviso: Colunas esperadas não encontradas. Colunas no arquivo: {df.columns.tolist()}")

            if len(df.columns) >= 2:
                df = df.iloc[:, [0, 1]]
                df.columns = ['text', 'label']

        df = df.dropna(subset=['text', 'label'])
        return df[['text', 'label']]

    except Exception as e:
        print(f"Erro crítico ao carregar dados: {e}")
        return pd.DataFrame()