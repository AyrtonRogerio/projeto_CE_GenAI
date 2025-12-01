import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)


from src.genai.benchmark_runner import run_benchmark


INPUT_FILE_PATH = os.path.join(project_root, 'data', 'raw', 'ouvidoria_sintetico.csv')

def main():
    print("--- Iniciando Pipeline de Benchmark GenAI ---")
    print(f"Diretório Raiz do Projeto: {project_root}")
    print(f"Arquivo de entrada: {INPUT_FILE_PATH}")


    if not os.path.exists(INPUT_FILE_PATH):
        print(f"\nERRO: Arquivo não encontrado em: {INPUT_FILE_PATH}")
        print("Verifique se o nome do arquivo está correto na pasta 'data/raw/'.")
        return


    try:
        run_benchmark(INPUT_FILE_PATH)
    except Exception as e:
        print(f"\nOcorreu um erro crítico durante a execução: {e}")

if __name__ == "__main__":
    main()