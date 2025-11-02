from src.tratamento_dados import abrir_arquivos_csv
from src.fila import DBQueimadasProcessor

if __name__ == '__main__':

    csv_files = abrir_arquivos_csv()

    # csv_files = ['dados_2020.csv']

    processor = DBQueimadasProcessor(csv_files, num_processes=2)
    
    # 3. Processar dados
    df_final = processor.processar_todos(aplicar_agregacao=False)
