from src.tratando_dados import Tratando_Dados

import os


if __name__ == '__main__':

    processor = Tratando_Dados(num_processes=2)
    
    df_final = processor.processar_todos()

    path = os.getcwd() + '/dbqueimadas_CSV'

    df_final.to_csv(path+'/df_final.csv', index=False)
