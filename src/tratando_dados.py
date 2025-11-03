import os
import logging
from time import sleep
from typing import List
from multiprocessing import Process, Queue, cpu_count

import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Tratando_Dados:
    """Processador de dados do DBQueimadas com multiprocessing"""
    
    def __init__(self, num_processes: int = None):
        """
        Args:
            num_processes: Número de processos (padrão: número de CPUs)
        """
        
        # Criar lista com os nomes do arquivos CSV que seram tratados.
        self.csv_files = self._abrir_arquivos_csv()
        
        # Cria a Queue que será utilizada para fazer o redirecionamento de dados que atendem as validações para uma lista.
        self.data_queue = Queue()

        # Variaveis utilizadas para controlar a quantidade de processos ativos, finalizados e dados coletados através do processos rodando ou rodados.
        self.processos_finalizados = 0
        self.processos_ativos = 0
        self.processes = []
        self.dados_coletados = []

        # Se não for passo o número de processos que será utilizado para a atrativa dos dados irá ser utilizado uma função do python que verifica 
        # o número total de núcleos (cores) físicos + lógicos disponíveis no seu processador. Ela é usada para saber quantos processos podem rodar
        # em paralelo de forma eficiente.
        self.num_processes = num_processes or cpu_count()
        
    def _validar_linha(self, row: pd.Series) -> bool:
        """
        Valida cada linha do CSV
        Customize esta função com suas regras de validação
        
        Campos CSV:

                DataHora    
                Satelite    
                Pais        
                Estado      
                Municipio   
                Bioma       
                DiaSemChuva 
                Precipitacao
                RiscoFogo   
                FRP         
                Latitude    
                Longitude   

        """
        try:
            # Exemplo de validações - AJUSTE CONFORME SEUS DADOS
            
            # 1. Verificar se campos obrigatórios não são nulos
            campos_obrigatorios = ['Latitude', 'Longitude', 'DataHora', 'Satelite']
            if row[campos_obrigatorios].isnull().any():
                return False
            
            # 2. Validar se a linha se fere ao Brasil (Amazônia)
            if not row['Pais'] == 'Brasil':
                return False
            
            # 3. Validar FRP (Fire Radiative Power) se existir
            if pd.notna(row['FRP']):
                if row['FRP'] < 0 or row['FRP'] > 10000:
                    return False
            
            # 4. Validar data
            try:
                pd.to_datetime(row['DataHora'])
            except:
                return False            
            
            # 5. Validar Bioma
            if not row['Bioma'] == 'Amazônia':
                return False

            return True
            
        except Exception as e:
            logger.warning(f"Erro na validação: {e}")
            return False
    
    def _processar_arquivo(self, csv_file: str, queue: Queue):
        """
        Processa um arquivo CSV e envia linhas válidas para a queue
        """
        try:
            logger.info(f"Processando arquivo: {csv_file}")
            
            path = os.getcwd() + '/dbqueimadas_CSV'

            # Ler CSV em chunks para economizar memória
            chunk_size = 1000
            linhas_validas = 0
            linhas_invalidas = 0
            
            for chunk in pd.read_csv(f'{path}/{csv_file}', chunksize=chunk_size, low_memory=False):
                for idx, row in chunk.iterrows():
                    if self._validar_linha(row):
                        # Enviar linha válida para a queue

                        row['DataHora'] = pd.to_datetime(row['DataHora'])

                        queue.put(row.to_dict())
                        linhas_validas += 1
                    else:
                        linhas_invalidas += 1
            
            logger.info(f"Arquivo {csv_file} processado: {linhas_validas} válidas, {linhas_invalidas} inválidas")
            
        except Exception as e:
            logger.error(f"Erro ao processar {csv_file}: {e}")
        

    def _engenharia_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica engenharia de features nos dados
        """
        logger.info("Aplicando engenharia de features...")
        
        df_features = df.copy()
        
        # 1. Features temporais
        if 'DataHora' in df_features.columns:
            df_features['ano'] = df_features['DataHora'].dt.year
            df_features['mes'] = df_features['DataHora'].dt.month
            df_features['dia'] = df_features['DataHora'].dt.day
            df_features['dia_semana'] = df_features['DataHora'].dt.dayofweek
            df_features['hora'] = df_features['DataHora'].dt.hour
            df_features['dia_ano'] = df_features['DataHora'].dt.dayofyear


        # 2. Features geográficas
        if 'Latitude' in df_features.columns and 'Latitude' in df_features.columns:
            # Região aproximada
            df_features['regiao'] = df_features.apply(lambda row:
                    'norte' if row['Latitude'] >= -10 and row['Longitude'] <= -50 else
                    'nordeste' if row['Latitude'] >= -15 and row['Longitude'] > -50 else
                    'centro_oeste' if -20 <= row['Latitude'] < -10 and -60 <= row['Longitude'] <= -45 else
                    'sudeste' if -25 <= row['Latitude'] < -15 and -50 <= row['Longitude'] <= -39 else
                    'sul',
                    axis=1
                )
        
        # 3. Features de FRP (Fire Radiative Power)
        if 'FRP' in df_features.columns:
            df_features['frp_log'] = np.log1p(df_features['FRP'])
            df_features['frp_categoria'] = pd.cut(
                df_features['FRP'], 
                bins=[0, 10, 50, 100, 500, float('inf')],
                labels=['muito_baixo', 'baixo', 'medio', 'alto', 'muito_alto']
            )
        
        # 4. Encoding de variáveis categóricas (Strings)
        # Essa engenharia de Features é feita por que o modelo não entende 'baixo', 'muito baixo' etc...
        # Então transformamos essas categorias em números para que o modelo possa interpretá-los corretamente.
        categorical_cols = df_features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'DataHora':
                df_features[col] = df_features[col].astype('category').cat.codes
        
        logger.info(f"Features criadas. Shape final: {df_features.shape}")
        return df_features
    
    def _agregar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza agregações nos dados
        """
        logger.info("Agregando por data (diário) para Amazônia...")

        df = df.copy()

        df['data'] = df['DataHora'].dt.floor('D')

        agg_dict = {}
        if 'FRP' in df.columns:
            agg_dict['FRP'] = ['count', 'sum']
        if 'Precipitacao' in df.columns:
            agg_dict['Precipitacao'] = ['mean', 'sum']
        if 'DiaSemChuva' in df.columns:
            agg_dict['DiaSemChuva'] = ['mean']
        if 'Latitude' in df.columns:
            agg_dict['Latitude'] = ['mean']
        if 'Longitude' in df.columns:
            agg_dict['Longitude'] = ['mean']

        if not agg_dict:
            logger.warning("Nenhuma coluna encontrada para agregação.")
            return df

        out = (
            df.groupby(['data'])
            .agg(agg_dict)
            .reset_index()
        )

        out.columns = ['_'.join(col).strip('_') for col in out.columns.values]

        if 'FRP_count' in out.columns:
            out = out.rename(columns={'FRP_count': 'focos_count'})
        
        return out

    
    def processar_todos(self) -> pd.DataFrame:
        """
        Processa todos os arquivos CSV com multiprocessing
        """
        logger.info(f"Iniciando processamento com {self.num_processes} processos")
        total_arquivos = len(self.csv_files)
        proximo_arquivo = 0

        while self.processos_finalizados < total_arquivos:
            
            if self.processos_ativos < self.num_processes and proximo_arquivo < total_arquivos:
                p = Process(target=self._processar_arquivo, args=(self.csv_files[proximo_arquivo], self.data_queue))
                p.start()
                self.processes.append(p)
                self.processos_ativos += 1
                proximo_arquivo += 1

            while not self.data_queue.empty():
                self.dados_coletados.append(self.data_queue.get())

            for p in self.processes:
                # O atributo .exitcode de um processo no Python (multiprocessing.Process) mostra o código de saída do processo filho depois que ele termina. 
                # Ele serve para saber se o processo ainda está rodando, se terminou com sucesso ou se deu erro.
                # Vamos utilizar ele para verificar se o processo já terminou para que seja possivel adicionar mais um processo se houver mais arquivos
                # CSV para ser tratados
                if p.exitcode is not None:
                    p.join()              

                    self.processes.remove(p)
                    self.processos_finalizados += 1

            self.processos_ativos = len(self.processes)

            sleep(0.3)

        # Aqui fazemos uma simples verificação se ficou mais alguma coisa na Queue antes de passar para o próximo passo da tratativa dos dados.
        while not self.data_queue.empty():
            self.dados_coletados.append(self.data_queue.get())

        logger.info(f"Total de linhas válidas coletadas: {len(self.dados_coletados)}")

        # Criar DataFrame
        df = pd.DataFrame(self.dados_coletados)
        
        if df.empty:
            logger.warning("Nenhum dado válido foi coletado!")
            return pd.DataFrame()
        
        # Aplicar agregação
        df = self._agregar_dados(df)
 
        # Aplicar engenharia de features
        df = self._engenharia_features(df)       
        
        return df


    def _ordenar_lista(self, lista: list) -> List[str]:

        lista_ordenada = list()

        for valor in (lista):

            valor =  int(valor[6:10])

            if not lista_ordenada:
                lista_ordenada.append(valor)
                continue

            if valor < 0:
                lista_ordenada.insert(0, valor)
            else:
                for index, valor_lista_ordenada in enumerate(lista_ordenada):
                    if (index+1) < len(lista_ordenada):
                        if index == 0 and valor < valor_lista_ordenada:
                            lista_ordenada.insert(0, valor)
                            break
                        elif valor > valor_lista_ordenada and valor < lista_ordenada[index+1]:
                            lista_ordenada.insert(index+1, valor)
                            break
                        
                    else:
                        lista_ordenada.append(valor)
                        break
        
        lista.clear()

        for index, valor in enumerate(lista_ordenada):
            del lista_ordenada[index]
            lista_ordenada.insert(index, f'dados_{valor}.csv')

        return lista_ordenada


    def _abrir_arquivos_csv(self) -> List[str]:
        path = os.getcwd() + '/dbqueimadas_CSV'

        arquivos_csv = os.listdir(path=path)

        arquivos_csv = self._ordenar_lista(lista=arquivos_csv)

        return arquivos_csv
