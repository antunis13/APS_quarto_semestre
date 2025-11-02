import os
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue, cpu_count
from typing import List
import logging
from time import sleep


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DBQueimadasProcessor:
    """Processador de dados do DBQueimadas com multiprocessing"""
    
    def __init__(self, csv_files: List[str], num_processes: int = None):
        """
        Args:
            csv_files: Lista de caminhos dos arquivos CSV
            num_processes: Número de processos (padrão: número de CPUs)
        """
        self.processos_finalizados = 0
        self.dados_coletados = []
        self.processes = []
        self.csv_files = csv_files
        self.num_processes = num_processes or cpu_count()
        self.data_queue = Queue()
        self.processos_ativos = 0
        
    def validar_linha(self, row: pd.Series) -> bool:
        """
        Valida cada linha do CSV
        Customize esta função com suas regras de validação
        
        Campos:

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
    
    def processar_arquivo(self, csv_file: str, queue: Queue):
        """
        Processa um arquivo CSV e envia linhas válidas para a queue
        """
        try:
            logger.info(f"Processando arquivo: {csv_file}")
            
            path = os.getcwd() + '/dbqueimadas_CSV'

            # Ler CSV em chunks para economizar memória
            chunk_size = 10000
            linhas_validas = 0
            linhas_invalidas = 0
            
            for chunk in pd.read_csv(f'{path}/{csv_file}', chunksize=chunk_size, low_memory=False):
                for idx, row in chunk.iterrows():
                    if self.validar_linha(row):
                        # Enviar linha válida para a queue

                        row['DataHora'] = pd.to_datetime(row['DataHora'])

                        queue.put(row.to_dict())
                        linhas_validas += 1
                    else:
                        linhas_invalidas += 1
            
            logger.info(f"Arquivo {csv_file} processado: {linhas_validas} válidas, {linhas_invalidas} inválidas")
            
        except Exception as e:
            logger.error(f"Erro ao processar {csv_file}: {e}")
        
        
    def engenharia_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        # Essa engenha de Featuresé feita por que o modelo não entende 'baixo', 'muito baixo' etc...
        # Então transformamos essas categorias em números para que o modelo possa interpretá-los corretamente.
        categorical_cols = df_features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'DataHora':
                df_features[col] = df_features[col].astype('category').cat.codes
        
        logger.info(f"Features criadas. Shape final: {df_features.shape}")
        return df_features
    
    def agregar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza agregações nos dados
        """
        logger.info("Realizando agregações...")
        
        # Exemplo de agregações - AJUSTE CONFORME NECESSÁRIO
        
        # Agregação por região e período
        if all(col in df.columns for col in ['regiao', 'ano', 'mes']):
            agg_dict = {
                'frp': ['mean', 'max', 'min', 'std', 'count'] if 'frp' in df.columns else ['count'],
                'lat': ['mean'],
                'lon': ['mean']
            }
            
            df_agregado = df.groupby(['regiao', 'ano', 'mes']).agg(agg_dict).reset_index()
            df_agregado.columns = ['_'.join(col).strip('_') for col in df_agregado.columns.values]
            
            logger.info(f"Dados agregados. Shape: {df_agregado.shape}")
            return df_agregado
        
        return df

    
    def processos(self): 
        while self.processos_finalizados < len(self.csv_files):
            # Verificar se há dados na queue
            if not self.data_queue.empty():
                self.dados_coletados.append(self.data_queue.get())
            
            processos_vivos = sum(1 for p in self.processes if p.is_alive())

            if processos_vivos < self.processos_ativos:
                # Calcular quantos terminaram
                finalizados = self.processos_ativos - processos_vivos
                self.processos_finalizados += finalizados
                self.processos_ativos = processos_vivos

            sleep(0.5)

        # Coletar dados restantes na queue
        while not self.data_queue.empty():
            self.dados_coletados.append(self.data_queue.get())
        
        # Aguardar todos os processos terminarem
        for p in self.processes:
            p.join()
        
        logger.info(f"Total de linhas válidas coletadas: {len(self.dados_coletados)}")


    def processar_todos(self, aplicar_agregacao: bool = False) -> pd.DataFrame:
        """
        Processa todos os arquivos CSV com multiprocessing
        """
        logger.info(f"Iniciando processamento com {self.num_processes} processos")

        # Criar wrapper para o método processos
        def _wrapper_processos(obj):
            obj.processos()
        
        pool_process = Process(target=_wrapper_processos, args=(self,))
        pool_process.start()

        # Criar processos para cada arquivo
        total_arquivos = len(self.csv_files)
        proximo_arquivo = 0  # ← ADICIONADO

        while self.processos_finalizados < total_arquivos:
            # Verificar se pode criar novo processo
            if (self.processos_ativos < self.num_processes and
                proximo_arquivo < total_arquivos):

                p = Process(
                    target=self.processar_arquivo, 
                    args=(self.csv_files[proximo_arquivo], self.data_queue)
                )
                p.start()
                self.processes.append(p)

                self.processos_ativos += 1
                proximo_arquivo += 1

            sleep(0.5)
        
        pool_process.join()

        # Criar DataFrame
        df = pd.DataFrame(self.dados_coletados)
        
        if df.empty:
            logger.warning("Nenhum dado válido foi coletado!")
            return pd.DataFrame()
        
        # Aplicar engenharia de features
        df = self.engenharia_features(df)
        
        # Aplicar agregação se solicitado
        if aplicar_agregacao:
            df = self.agregar_dados(df)
        
        return df
