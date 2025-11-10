import os
import logging
import numpy as np
from typing import List
from multiprocessing import Process, Queue, cpu_count

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
        self.processes = [] # processos_em_execucao
        self.dados_coletados = []

        # Se não for passo o número de processos que será utilizado para a atrativa dos dados irá ser utilizado uma função do python que verifica 
        # o número total de núcleos (cores) físicos + lógicos disponíveis no seu processador. Ela é usada para saber quantos processos podem rodar
        # em paralelo de forma eficiente.
        self.num_processes = num_processes if num_processes else cpu_count()//2
        
    def _validar_linha(self, row: pd.Series) -> bool:
        """
        Valida linhas do arquivo CSV.

        Colunas CSV:

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

        Args:
            row (pd.Series): Linha do DataFreme

        Returns:
            bool: Se ele entregar em alguma condição o retorno é False, ou seja, não pode ser utilizado a row
            que entrou na função, se ele não atender a nenhuma condição ele retorna True, ou seja, a row
            que entrou na função pode ser utilizada.
        """
        try:

            # Verificar se o campo ´FRP´ é NaN
            if pd.isna(row['FRP']):
                return False

            # Verificar se campos obrigatórios não são nulos
            campos_obrigatorios = ['Latitude', 'Longitude', 'DataHora', 'Satelite']
            if row[campos_obrigatorios].isnull().any():
                return False
            
            # Validar se a linha se fere ao Brasil (Amazônia)
            if not row['Pais'] == 'Brasil':
                return False
            
            # Validar FRP (Fire Radiative Power) se existir
            if pd.notna(row['FRP']):
                if row['FRP'] < 0:
                    return False
            
            # Validar data
            try:
                pd.to_datetime(row['DataHora'])
            except:
                return False
            
            # Validar Bioma
            if not row['Bioma'] == 'Amazônia':
                return False

            return True
            
        except Exception as e:
            logger.warning(f"Erro na validação: {e}")
            return False
    
    def _processar_arquivo(self, csv_file: str, queue: Queue):
        """
            Está função é utilizada para abrir um arquivo CSV por partes para que a tartativa dos dados aconteça mais rapidamente.
            A quantidade de linhas carregadas por vez é determinada na variável ´chunk_size´, com isso é feito a iteração pelas linhas,
            essas linhas são passadas para uma função que vai fazer algumas validações, se passar por todas ela retorna True se não False.
            Se o resultado por True ele adiciona o valor a Fila se não ele vai para a próxima linha.

        Args:
            csv_file (str): path do arquivo csv que será tratado.
            queue (Queue): Fila que será utilizado para colocar os dados que atendem aos requisitos.
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

                        row = self._codigos_municipio(row=row)

                        queue.put(row.to_dict())
                        linhas_validas += 1
                    else:
                        linhas_invalidas += 1
            
            logger.info(f"Arquivo {csv_file} processado: {linhas_validas} válidas, {linhas_invalidas} inválidas")
            
        except Exception as e:
            logger.error(f"Erro ao processar {csv_file}: {e}")


    def _engenharia_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Está função é utilizada para criar as features com base nas colunas do DataFreme original,
            essas features são criadas para que o modelo tenha mais conhecimento sobre os dados e ele
            consiga predizer o dado determiando como target de uma forma mais eficiente.

        Args:
            df (pd.DataFrame): DataFreme que será utilizado para criar as features.

        Returns:
            pd.DataFrame: DataFreme com as features criadas.
        """
    
        logger.info("Aplicando engenharia de features...")
        df = df.copy()

        df['Data'] = pd.to_datetime(df['Data'])

        # ===== FEATURES TEMPORAIS =====
        df['Ano'] = df['Data'].dt.year
        df['Mes'] = df['Data'].dt.month
        df['Dia'] = df['Data'].dt.day
        df['DiaAno'] = df['Data'].dt.dayofyear

        # ===== FEATURES LÓGICAS =====
        # Nós criamos essas features para que o modelo entenda que os valores 'Dia' e 'Mes'
        # não são valores continuos, eles tem uma lógica por trás. 
        # A forma com que implementamos isso é como se criassemos um relógio onde cada valor 
        # tem sua posição dentro dele e toda vez que o ultimo valor do relógio é atingido
        # o ciclo se reinicia e volta para o valor inicial.
        # Criamos essa features por que se mantivessemos as features somente como 'Dia' e 'Mes'
        # o modelo iria entender esses valores como valores continuos, mas na verdade não são.
        df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
        df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
        df['DiaAno_sin'] = np.sin(2 * np.pi * df['DiaAno'] / 365)
        df['DiaAno_cos'] = np.cos(2 * np.pi * df['DiaAno'] / 365)

        # ===== FEATURES DE INTERAÇÃO =====
        # Essa feature é importante para determinar o Risco das queimadas, ou seja, se estiver tendo fogo e chuva 
        # o perigo não é tão alto, agora se estiver com Risco de Fogo e não estiver chovendo o perigo é grande.
        # Então seria uma forma de entender qual o Risco da queimada que está ou irá acontecer.
        df['RiscoFogo_x_DiaSemChuva'] = df['RiscoFogo'] * df['DiaSemChuva']

        # ===== FEATURES EXPANÇÃO POLINOMIAL =====
        # Utilizamos o conceito de expansão polinomial para entendermos a não linearidade dos dados e o modelo 
        # conseguir predizer de uma forma mais acertiva.
        df['RiscoFogo_squared'] = df['RiscoFogo'] ** 2
        df['DiaSemChuva_squared'] = df['DiaSemChuva'] ** 2

        # ===== FEATURES GEOGRÁFICAS NORMALIZADAS =====
        # Aqui normalizamos a Latitude e Longitude para igualar a importância numérica, assim o modelo não entende que 
        # a Latitude e Longitude tem mais importância do que os demais parametros.
        df['Latitude_norm'] = (df['Latitude'] - df['Latitude'].min()) / (df['Latitude'].max() - df['Latitude'].min())
        df['Longitude_norm'] = (df['Longitude'] - df['Longitude'].min()) / (df['Longitude'].max() - df['Longitude'].min())
        
        # ===== FEATURES DE MÉDIA MÓVEL =====
        df['RiscoFogo_media_movel_7'] = df.groupby('Municipio')['RiscoFogo'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['Precipitacao_media_movel_7'] = df.groupby('Municipio')['Precipitacao'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['DiaSemChuva_media_movel_14'] = df.groupby('Municipio')['DiaSemChuva'].transform(
            lambda x: x.rolling(window=14, min_periods=1).mean()
        )
        
        # ===== FEATURES DE VOLATILIDADE =====
        df['RiscoFogo_volatilidade_7'] = df.groupby('Municipio')['RiscoFogo'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std().fillna(0)
        )
        df['Precipitacao_volatilidade_7'] = df.groupby('Municipio')['Precipitacao'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std().fillna(0)
        )
        
        # ===== FEATURES DE EXTREMOS =====
        df['RiscoFogo_max_14'] = df.groupby('Municipio')['RiscoFogo'].transform(
            lambda x: x.rolling(window=14, min_periods=1).max()
        )
        df['Precipitacao_min_7'] = df.groupby('Municipio')['Precipitacao'].transform(
            lambda x: x.rolling(window=7, min_periods=1).min()
        )
        
        # ===== FEATURES DE ACUMULAÇÃO =====
        df['Precipitacao_acumulada_7'] = df.groupby('Municipio')['Precipitacao'].transform(
            lambda x: x.rolling(window=7, min_periods=1).sum()
        )
        df['Precipitacao_acumulada_30'] = df.groupby('Municipio')['Precipitacao'].transform(
            lambda x: x.rolling(window=30, min_periods=1).sum()
        )
        
        logger.info(f"✓ Features avançadas criadas com sucesso! Total: {df.shape[1]}")
        
        return df
    
    def _agregar_por_dia_municipio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Nesta função será feito a agregação dos dados apartir de dia por data e municipio, ou seja, 
            para cada dia que o datafreme tem ele vai separar as linhas de um unico dia e municipio e efetuar
            alguns calculos que são passados na função: agg.

        Args:
            df (pd.DataFrame): DataFreme onde será feito a agregação dos dados.

        Returns:
            pd.DataFrame: DataFreme com os dados agregados.
        """
        df = df.copy()
        df['DataHora'] = pd.to_datetime(df['DataHora'])
        df['Data'] = df['DataHora'].dt.date

        df_daily = df.groupby(['Data', 'Municipio']).agg({
            'FRP': 'sum',
            
            # Para RiscoFogo: média apenas dos valores válidos
            'RiscoFogo': lambda x: x[x != -999.0].mean(),
            
            # Para DiaSemChuva: máximo apenas dos valores válidos
            'DiaSemChuva': lambda x: x[x != -999.0].max(),
            
            # Para Precipitacao: média apenas dos valores válidos
            'Precipitacao': lambda x: x[x != -999.0].mean(),
            
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()

        df_daily = df_daily.dropna()

        df['Data'] = pd.to_datetime(df['Data'])

        return df_daily
    
    def _criar_categorias_risco(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria categorias de risco baseado em FRP.
        Baixo: FRP < 100
        Médio: 100 <= FRP < 500
        Alto: FRP >= 500
        """
        df = df.copy()
        
        def categorizar_frp(frp):
            if frp < 100:
                return 'Baixo'
            elif frp < 500:
                return 'Médio'
            else:
                return 'Alto'
        
        df['Categoria_Risco'] = df['FRP'].apply(categorizar_frp)
        
        print("\n" + "="*60)
        print("DISTRIBUIÇÃO DE CATEGORIAS")
        print("="*60)
        print(df['Categoria_Risco'].value_counts())
        print("="*60 + "\n")
        
        return df

    def processar_todos(self) -> pd.DataFrame:
        """
            Essa é a função principal da class, é onde será tratado todos os dados para gerar um arquivo CSV final
            que será utilizado para criar o modelo de ML.

        Returns:
            pd.DataFrame: DataFreme com os dados tratados.
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

            if not self.data_queue.empty():
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


        logger.info(f"Total de linhas válidas coletadas: {len(self.dados_coletados)}")

        # Criar DataFrame
        df = pd.DataFrame(self.dados_coletados)
        
        if df.empty:
            logger.warning("Nenhum dado válido foi coletado!")
            return pd.DataFrame()
        
        # Agregar
        df = self._agregar_por_dia_municipio(df=df)

        # Criar categorias
        df = self._criar_categorias_risco(df=df)

        # Engenharia de features
        df = self._engenharia_features(df=df)

        return df


    def _ordenar_lista(self, lista: list) -> List[str]:

        lista_ordenada = list()

        for valor in (lista):
            try:
                valor =  int(valor[6:10])
            except ValueError:
                logger.error(f"Erro ao carregar o CSV: {valor}")
                continue

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

    def _codigos_municipio(self, row: pd.Series) -> pd.Series:
        municipios = {
            'Alvarães': 1300029,
            'Amaturá': 1300060,
            'Anamã': 1300086,
            'Anori': 1300102,
            'Apuí': 1300144,
            'Atalaia Do Norte': 1300201,
            'Autazes': 1300300,
            'Barcelos': 1300409,
            'Barreirinha': 1300508,
            'Benjamin Constant': 1300607,
            'Beruri': 1300631,
            'Boa Vista Do Ramos': 1300680,
            'Boca Do Acre': 1300706,
            'Borba': 1300805,
            'Caapiranga': 1300839,
            'Canutama': 1300904,
            'Carauari': 1301001,
            'Careiro': 1301100,
            'Careiro Da Várzea': 1301159,
            'Coari': 1301209,
            'Codajás': 1301308,
            'Eirunepé': 1301407,
            'Envira': 1301506,
            'Fonte Boa': 1301605,
            'Guajará': 1301654,
            'Humaitá': 1301704,
            'Ipixuna': 1301803,
            'Iranduba': 1301852,
            'Itacoatiara': 1301902,
            'Itamarati': 1301951,
            'Itapiranga': 1302009,
            'Japurá': 1302108,
            'Juruá': 1302207,
            'Jutaí': 1302306,
            'Lábrea': 1302405,
            'Manacapuru': 1302504,
            'Manaquiri': 1302553,
            'Manaus': 1302603,
            'Manicoré': 1302702,
            'Maraã': 1302801,
            'Maués': 1302900,
            'Nhamundá': 1303007,
            'Nova Olinda Do Norte': 1303106,
            'Novo Airão': 1303205,
            'Novo Aripuanã': 1303304,
            'Parintins': 1303403,
            'Pauini': 1303502,
            'Presidente Figueiredo': 1303536,
            'Rio Preto Da Eva': 1303569,
            'Santa Isabel Do Rio Negro': 1303601,
            'Santo Antônio Do Içá': 1303700,
            'São Gabriel Da Cachoeira': 1303809,
            'São Paulo De Olivença': 1303908,
            'São Sebastião Do Uatumã': 1303957,
            'Silves': 1304005,
            'Tabatinga': 1304062,
            'Tapauá': 1304104,
            'Tefé': 1304203,
            'Tonantins': 1304237,
            'Uarini': 1304260,
            'Urucará': 1304302,
            'Urucurituba': 1304401,
        }

        row['Municipio'] = municipios[row['Municipio'].title()]

        return row
    