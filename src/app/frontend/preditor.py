import os
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class PredictorQueimadas:
    """
    Classe para fazer predições de risco de queimadas usando o modelo treinado.
    Usa médias calculadas de toda a fonte de dados históricos.
    """
    
    def __init__(self, caminho_modelo: str, caminho_dados: str):
        """
        Inicializa o preditor.
        
        Args:
            caminho_modelo: Caminho do arquivo .jkl com o modelo treinado
            caminho_dados: Caminho do arquivo CSV com os dados históricos
        """
        self.modelo = self._carregar_modelo(caminho_modelo)
        self.df_historico = pd.read_csv(caminho_dados)
        self.df_historico['Data'] = pd.to_datetime(self.df_historico['Data'])
        
        # Calcular médias de TODA a fonte
        self.medias_globais = self._calcular_medias_globais()
        
        # Dicionário de municípios para referência
        self.municipios = {
            1300029: "Alvarães", 1300060: "Amaturá", 1300086: "Anamã", 1300102: "Anori",
            1300144: "Apuí", 1300201: "Atalaia Do Norte", 1300300: "Autazes", 1300409: "Barcelos",
            1300508: "Barreirinha", 1300607: "Benjamin Constant", 1300631: "Beruri",
            1300680: "Boa Vista Do Ramos", 1300706: "Boca Do Acre", 1300805: "Borba",
            1300839: "Caapiranga", 1300904: "Canutama", 1301001: "Carauari", 1301100: "Careiro",
            1301159: "Careiro Da Várzea", 1301209: "Coari", 1301308: "Codajás", 1301407: "Eirunepé",
            1301506: "Envira", 1301605: "Fonte Boa", 1301654: "Guajará", 1301704: "Humaitá",
            1301803: "Ipixuna", 1301852: "Iranduba", 1301902: "Itacoatiara", 1301951: "Itamarati",
            1302009: "Itapiranga", 1302108: "Japurá", 1302207: "Juruá", 1302306: "Jutaí",
            1302405: "Lábrea", 1302504: "Manacapuru", 1302553: "Manaquiri", 1302603: "Manaus",
            1302702: "Manicoré", 1302801: "Maraã", 1302900: "Maués", 1303007: "Nhamundá",
            1303106: "Nova Olinda Do Norte", 1303205: "Novo Airão", 1303304: "Novo Aripuanã",
            1303403: "Parintins", 1303502: "Pauini", 1303536: "Presidente Figueiredo",
            1303569: "Rio Preto Da Eva", 1303601: "Santa Isabel Do Rio Negro",
            1303700: "Santo Antônio Do Içá", 1303809: "São Gabriel Da Cachoeira",
            1303908: "São Paulo De Olivença", 1303957: "São Sebastião Do Uatumã", 1304005: "Silves",
            1304062: "Tabatinga", 1304104: "Tapauá", 1304203: "Tefé", 1304237: "Tonantins",
            1304260: "Uarini", 1304302: "Urucará", 1304401: "Urucurituba",
        }
        
        print(f"✓ Preditor inicializado com sucesso!")
        print(f"  Modelo carregado: {caminho_modelo}")
        print(f"  Dados históricos: {len(self.df_historico)} linhas")
        print(f"  Municípios com médias: {len(self.medias_globais)}")
    
    @staticmethod
    def _carregar_modelo(caminho_modelo: str):
        """Carrega o modelo salvo."""
        try:
            modelo = joblib.load(caminho_modelo)
            logger.info(f"✓ Modelo carregado: {caminho_modelo}")
            return modelo
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _calcular_medias_globais(self) -> pd.DataFrame:
        """
        Calcula as médias de TODA a fonte histórica por DiaAno e Município.
        
        Returns:
            DataFrame com colunas: DiaAno, Municipio, RiscoFogo_media, 
                                   DiaSemChuva_media, Precipitacao_media
        """
        df = self.df_historico.copy()
        df['DiaAno'] = df['Data'].dt.dayofyear
        
        medias = df.groupby(['DiaAno', 'Municipio']).agg({
            'RiscoFogo': 'mean',
            'DiaSemChuva': 'mean',
            'Precipitacao': 'mean',
            'Latitude': 'first',
            'Longitude': 'first',
        }).reset_index()
        
        medias.rename(columns={
            'RiscoFogo': 'RiscoFogo_media',
            'DiaSemChuva': 'DiaSemChuva_media',
            'Precipitacao': 'Precipitacao_media',
        }, inplace=True)
        
        logger.info(f"Médias globais calculadas: {len(medias)} combinações DiaAno/Municipio")
        
        return medias
    
    def _obter_medias(self, data: datetime, municipio: int) -> Tuple[float, float, float]:
        """
        Obtém as médias para um dia e município específico.
        
        Args:
            data: Data da predição
            municipio: Código IBGE do município
            
        Returns:
            Tuple: (risco_fogo, dias_sem_chuva, precipitacao)
        """
        dia_ano = data.timetuple().tm_yday
        
        # Buscar média específica
        media_row = self.medias_globais[
            (self.medias_globais['DiaAno'] == dia_ano) &
            (self.medias_globais['Municipio'] == municipio)
        ]
        
        if not media_row.empty:
            row = media_row.iloc[0]
            return row['RiscoFogo_media'], row['DiaSemChuva_media'], row['Precipitacao_media']
        
        # Fallback: média global do município
        media_mun = self.medias_globais[self.medias_globais['Municipio'] == municipio]
        if not media_mun.empty:
            return (
                media_mun['RiscoFogo_media'].mean(),
                media_mun['DiaSemChuva_media'].mean(),
                media_mun['Precipitacao_media'].mean()
            )
        
        # Fallback final: média global de tudo
        return (
            self.medias_globais['RiscoFogo_media'].mean(),
            self.medias_globais['DiaSemChuva_media'].mean(),
            self.medias_globais['Precipitacao_media'].mean()
        )
    
    def _calcular_features(self, data: datetime, municipio: int, 
                          risco_fogo: float, dias_sem_chuva: float, 
                          precipitacao: float) -> Dict:
        """
        Calcula todas as features para a predição.
        
        Args:
            data: Data da predição
            municipio: Código IBGE do município
            risco_fogo: Índice de risco de fogo
            dias_sem_chuva: Dias sem chuva
            precipitacao: Quantidade de precipitação
            
        Returns:
            Dicionário com todas as features
        """
        ano = data.year
        mes = data.month
        dia = data.day
        dia_ano = data.timetuple().tm_yday
        
        # Features cíclicas
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        dia_ano_sin = np.sin(2 * np.pi * dia_ano / 365)
        dia_ano_cos = np.cos(2 * np.pi * dia_ano / 365)
        
        # Features de interação
        risco_x_dias = risco_fogo * dias_sem_chuva
        
        # Features polinomiais
        risco_squared = risco_fogo ** 2
        dias_squared = dias_sem_chuva ** 2
        
        # Features de média móvel (usar médias como proxy)
        risco_media_movel_7 = risco_fogo
        precip_media_movel_7 = precipitacao
        dias_media_movel_14 = dias_sem_chuva
        
        
        # Features de extremos
        risco_max_14 = risco_fogo
        precip_min_7 = precipitacao
        
        # Features de acumulação
        precip_acumulada_7 = precipitacao * 7
        precip_acumulada_30 = precipitacao * 30
        
        # Normalizar latitude/longitude
        lat_norm = (self.df_historico['Latitude'].mean() - self.df_historico['Latitude'].min()) / \
                   (self.df_historico['Latitude'].max() - self.df_historico['Latitude'].min())
        lon_norm = (self.df_historico['Longitude'].mean() - self.df_historico['Longitude'].min()) / \
                   (self.df_historico['Longitude'].max() - self.df_historico['Longitude'].min())
        
        return {
            'Ano': ano,
            'Mes': mes,
            'Dia': dia,
            'DiaAno': dia_ano,
            'Mes_sin': mes_sin,
            'Mes_cos': mes_cos,
            'DiaAno_sin': dia_ano_sin,
            'DiaAno_cos': dia_ano_cos,
            'RiscoFogo': risco_fogo,
            'DiaSemChuva': dias_sem_chuva,
            'Precipitacao': precipitacao,
            'RiscoFogo_squared': risco_squared,
            'DiaSemChuva_squared': dias_squared,
            'RiscoFogo_x_DiaSemChuva': risco_x_dias,
            'RiscoFogo_media_movel_7': risco_media_movel_7,
            'Precipitacao_media_movel_7': precip_media_movel_7,
            'DiaSemChuva_media_movel_14': dias_media_movel_14,
            'RiscoFogo_max_14': risco_max_14,
            'Precipitacao_min_7': precip_min_7,
            'Precipitacao_acumulada_7': precip_acumulada_7,
            'Precipitacao_acumulada_30': precip_acumulada_30,
            'Municipio': municipio,
            'Latitude_norm': lat_norm,
            'Longitude_norm': lon_norm,
        }
    
    def prever(self, data: datetime, municipio: int) -> Dict:
        """
        Faz uma predição de risco de queimada para um município em uma data específica.
        
        Args:
            data: Data da predição (datetime)
            municipio: Código IBGE do município
            
        Returns:
            Dicionário com:
            - 'categoria': 'Baixo', 'Médio' ou 'Alto'
            - 'confianca': Confiança da predição (0-1)
            - 'probabilidades': Dict com probabilidades de cada classe
            - 'data': Data da predição
            - 'municipio_nome': Nome do município
            - 'municipio_codigo': Código do município
        """
        try:
            # Obter médias
            risco_fogo, dias_sem_chuva, precipitacao = self._obter_medias(data, municipio)
            
            # Calcular features
            features_dict = self._calcular_features(data, municipio, risco_fogo, dias_sem_chuva, precipitacao)
            
            # Preparar para o modelo
            lista_features = [
                'Ano', 'Mes', 'Dia', 'DiaAno',
                'Mes_sin', 'Mes_cos', 'DiaAno_sin', 'DiaAno_cos',
                'RiscoFogo', 'DiaSemChuva', 'Precipitacao',
                'RiscoFogo_squared', 'DiaSemChuva_squared',
                'RiscoFogo_x_DiaSemChuva',
                'RiscoFogo_media_movel_7', 'Precipitacao_media_movel_7', 'DiaSemChuva_media_movel_14',
                'RiscoFogo_max_14', 'Precipitacao_min_7',
                'Precipitacao_acumulada_7', 'Precipitacao_acumulada_30',
                'Municipio',
                'Latitude_norm', 'Longitude_norm',
            ]
            
            X = pd.DataFrame([features_dict])[lista_features]
            
            # Fazer predição
            predicao = self.modelo.predict(X)[0]
            probabilidades = self.modelo.predict_proba(X)[0]
            confianca = max(probabilidades)
            
            # Mapear classes
            classes = self.modelo.classes_
            prob_dict = {classes[i]: float(probabilidades[i]) for i in range(len(classes))}
            
            return {
                'categoria': predicao,
                'confianca': float(confianca),
                'probabilidades': prob_dict,
                'data': data.strftime('%d/%m/%Y'),
                'municipio_nome': self.municipios.get(municipio, 'Desconhecido'),
                'municipio_codigo': municipio,
                'risco_fogo': float(risco_fogo),
                'dias_sem_chuva': float(dias_sem_chuva),
                'precipitacao': float(precipitacao),
            }
        
        except Exception as e:
            logger.error(f"Erro ao fazer predição: {e}")
            return {'erro': str(e)}
    
    def prever_multiplos_municipios(self, data: datetime, municipios: list) -> pd.DataFrame:
        """
        Faz predições para múltiplos municípios em uma mesma data.
        
        Args:
            data: Data da predição
            municipios: Lista de códigos IBGE dos municípios
            
        Returns:
            DataFrame com resultados para todos os municípios incluindo coordenadas
        """
        resultados = []
        for municipio in municipios:
            resultado = self.prever(data, municipio)
            if 'erro' not in resultado:
                resultados.append(resultado)
        
        df_resultados = pd.DataFrame(resultados)
        
        # Adicionar coordenadas reais (média dos dados históricos para cada município)
        coordenadas = self.df_historico.groupby('Municipio').agg({
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()
        
        coordenadas.rename(columns={
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        }, inplace=True)
        
        # Mesclar com resultados
        df_resultados = df_resultados.merge(
            coordenadas,
            left_on='municipio_codigo',
            right_on='Municipio',
            how='left'
        )
        
        return df_resultados