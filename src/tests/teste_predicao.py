import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple
from multiprocessing import Process, Queue, cpu_count
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModeloRandomForestQueimadas:
    """
    Classe para tratamento de dados e treinamento de modelo RandomForest
    para CLASSIFICAÇÃO de risco de queimadas (Baixo, Médio, Alto).
    Usa apenas as features base originais.
    """
    
    def __init__(self, num_processes: int = None):
        """Inicializa a classe."""
        self.csv_files = self._abrir_arquivos_csv()
        self.data_queue = Queue()
        self.processos_finalizados = 0
        self.processos_ativos = 0
        self.processes = []
        self.dados_coletados = []
        self.num_processes = num_processes if num_processes else cpu_count() // 2
        self.modelo = None
        self.df_final = None
        
    def _validar_linha(self, row: pd.Series) -> bool:
        """Valida linhas do CSV."""
        try:
            if pd.isna(row['FRP']):
                return False
            
            campos_obrigatorios = ['Latitude', 'Longitude', 'DataHora', 'Satelite']
            if row[campos_obrigatorios].isnull().any():
                return False
            
            if not row['Pais'] == 'Brasil':
                return False
            
            if pd.notna(row['FRP']) and row['FRP'] < 0:
                return False
            
            try:
                pd.to_datetime(row['DataHora'])
            except:
                return False
            
            if not row['Bioma'] == 'Amazônia':
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Erro na validação: {e}")
            return False
    
    def _processar_arquivo(self, csv_file: str, queue: Queue):
        """Processa arquivo CSV em chunks."""
        try:
            logger.info(f"Processando arquivo: {csv_file}")
            path = os.getcwd() + '/dbqueimadas_CSV'
            
            chunk_size = 1000
            linhas_validas = 0
            linhas_invalidas = 0
            
            for chunk in pd.read_csv(f'{path}/{csv_file}', chunksize=chunk_size, low_memory=False):
                for idx, row in chunk.iterrows():
                    if self._validar_linha(row):
                        row['DataHora'] = pd.to_datetime(row['DataHora'])
                        row = self._codigos_municipio(row=row)
                        queue.put(row.to_dict())
                        linhas_validas += 1
                    else:
                        linhas_invalidas += 1
            
            logger.info(f"Arquivo {csv_file} processado: {linhas_validas} válidas, {linhas_invalidas} inválidas")
        except Exception as e:
            logger.error(f"Erro ao processar {csv_file}: {e}")
    
    def _agregar_por_dia_municipio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados por dia e município."""
        df = df.copy()
        df['DataHora'] = pd.to_datetime(df['DataHora'])
        df['Data'] = df['DataHora'].dt.date
        
        df_daily = df.groupby(['Data', 'Municipio']).agg({
            'FRP': 'sum',
            'RiscoFogo': lambda x: x[x != -999.0].mean(),
            'DiaSemChuva': lambda x: x[x != -999.0].max(),
            'Precipitacao': lambda x: x[x != -999.0].mean(),
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()
        
        df_daily = df_daily.dropna()
        df_daily['Data'] = pd.to_datetime(df_daily['Data'])
        
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
    
    def _engenharia_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features base + features avançadas."""
        df = df.copy()
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values(['Municipio', 'Data']).reset_index(drop=True)
        
        logger.info("Aplicando engenharia de features AVANÇADA...")
        
        # ===== FEATURES TEMPORAIS =====
        df['Ano'] = df['Data'].dt.year
        df['Mes'] = df['Data'].dt.month
        df['Dia'] = df['Data'].dt.day
        df['DiaAno'] = df['Data'].dt.dayofyear
        
        # ===== FEATURES CÍCLICAS =====
        df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
        df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
        df['DiaAno_sin'] = np.sin(2 * np.pi * df['DiaAno'] / 365)
        df['DiaAno_cos'] = np.cos(2 * np.pi * df['DiaAno'] / 365)
        
        # ===== FEATURES DE INTERAÇÃO =====
        df['RiscoFogo_x_DiaSemChuva'] = df['RiscoFogo'] * df['DiaSemChuva']
        
        # ===== FEATURES POLINOMIAIS =====
        df['RiscoFogo_squared'] = df['RiscoFogo'] ** 2
        df['DiaSemChuva_squared'] = df['DiaSemChuva'] ** 2
        
        # ===== FEATURES GEOGRÁFICAS NORMALIZADAS =====
        df['Latitude_norm'] = (df['Latitude'] - df['Latitude'].min()) / (df['Latitude'].max() - df['Latitude'].min())
        df['Longitude_norm'] = (df['Longitude'] - df['Longitude'].min()) / (df['Longitude'].max() - df['Longitude'].min())
        
        # ===== FEATURES DE TENDÊNCIA =====
        df['RiscoFogo_mudanca'] = df.groupby('Municipio')['RiscoFogo'].diff().fillna(0)
        df['Precipitacao_mudanca'] = df.groupby('Municipio')['Precipitacao'].diff().fillna(0)
        df['DiaSemChuva_mudanca'] = df.groupby('Municipio')['DiaSemChuva'].diff().fillna(0)
        
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
        
        # ===== FEATURES DE ANOMALIA =====
        media_risco_30 = df.groupby('Municipio')['RiscoFogo'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        df['RiscoFogo_anomalia'] = df['RiscoFogo'] - media_risco_30
        
        media_precip_30 = df.groupby('Municipio')['Precipitacao'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        df['Precipitacao_anomalia'] = df['Precipitacao'] - media_precip_30
        
        # ===== FEATURES DE LAG =====
        df['RiscoFogo_lag1'] = df.groupby('Municipio')['RiscoFogo'].shift(1).fillna(0)
        df['RiscoFogo_lag7'] = df.groupby('Municipio')['RiscoFogo'].shift(7).fillna(0)
        df['DiaSemChuva_lag1'] = df.groupby('Municipio')['DiaSemChuva'].shift(1).fillna(0)
        
        logger.info(f"✓ Features avançadas criadas com sucesso! Total: {df.shape[1]}")
        
        return df
    
    def _calcular_metricas(self, y_true, y_pred, nome_dataset: str) -> dict:
        """Calcula métricas para classificação."""
        acuracia = accuracy_score(y_true, y_pred)
        precisao = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'dataset': nome_dataset,
            'acuracia': acuracia,
            'precisao': precisao,
            'recall': recall,
            'f1': f1
        }

    def _validacao_testes_com_medias(
        self,
        df_completo: pd.DataFrame,
        modelo,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ):
        """
        Valida o modelo de duas formas:
        1. Com dados REAIS (como antes)
        2. Com MÉDIAS dos últimos 2 meses (como será na produção)
        """
        
        # ============================================================
        # VALIDAÇÃO COM DADOS REAIS (baseline)
        # ============================================================
        y_pred_train = modelo.predict(X_train)
        y_pred_val = modelo.predict(X_val)
        y_pred_test = modelo.predict(X_test)
        
        metricas_train_real = self._calcular_metricas(y_train, y_pred_train, "Train")
        metricas_val_real = self._calcular_metricas(y_val, y_pred_val, "Val")
        metricas_test_real = self._calcular_metricas(y_test, y_pred_test, "Test")
        
        # ============================================================
        # VALIDAÇÃO COM MÉDIAS (como será em produção)
        # ============================================================
        
        # Separar dados em treino, validação e teste
        n_total = len(df_completo)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        df_train = df_completo.iloc[:n_train].copy()
        df_val = df_completo.iloc[n_train:n_train + n_val].copy()
        df_test = df_completo.iloc[n_train + n_val:].copy()
        
        # Calcular médias apenas do conjunto de TREINO
        df_train['DiaAno'] = df_train['Data'].dt.dayofyear
        
        medias_treino = df_train.groupby(['DiaAno', 'Municipio']).agg({
            'RiscoFogo': 'mean',
            'DiaSemChuva': 'mean',
            'Precipitacao': 'mean',
            'Latitude': 'first',
            'Longitude': 'first',
        }).reset_index()
        
        medias_treino.rename(columns={
            'RiscoFogo': 'RiscoFogo_media',
            'DiaSemChuva': 'DiaSemChuva_media',
            'Precipitacao': 'Precipitacao_media',
        }, inplace=True)
        
        # Função auxiliar para calcular todas as features a partir das médias
        def calcular_features_com_medias(df_periodo, medias_df):
            """
            Calcula todas as features usando as médias como input.
            Se não encontrar média para um município específico, usa a média global.
            """
            df_periodo = df_periodo.copy()
            df_periodo['DiaAno'] = df_periodo['Data'].dt.dayofyear
            
            # Calcular média global como fallback
            media_global = medias_df.groupby('DiaAno').agg({
                'RiscoFogo_media': 'mean',
                'DiaSemChuva_media': 'mean',
                'Precipitacao_media': 'mean',
            }).reset_index()
            
            print(f"Média global calculada para {len(media_global)} dias do ano")
            
            features_list = []
            linhas_processadas = 0
            linhas_com_fallback = 0
            linhas_puladas = 0
            
            for idx, row in df_periodo.iterrows():
                try:
                    dia_ano = row['DiaAno']
                    municipio = row['Municipio']
                    
                    # Tentar encontrar média específica do município
                    media_row = medias_df[
                        (medias_df['DiaAno'] == dia_ano) &
                        (medias_df['Municipio'] == municipio)
                    ]
                    
                    if media_row.empty:
                        # Tentar encontrar média global do dia
                        media_row = media_global[media_global['DiaAno'] == dia_ano]
                        
                        if media_row.empty:
                            # Se ainda não encontrou, usar a média global de tudo
                            risco_fogo = medias_df['RiscoFogo_media'].mean()
                            dias_sem_chuva = medias_df['DiaSemChuva_media'].mean()
                            precipitacao = medias_df['Precipitacao_media'].mean()
                        else:
                            media_row = media_row.iloc[0]
                            risco_fogo = media_row['RiscoFogo_media']
                            dias_sem_chuva = media_row['DiaSemChuva_media']
                            precipitacao = media_row['Precipitacao_media']
                        
                        linhas_com_fallback += 1
                    else:
                        media_row = media_row.iloc[0]
                        risco_fogo = media_row['RiscoFogo_media']
                        dias_sem_chuva = media_row['DiaSemChuva_media']
                        precipitacao = media_row['Precipitacao_media']
                    
                    # Calcular features temporais
                    ano = row['Data'].year
                    mes = row['Data'].month
                    dia = row['Data'].day
                    dia_ano_calc = row['Data'].timetuple().tm_yday
                    
                    mes_sin = np.sin(2 * np.pi * mes / 12)
                    mes_cos = np.cos(2 * np.pi * mes / 12)
                    dia_ano_sin = np.sin(2 * np.pi * dia_ano_calc / 365)
                    dia_ano_cos = np.cos(2 * np.pi * dia_ano_calc / 365)
                    
                    # Calcular features de interação e polinomiais
                    risco_x_dias = risco_fogo * dias_sem_chuva
                    risco_squared = risco_fogo ** 2
                    dias_squared = dias_sem_chuva ** 2
                    
                    # Calcular features de tendência
                    risco_mudanca = 0
                    precip_mudanca = 0
                    dias_mudanca = 0
                    
                    # Calcular features de média móvel
                    risco_media_movel_7 = risco_fogo
                    precip_media_movel_7 = precipitacao
                    dias_media_movel_14 = dias_sem_chuva
                    
                    # Calcular features de volatilidade
                    risco_volatilidade_7 = 0
                    precip_volatilidade_7 = 0
                    
                    # Calcular features de extremos
                    risco_max_14 = risco_fogo
                    precip_min_7 = precipitacao
                    
                    # Calcular features de acumulação
                    precip_acumulada_7 = precipitacao * 7
                    precip_acumulada_30 = precipitacao * 30
                    
                    # Calcular features de anomalia
                    risco_anomalia = 0
                    precip_anomalia = 0
                    
                    # Calcular features de lag
                    risco_lag1 = 0
                    risco_lag7 = 0
                    dias_lag1 = 0
                    
                    # Latitude e Longitude normalizadas
                    latitude_norm = (row['Latitude'] - df_completo['Latitude'].min()) / (df_completo['Latitude'].max() - df_completo['Latitude'].min())
                    longitude_norm = (row['Longitude'] - df_completo['Longitude'].min()) / (df_completo['Longitude'].max() - df_completo['Longitude'].min())
                    
                    features = {
                        'Ano': ano,
                        'Mes': mes,
                        'Dia': dia,
                        'DiaAno': dia_ano_calc,
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
                        'RiscoFogo_mudanca': risco_mudanca,
                        'Precipitacao_mudanca': precip_mudanca,
                        'DiaSemChuva_mudanca': dias_mudanca,
                        'RiscoFogo_media_movel_7': risco_media_movel_7,
                        'Precipitacao_media_movel_7': precip_media_movel_7,
                        'DiaSemChuva_media_movel_14': dias_media_movel_14,
                        'RiscoFogo_volatilidade_7': risco_volatilidade_7,
                        'Precipitacao_volatilidade_7': precip_volatilidade_7,
                        'RiscoFogo_max_14': risco_max_14,
                        'Precipitacao_min_7': precip_min_7,
                        'Precipitacao_acumulada_7': precip_acumulada_7,
                        'Precipitacao_acumulada_30': precip_acumulada_30,
                        'RiscoFogo_anomalia': risco_anomalia,
                        'Precipitacao_anomalia': precip_anomalia,
                        'RiscoFogo_lag1': risco_lag1,
                        'RiscoFogo_lag7': risco_lag7,
                        'DiaSemChuva_lag1': dias_lag1,
                        'Municipio': row['Municipio'],
                        'Latitude_norm': latitude_norm,
                        'Longitude_norm': longitude_norm,
                        'Categoria_Real': row['Categoria_Risco'],
                    }
                    
                    features_list.append(features)
                    linhas_processadas += 1
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar linha {idx}: {e}")
                    linhas_puladas += 1
                    continue
            
            print(f"\n✓ Processamento concluído:")
            print(f"  Linhas processadas: {linhas_processadas}")
            print(f"  Linhas com fallback (média global): {linhas_com_fallback}")
            print(f"  Linhas puladas: {linhas_puladas}")
            
            return pd.DataFrame(features_list)
        
        # Calcular features com médias para validação e teste
        df_val_com_medias = calcular_features_com_medias(df_val, medias_treino)
        df_test_com_medias = calcular_features_com_medias(df_test, medias_treino)
        
        # Extrair X e y
        lista_features = self._listar_features_para_treinamento()
        
        X_val_medias = df_val_com_medias[lista_features].copy()
        y_val_medias = df_val_com_medias['Categoria_Real'].copy()
        
        X_test_medias = df_test_com_medias[lista_features].copy()
        y_test_medias = df_test_com_medias['Categoria_Real'].copy()
        
        # Fazer predições com médias
        y_pred_val_medias = modelo.predict(X_val_medias)
        y_pred_test_medias = modelo.predict(X_test_medias)
        
        metricas_val_medias = self._calcular_metricas(y_val_medias, y_pred_val_medias, "Val (Médias)")
        metricas_test_medias = self._calcular_metricas(y_test_medias, y_pred_test_medias, "Test (Médias)")
        
        # ============================================================
        # EXIBIR RESULTADOS COMPARATIVOS
        # ============================================================
        
        print(f'''
    ╔════════════════════════════════════════════════════════════════╗
    ║          VALIDAÇÃO COM DADOS REAIS vs COM MÉDIAS              ║
    ╠════════════════════════════════════════════════════════════════╣
    ║
    ║  TRAIN (dados reais):
    ║    Acurácia: {metricas_train_real['acuracia']:.4f}
    ║    Precisão: {metricas_train_real['precisao']:.4f}
    ║    Recall:   {metricas_train_real['recall']:.4f}
    ║    F1-Score: {metricas_train_real['f1']:.4f}
    ║
    ║  VALIDAÇÃO:
    ║    ├─ Com dados reais:
    ║    │   Acurácia: {metricas_val_real['acuracia']:.4f}
    ║    │   Precisão: {metricas_val_real['precisao']:.4f}
    ║    │   F1-Score: {metricas_val_real['f1']:.4f}
    ║    └─ Com MÉDIAS:
    ║        Acurácia: {metricas_val_medias['acuracia']:.4f}
    ║        Precisão: {metricas_val_medias['precisao']:.4f}
    ║        F1-Score: {metricas_val_medias['f1']:.4f}
    ║        Degradação: {(metricas_val_real['acuracia'] - metricas_val_medias['acuracia']):.4f}
    ║
    ║  TESTE:
    ║    ├─ Com dados reais:
    ║    │   Acurácia: {metricas_test_real['acuracia']:.4f}
    ║    │   Precisão: {metricas_test_real['precisao']:.4f}
    ║    │   F1-Score: {metricas_test_real['f1']:.4f}
    ║    └─ Com MÉDIAS:
    ║        Acurácia: {metricas_test_medias['acuracia']:.4f}
    ║        Precisão: {metricas_test_medias['precisao']:.4f}
    ║        F1-Score: {metricas_test_medias['f1']:.4f}
    ║        Degradação: {(metricas_test_real['acuracia'] - metricas_test_medias['acuracia']):.4f}
    ║
    ╚════════════════════════════════════════════════════════════════╝
        ''')
        
        # ============================================================
        # GRÁFICOS COMPARATIVOS
        # ============================================================
        
        fig = plt.figure(figsize=(16, 10))
        
        # Comparação de Acurácia: Reais vs Médias
        ax1 = plt.subplot(2, 3, 1)
        datasets = ['Val (Real)', 'Val (Médias)', 'Test (Real)', 'Test (Médias)']
        acuracias = [
            metricas_val_real['acuracia'],
            metricas_val_medias['acuracia'],
            metricas_test_real['acuracia'],
            metricas_test_medias['acuracia']
        ]
        cores = ['blue', 'orange', 'green', 'red']
        ax1.bar(datasets, acuracias, color=cores, alpha=0.7)
        ax1.set_ylabel('Acurácia')
        ax1.set_title('Acurácia: Dados Reais vs Médias')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(acuracias):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        ax1.tick_params(axis='x', rotation=45)
        
        # Comparação de F1-Score
        ax2 = plt.subplot(2, 3, 2)
        f1_scores = [
            metricas_val_real['f1'],
            metricas_val_medias['f1'],
            metricas_test_real['f1'],
            metricas_test_medias['f1']
        ]
        ax2.bar(datasets, f1_scores, color=cores, alpha=0.7)
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score: Dados Reais vs Médias')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        ax2.tick_params(axis='x', rotation=45)
        
        # Matriz de confusão - Teste com Médias
        ax3 = plt.subplot(2, 3, 3)
        cm_medias = confusion_matrix(y_test_medias, y_pred_test_medias, labels=['Baixo', 'Médio', 'Alto'])
        sns.heatmap(cm_medias, annot=True, fmt='d', cmap='Blues', ax=ax3,
                    xticklabels=['Baixo', 'Médio', 'Alto'],
                    yticklabels=['Baixo', 'Médio', 'Alto'])
        ax3.set_xlabel('Predito')
        ax3.set_ylabel('Real')
        ax3.set_title('Matriz de Confusão - Teste com Médias')
        
        # Degradação de performance
        ax4 = plt.subplot(2, 3, 4)
        degradacoes = [
            (metricas_val_real['acuracia'] - metricas_val_medias['acuracia']) * 100,
            (metricas_test_real['acuracia'] - metricas_test_medias['acuracia']) * 100
        ]
        datasets_deg = ['Validação', 'Teste']
        cores_deg = ['orange', 'red']
        ax4.bar(datasets_deg, degradacoes, color=cores_deg, alpha=0.7)
        ax4.set_ylabel('Degradação (%)')
        ax4.set_title('Degradação de Acurácia (Reais vs Médias)')
        ax4.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(degradacoes):
            ax4.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
        
        # Precisão e Recall - Teste com Médias
        ax5 = plt.subplot(2, 3, 5)
        report_medias = classification_report(y_test_medias, y_pred_test_medias, output_dict=True, zero_division=0)
        classes = ['Baixo', 'Médio', 'Alto']
        precisoes_m = [report_medias[c]['precision'] for c in classes]
        recalls_m = [report_medias[c]['recall'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        ax5.bar(x - width/2, precisoes_m, width, label='Precisão', alpha=0.8)
        ax5.bar(x + width/2, recalls_m, width, label='Recall', alpha=0.8)
        ax5.set_ylabel('Score')
        ax5.set_title('Precisão e Recall - Teste com Médias')
        ax5.set_xticks(x)
        ax5.set_xticklabels(classes)
        ax5.legend()
        ax5.set_ylim([0, 1])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Distribuição de predições
        ax6 = plt.subplot(2, 3, 6)
        pred_dist = pd.Series(y_pred_test_medias).value_counts().sort_index()
        cores_dist = {'Baixo': 'green', 'Médio': 'orange', 'Alto': 'red'}
        cores_lista = [cores_dist.get(c, 'blue') for c in pred_dist.index]
        ax6.bar(pred_dist.index, pred_dist.values, color=cores_lista, alpha=0.7)
        ax6.set_ylabel('Quantidade')
        ax6.set_title('Distribuição de Predições - Teste com Médias')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('validacao_com_medias.png', dpi=100, bbox_inches='tight')
        print("\n✓ Gráficos salvos em: validacao_com_medias.png")
        plt.show()

    def _validacao_testes(self, modelo, X_train, X_val, X_test, y_train, y_val, y_test):
        """Valida o modelo e gera gráficos."""
        y_pred_train = modelo.predict(X_train)
        y_pred_val = modelo.predict(X_val)
        y_pred_test = modelo.predict(X_test)
        
        metricas_train = self._calcular_metricas(y_train, y_pred_train, "Train")
        metricas_val = self._calcular_metricas(y_val, y_pred_val, "Val")
        metricas_test = self._calcular_metricas(y_test, y_pred_test, "Test")
        
        print(f'''
╔════════════════════════════════════════════╗
║   MÉTRICAS DO MODELO (CLASSIFICAÇÃO)       ║
╠════════════════════════════════════════════╣
║  TRAIN:
║    Acurácia: {metricas_train['acuracia']:.4f}
║    Precisão: {metricas_train['precisao']:.4f}
║    Recall:   {metricas_train['recall']:.4f}
║    F1-Score: {metricas_train['f1']:.4f}
║
║  VALIDAÇÃO:
║    Acurácia: {metricas_val['acuracia']:.4f}
║    Precisão: {metricas_val['precisao']:.4f}
║    Recall:   {metricas_val['recall']:.4f}
║    F1-Score: {metricas_val['f1']:.4f}
║
║  TESTE:
║    Acurácia: {metricas_test['acuracia']:.4f}
║    Precisão: {metricas_test['precisao']:.4f}
║    Recall:   {metricas_test['recall']:.4f}
║    F1-Score: {metricas_test['f1']:.4f}
╚════════════════════════════════════════════╝
        ''')
        
        # Gráficos
        fig = plt.figure(figsize=(15, 10))
        
        # Acurácia por dataset
        ax1 = plt.subplot(2, 3, 1)
        datasets = ['Train', 'Val', 'Test']
        acuracias = [metricas_train['acuracia'], metricas_val['acuracia'], metricas_test['acuracia']]
        ax1.bar(datasets, acuracias, color=['green', 'orange', 'red'], alpha=0.7)
        ax1.set_ylabel('Acurácia')
        ax1.set_title('Acurácia por Dataset')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(acuracias):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # F1-Score por dataset
        ax2 = plt.subplot(2, 3, 2)
        f1_scores = [metricas_train['f1'], metricas_val['f1'], metricas_test['f1']]
        ax2.bar(datasets, f1_scores, color=['green', 'orange', 'red'], alpha=0.7)
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score por Dataset')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Distribuição de classes no teste
        ax3 = plt.subplot(2, 3, 3)
        classes_count = pd.Series(y_test).value_counts().sort_index()
        cores = {'Baixo': 'green', 'Médio': 'orange', 'Alto': 'red'}
        cores_lista = [cores.get(c, 'blue') for c in classes_count.index]
        ax3.bar(classes_count.index, classes_count.values, color=cores_lista, alpha=0.7)
        ax3.set_ylabel('Quantidade')
        ax3.set_title('Distribuição de Classes no Teste')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Matriz de confusão
        ax4 = plt.subplot(2, 3, 4)
        cm = confusion_matrix(y_test, y_pred_test, labels=['Baixo', 'Médio', 'Alto'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
                    xticklabels=['Baixo', 'Médio', 'Alto'],
                    yticklabels=['Baixo', 'Médio', 'Alto'])
        ax4.set_xlabel('Predito')
        ax4.set_ylabel('Real')
        ax4.set_title('Matriz de Confusão (Teste)')
        
        # Precisão e Recall por classe
        ax5 = plt.subplot(2, 3, 5)
        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        classes = ['Baixo', 'Médio', 'Alto']
        precisoes = [report[c]['precision'] for c in classes]
        recalls = [report[c]['recall'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        ax5.bar(x - width/2, precisoes, width, label='Precisão', alpha=0.8)
        ax5.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
        ax5.set_ylabel('Score')
        ax5.set_title('Precisão e Recall por Classe')
        ax5.set_xticks(x)
        ax5.set_xticklabels(classes)
        ax5.legend()
        ax5.set_ylim([0, 1])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Feature importance
        ax6 = plt.subplot(2, 3, 6)
        feature_importance = modelo.feature_importances_
        feature_names = self._listar_features_para_treinamento()
        indices = np.argsort(feature_importance)[-10:]
        ax6.barh(range(len(indices)), feature_importance[indices], alpha=0.7)
        ax6.set_yticks(range(len(indices)))
        ax6.set_yticklabels([feature_names[i] for i in indices])
        ax6.set_xlabel('Importância')
        ax6.set_title('Top 10 Features Mais Importantes')
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('graficos_modelo_classificacao.png', dpi=100, bbox_inches='tight')
        logger.info("✓ Gráficos salvos em: graficos_modelo_classificacao.png")
        plt.show()
    
    def _listar_features_para_treinamento(self) -> list:
        """Retorna lista de features base + avançadas para treinamento."""
        return [
            'Ano', 'Mes', 'Dia', 'DiaAno',
            'Mes_sin', 'Mes_cos', 'DiaAno_sin', 'DiaAno_cos',
            'RiscoFogo', 'DiaSemChuva', 'Precipitacao',
            'RiscoFogo_squared', 'DiaSemChuva_squared',
            'RiscoFogo_x_DiaSemChuva',
            'RiscoFogo_mudanca', 'Precipitacao_mudanca', 'DiaSemChuva_mudanca',
            'RiscoFogo_media_movel_7', 'Precipitacao_media_movel_7', 'DiaSemChuva_media_movel_14',
            'RiscoFogo_volatilidade_7', 'Precipitacao_volatilidade_7',
            'RiscoFogo_max_14', 'Precipitacao_min_7',
            'Precipitacao_acumulada_7', 'Precipitacao_acumulada_30',
            'RiscoFogo_anomalia', 'Precipitacao_anomalia',
            'RiscoFogo_lag1', 'RiscoFogo_lag7', 'DiaSemChuva_lag1',
            'Municipio',
            'Latitude_norm', 'Longitude_norm',
        ]
    
    def _treinar_modelo(self, df: pd.DataFrame) -> RandomForestClassifier:
        """Treina o modelo RandomForest para classificação."""
        logger.info("Iniciando treinamento do modelo de classificação...")
        
        lista_parametros = self._listar_features_para_treinamento()
        X = df[lista_parametros].copy()
        y = df['Categoria_Risco'].copy()
        
        n_total = len(df)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        X_val = X.iloc[n_train:n_train + n_val]
        y_val = y.iloc[n_train:n_train + n_val]
        X_test = X.iloc[n_train + n_val:]
        y_test = y.iloc[n_train + n_val:]
        
        print(f"\n✓ Dados separados:")
        print(f"  Treino: {len(X_train)} amostras ({len(X_train)/n_total*100:.1f}%)")
        print(f"  Validação: {len(X_val)} amostras ({len(X_val)/n_total*100:.1f}%)")
        print(f"  Teste: {len(X_test)} amostras ({len(X_test)/n_total*100:.1f}%)")
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        rf_model.fit(X_train, y_train)
        logger.info("✓ Modelo treinado com sucesso!")
        self._validacao_testes_com_medias(df, rf_model, X_train, X_val, X_test, y_train, y_val, y_test)
        # self._validacao_testes(rf_model, X_train, X_val, X_test, y_train, y_val, y_test)
        
        return rf_model
    
    def _salvar_modelo(self, modelo: RandomForestClassifier, caminho_arquivo: str):
        """Salva o modelo em arquivo."""
        try:
            joblib.dump(modelo, caminho_arquivo)
            logger.info(f"✓ Modelo salvo com sucesso em: {caminho_arquivo}")
        except Exception as e:
            logger.error(f"✗ Erro ao salvar modelo: {e}")
    
    def _ordenar_lista(self, lista: list) -> List[str]:
        """Ordena arquivos CSV por ano."""
        lista_ordenada = []
        
        for valor in lista:
            try:
                valor_int = int(valor[6:10])
            except ValueError:
                logger.error(f"Erro ao carregar o CSV: {valor}")
                continue
            
            if not lista_ordenada:
                lista_ordenada.append(valor_int)
            else:
                for index, valor_lista in enumerate(lista_ordenada):
                    if (index + 1) < len(lista_ordenada):
                        if index == 0 and valor_int < valor_lista:
                            lista_ordenada.insert(0, valor_int)
                            break
                        elif valor_int > valor_lista and valor_int < lista_ordenada[index + 1]:
                            lista_ordenada.insert(index + 1, valor_int)
                            break
                    else:
                        lista_ordenada.append(valor_int)
                        break
        
        return [f'dados_{valor}.csv' for valor in lista_ordenada]
    
    def _abrir_arquivos_csv(self) -> List[str]:
        """Abre e ordena arquivos CSV."""
        path = os.getcwd() + '/dbqueimadas_CSV'
        arquivos_csv = os.listdir(path=path)
        return self._ordenar_lista(lista=arquivos_csv)
    
    def _codigos_municipio(self, row: pd.Series) -> pd.Series:
        """Converte nome do município para código IBGE."""
        municipios = {
            'Alvarães': 1300029, 'Amaturá': 1300060, 'Anamã': 1300086, 'Anori': 1300102,
            'Apuí': 1300144, 'Atalaia Do Norte': 1300201, 'Autazes': 1300300, 'Barcelos': 1300409,
            'Barreirinha': 1300508, 'Benjamin Constant': 1300607, 'Beruri': 1300631,
            'Boa Vista Do Ramos': 1300680, 'Boca Do Acre': 1300706, 'Borba': 1300805,
            'Caapiranga': 1300839, 'Canutama': 1300904, 'Carauari': 1301001, 'Careiro': 1301100,
            'Careiro Da Várzea': 1301159, 'Coari': 1301209, 'Codajás': 1301308, 'Eirunepé': 1301407,
            'Envira': 1301506, 'Fonte Boa': 1301605, 'Guajará': 1301654, 'Humaitá': 1301704,
            'Ipixuna': 1301803, 'Iranduba': 1301852, 'Itacoatiara': 1301902, 'Itamarati': 1301951,
            'Itapiranga': 1302009, 'Japurá': 1302108, 'Juruá': 1302207, 'Jutaí': 1302306,
            'Lábrea': 1302405, 'Manacapuru': 1302504, 'Manaquiri': 1302553, 'Manaus': 1302603,
            'Manicoré': 1302702, 'Maraã': 1302801, 'Maués': 1302900, 'Nhamundá': 1303007,
            'Nova Olinda Do Norte': 1303106, 'Novo Airão': 1303205, 'Novo Aripuanã': 1303304,
            'Parintins': 1303403, 'Pauini': 1303502, 'Presidente Figueiredo': 1303536,
            'Rio Preto Da Eva': 1303569, 'Santa Isabel Do Rio Negro': 1303601,
            'Santo Antônio Do Içá': 1303700, 'São Gabriel Da Cachoeira': 1303809,
            'São Paulo De Olivença': 1303908, 'São Sebastião Do Uatumã': 1303957, 'Silves': 1304005,
            'Tabatinga': 1304062, 'Tapauá': 1304104, 'Tefé': 1304203, 'Tonantins': 1304237,
            'Uarini': 1304260, 'Urucará': 1304302, 'Urucurituba': 1304401,
        }
        row['Municipio'] = municipios[row['Municipio'].title()]
        return row
    
    def processar_e_treinar(self) -> Tuple[pd.DataFrame, RandomForestClassifier]:
        """Executa todo o pipeline: tratamento, features e treinamento."""
        logger.info("=" * 60)
        logger.info("INICIANDO PIPELINE DE CLASSIFICAÇÃO (FEATURES BASE)")
        logger.info("=" * 60)
        
        # Processar arquivos
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
            return None, None
        
        # Agregar
        df = self._agregar_por_dia_municipio(df=df)
        
        # Criar categorias
        df = self._criar_categorias_risco(df=df)
        
        # Engenharia de features
        df = self._engenharia_features(df=df)
        
        self.df_final = df 
        
        # Treinar modelo
        self.modelo = self._treinar_modelo(df)
        
        # Salvar
        path = os.getcwd()
        self._salvar_modelo(self.modelo, path + '/modelo_RF_classificacao.jkl')
        df.to_csv(path + '/dbqueimadas_CSV/df_final_classificacao.csv', index=False)
        logger.info(f"✓ Dados salvos em: {path}/dbqueimadas_CSV/df_final_classificacao.csv")
        
        logger.info("=" * 60)
        logger.info("PIPELINE CONCLUÍDO COM SUCESSO")
        logger.info("=" * 60)
        
        return df, self.modelo


# ===== EXEMPLO DE USO =====

if __name__ == "__main__":
    pipeline = ModeloRandomForestQueimadas()
    df_final, modelo = pipeline.processar_e_treinar()