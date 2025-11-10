import os
import joblib
import numpy as np
from matplotlib import pyplot as plt

from src.tratando_dados import Tratando_Dados

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

class Modelo_RF(Tratando_Dados):
    def __init__(self, num_processes = None):
        super().__init__(num_processes)

        self.lista_parametros = [
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

        df_final = self.processar_todos()

        path = os.getcwd() 

        modelo = self._modelo(df=df_final)

        self._salvar_modelo(
            modelo=modelo,
            caminho_arquivo=path+'/modelo_RF.jkl',
        )

        df_final.to_csv(path+'/dbqueimadas_CSV/df_final.csv', index=False)

    def _calcular_metricas(self, y_true, y_pred, nome_dataset):
        """
        Calcula métricas de desempenho do modelo de classificação.
        
        Métricas:
        
        ACURÁCIA:
            Percentual de predições corretas sobre o total de predições.
            Responde: "De todas as predições, quantas acertei?"
            Fórmula: (Acertos) / (Total de predições)
            Exemplo: Se acertei 80 de 100, acurácia = 80%
            Uso: Bom para datasets balanceados
        
        PRECISÃO:
            De todas as predições positivas que fiz, quantas estavam corretas?
            Responde: "Quando meu modelo diz 'Alto risco', ele acerta?"
            Fórmula: (Verdadeiros Positivos) / (Verdadeiros Positivos + Falsos Positivos)
            Exemplo: Se predisse "Alto" 100 vezes e acertei 80, precisão = 80%
            Uso: Importante quando falso positivo é custoso
        
        RECALL (Sensibilidade):
            De todos os casos reais positivos, quantos meu modelo encontrou?
            Responde: "Quantos 'Alto risco' reais meu modelo conseguiu identificar?"
            Fórmula: (Verdadeiros Positivos) / (Verdadeiros Positivos + Falsos Negativos)
            Exemplo: Se havia 100 casos "Alto" reais e encontrei 80, recall = 80%
            Uso: Importante quando falso negativo é custoso (perder um caso real)
        
        F1-SCORE:
            Equilíbrio entre Precisão e Recall. Média harmônica dos dois.
            Responde: "Qual é o balanço entre precisão e recall?"
            Fórmula: 2 * (Precisão * Recall) / (Precisão + Recall)
            Exemplo: Se precisão=80% e recall=80%, F1=80%
            Uso: Melhor métrica quando você quer equilibrio entre precisão e recall

        Args:
            y_true (_type_): _description_
            y_pred (_type_): _description_
            nome_dataset (_type_): _description_

        Returns:
            dict: {str: float}
        """
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
    

    def _modelo(self, df: pd.DataFrame):
        # Separação dos dados
        X = df[self.lista_parametros].copy()
        y = df['Categoria_Risco'].copy()


        # Validação temporal: 80% treino, 20% teste
        n_total = len(df)
        n_train = int(n_total * 0.80)

        '''
        Exemplo:
        ├─ 2014-01-01 a 2021-12-31 → Treino (80%)
        ├─ 2022-01-01 a 2024-12-31 → Validação (20%)
        '''
        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]

        X_test = X.iloc[n_train:]
        y_test = y.iloc[n_train:]


        print(f"Train: {len(X_train)} amostras ({len(X_train)/n_total*100:.1f}%)")
        print(f"Test:  {len(X_test)} amostras ({len(X_test)/n_total*100:.1f}%)")

        # MODELO 1: RANDOM FOREST
        rf_model = RandomForestClassifier(
            n_estimators=200,       # Árvores a serem criadas
            max_depth=20,           # Máximo de pergunta por árvore
            min_samples_split=5,    # Minimo de dados para fechar as perguntas de uma árvore
            min_samples_leaf=2,     # Minimo para criar uma folha
            random_state=42,        # Defini o mesmo resultado para uma mesma pergunta
            n_jobs=-1,              # Processadores para criar o modelo (-1 usa todos os processadores livres)
            verbose=0               # Não mostra log de criação do modelo.
        )

        # Aqui treinamos o modelo com os dados de treinamento. features_para_usar + target
        rf_model.fit(X_train, y_train)

        self._validacao_testes(
            modelo=rf_model,
            df_completo=df,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        return rf_model

    def _validacao_testes(
        self,
        df_completo: pd.DataFrame,
        modelo: RandomForestClassifier,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
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
        y_pred_test = modelo.predict(X_test)
        
        metricas_train_real = self._calcular_metricas(y_train, y_pred_train, "Train")
        metricas_test_real = self._calcular_metricas(y_test, y_pred_test, "Test")
        
        # ============================================================
        # VALIDAÇÃO COM MÉDIAS (como será em produção)
        # ============================================================
        
        # Separar dados em treino, validação e teste
        n_total = len(df_completo)
        n_train = int(n_total * 0.80)
        
        df_train = df_completo.iloc[:n_train].copy()
        df_test = df_completo.iloc[n_train:].copy()
        
        # Calcular médias apenas do conjunto de TREINO        
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
                    
                    # Calcular features de média móvel
                    risco_media_movel_7 = risco_fogo
                    precip_media_movel_7 = precipitacao
                    dias_media_movel_14 = dias_sem_chuva
                    
                    # Calcular features de extremos
                    risco_max_14 = risco_fogo
                    precip_min_7 = precipitacao
                    
                    # Calcular features de acumulação
                    precip_acumulada_7 = precipitacao * 7
                    precip_acumulada_30 = precipitacao * 30
                    
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
                        'RiscoFogo_media_movel_7': risco_media_movel_7,
                        'Precipitacao_media_movel_7': precip_media_movel_7,
                        'DiaSemChuva_media_movel_14': dias_media_movel_14,
                        'RiscoFogo_max_14': risco_max_14,
                        'Precipitacao_min_7': precip_min_7,
                        'Precipitacao_acumulada_7': precip_acumulada_7,
                        'Precipitacao_acumulada_30': precip_acumulada_30,
                        'Municipio': row['Municipio'],
                        'Latitude_norm': latitude_norm,
                        'Longitude_norm': longitude_norm,
                        'Categoria_Real': row['Categoria_Risco'],
                    }
                    
                    features_list.append(features)
                    linhas_processadas += 1
                    
                except Exception as e:
                    linhas_puladas += 1
                    continue
            
            print(f"\n✓ Processamento concluído:")
            print(f"  Linhas processadas: {linhas_processadas}")
            print(f"  Linhas com fallback (média global): {linhas_com_fallback}")
            print(f"  Linhas puladas: {linhas_puladas}")
            
            return pd.DataFrame(features_list)
        
        # Calcular features com médias para validação e teste
        df_test_com_medias = calcular_features_com_medias(df_test, medias_treino)
                
        X_test_medias = df_test_com_medias[self.lista_parametros].copy()
        y_test_medias = df_test_com_medias['Categoria_Real'].copy()
        
        # Fazer predições com médias
        y_pred_test_medias = modelo.predict(X_test_medias)
        
        metricas_test_medias = self._calcular_metricas(y_test_medias, y_pred_test_medias, "Test (Médias)")
        
        # ============================================================
        # EXIBIR RESULTADOS COMPARATIVOS
        # ============================================================
        
        print(f'''
        TRAIN (dados reais):
            Acurácia: {metricas_train_real['acuracia']:.4f}
            Precisão: {metricas_train_real['precisao']:.4f}
            Recall:   {metricas_train_real['recall']:.4f}
            F1-Score: {metricas_train_real['f1']:.4f}
        TESTE:
        ├─ Com dados reais:
        │   Acurácia: {metricas_test_real['acuracia']:.4f}
        │   Precisão: {metricas_test_real['precisao']:.4f}
        │   F1-Score: {metricas_test_real['f1']:.4f}
        └─ Com MÉDIAS:
            Acurácia: {metricas_test_medias['acuracia']:.4f}
            Precisão: {metricas_test_medias['precisao']:.4f}
            F1-Score: {metricas_test_medias['f1']:.4f}
            Degradação: {(metricas_test_real['acuracia'] - metricas_test_medias['acuracia']):.4f}
        ''')
        
        # ============================================================
        # GRÁFICOS COMPARATIVOS
        # ============================================================
                
        # Comparação de Acurácia: Reais vs Médias
        ax1 = plt.subplot(2, 3, 1)
        datasets = ['Test (Real)', 'Test (Médias)']
        acuracias = [
            metricas_test_real['acuracia'],
            metricas_test_medias['acuracia']
        ]
        cores = ['blue', 'red']
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


    def _salvar_modelo(self, modelo: RandomForestClassifier, caminho_arquivo: str):
        """Salva o modelo em arquivo."""
        try:
            joblib.dump(modelo, caminho_arquivo)
        except Exception as e:
            print(e)