import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path = os.getcwd() + '/dbqueimadas_CSV/df_final.csv'
df = pd.read_csv(path)


'''df['DataHora'] = pd.to_datetime(df['DataHora'])
df['Data'] = df['DataHora'].dt.date

df = df.groupby(['Data', 'Municipio']).agg({
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

df = df.dropna()

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

# ===== TARGET TRANSFORMADO =====
# Com log1p:
# FRP_log = [log(3.3), log(6), log(101), log(501), log(1001), log(50001), log(100001)]
#         = [1.19, 1.79, 4.62, 6.22, 6.91, 10.82, 11.51]

# Distribuição: MAIS NORMAL (gaussiana)
# - Valores mais espalhados
# - Menos extremos
# - Modelo entende melhor

# Podemos entender que por se tratar de dados importantes não poderiamos padronizar esses dados.
# porém o sklearn tem uma função chamada ´expm1´ que podemos utilizar quando utilizamos o modelo para predizer um valor,
# com esse valor nós passamos por essa função ela e ela nos retorna o valor de FRP correto.  
df['FRP_log'] = np.log1p(df['FRP'])
'''

# Lista de parametros que iremos utilizar para treinar o modelo.
features_para_usar = [
    'Ano', 'Mes', 'Dia', 'DiaAno',
    'Mes_sin', 'Mes_cos', 'DiaAno_sin', 'DiaAno_cos',
    'RiscoFogo', 'DiaSemChuva', 'Precipitacao','RiscoFogo_squared','DiaSemChuva_squared',
    'RiscoFogo_x_DiaSemChuva',
    'Municipio',
    'Latitude_norm', 'Longitude_norm',
]

# O target é o valor que desejamos predizer com o modelo.
target = 'FRP_log'

X = df[features_para_usar].copy()
y = df[target].copy()


# Validação temporal: 70% treino, 15% validação, 15% teste
n_total = len(df)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)

'''
Exemplo:
├─ 2014-01-01 a 2019-10-31 → Treino (70%)
├─ 2019-11-01 a 2021-04-30 → Validação (15%)
└─ 2021-05-01 a 2024-12-01 → Teste (15%)
'''
X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]

X_val = X.iloc[n_train:n_train + n_val]
y_val = y.iloc[n_train:n_train + n_val]

X_test = X.iloc[n_train + n_val:]
y_test = y.iloc[n_train + n_val:]

print(f"Train: {len(X_train)} amostras ({len(X_train)/n_total*100:.1f}%)")
print(f"Val:   {len(X_val)} amostras ({len(X_val)/n_total*100:.1f}%)")
print(f"Test:  {len(X_test)} amostras ({len(X_test)/n_total*100:.1f}%)")

# ==========================================
# MODELO 1: RANDOM FOREST
# ==========================================

rf_model = RandomForestRegressor(
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

# Aqui passamos os valores de 'Treino', 'Validação' e 'Teste' para o modelo predizer e vermos qual o status de acerto dele.
y_pred_train_rf = rf_model.predict(X_train)
y_pred_val_rf = rf_model.predict(X_val)
y_pred_test_rf = rf_model.predict(X_test)

# ==========================================
# MODELO 2: GRADIENT BOOSTING
# ==========================================

gb_model = GradientBoostingRegressor(
    n_estimators=300,           # Árvores a serem criadas (300 iterações sequenciais)
    learning_rate=0.03,         # Taxa de aprendizado (quanto cada árvore "aprende" com erros)
    max_depth=8,                # Máximo de perguntas por árvore (mais raso que RF)
    min_samples_split=6,        # Minimo de dados para fechar as perguntas de uma árvore
    min_samples_leaf=3,         # Minimo para criar uma folha
    subsample=0.8,              # Usa 80% dos dados para cada árvore (aleatoriedade)
    max_features='sqrt',        # Usa raiz quadrada das features em cada divisão
    validation_fraction=0.1,    # Reserva 10% dos dados para validação interna
)

# Aqui treinamos o modelo com os dados de treinamento. features_para_usar + target
gb_model.fit(X_train, y_train)

# Aqui passamos os valores de 'Treino', 'Validação' e 'Teste' para o modelo predizer e vermos qual o status de acerto dele.
y_pred_train_gb = gb_model.predict(X_train)
y_pred_val_gb = gb_model.predict(X_val)
y_pred_test_gb = gb_model.predict(X_test)

# ==========================================
# AVALIAÇÃO
# ==========================================

def calcular_metricas(y_true, y_pred, nome_dataset):
    """
        MAE (Mean Absolute Error) = Erro Médio Absoluto
        Calcula o ERRO MÉDIO em valores absolutos

        Exemplo:
        Dia 1: Real=100, Predito=95 → Erro=5
        
        RMSE (Root Mean Square Error) = Raiz do Erro Quadrático Médio
        Calcula o erro, MAS PENALIZA ERROS GRANDES

        Exemplo (mesmo dos dias acima):
        Dia 1: Real=100, Predito=95 → Erro²=25

        R² (Coeficiente de Determinação) = Explicação da Variância
        Mostra QUANTO da variação total o modelo explica em porcentagem.
        
    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        nome_dataset (_type_): _description_

    Returns:
        dict: {str: int or str}
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'dataset': nome_dataset,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

# Metricas RandomForest
metricas_rf_train = calcular_metricas(y_train, y_pred_train_rf, "Train")
metricas_rf_val = calcular_metricas(y_val, y_pred_val_rf, "Val")
metricas_rf_test = calcular_metricas(y_test, y_pred_test_rf, "Test")


# Metricas Gradient Boosting
metricas_gb_train = calcular_metricas(y_train, y_pred_train_gb, "Train")
metricas_gb_val = calcular_metricas(y_val, y_pred_val_gb, "Val")
metricas_gb_test = calcular_metricas(y_test, y_pred_test_gb, "Test")

# ==========================================
# VISUALIZAÇÕES
# ==========================================

fig = plt.figure(figsize=(18, 12))

# ===== LINHA 1: COMPARAÇÃO DE R² =====

# R² por dataset
ax1 = plt.subplot(3, 3, 1)
datasets = ['Train', 'Val', 'Test']
rf_r2 = [metricas_rf_train['r2'], metricas_rf_val['r2'], metricas_rf_test['r2']]
gb_r2 = [metricas_gb_train['r2'], metricas_gb_val['r2'], metricas_gb_test['r2']]

x = np.arange(len(datasets))
width = 0.35
ax1.bar(x - width/2, rf_r2, width, label='RandomForest', alpha=0.8)
ax1.bar(x + width/2, gb_r2, width, label='Gradient Boosting', alpha=0.8)
ax1.set_ylabel('R²')
ax1.set_title('Comparação R² - Todos os Datasets')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# MAE por dataset
ax2 = plt.subplot(3, 3, 2)
rf_mae = [metricas_rf_train['mae'], metricas_rf_val['mae'], metricas_rf_test['mae']]
gb_mae = [metricas_gb_train['mae'], metricas_gb_val['mae'], metricas_gb_test['mae']]

ax2.bar(x - width/2, rf_mae, width, label='RandomForest', alpha=0.8)
ax2.bar(x + width/2, gb_mae, width, label='Gradient Boosting', alpha=0.8)
ax2.set_ylabel('MAE')
ax2.set_title('Comparação MAE - Todos os Datasets')
ax2.set_xticks(x)
ax2.set_xticklabels(datasets)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Overfitting (Train vs Test)
ax3 = plt.subplot(3, 3, 3)
overfitting_rf = metricas_rf_train['r2'] - metricas_rf_test['r2']
overfitting_gb = metricas_gb_train['r2'] - metricas_gb_test['r2']

ax3.bar(['RandomForest', 'Gradient Boosting'], [overfitting_rf, overfitting_gb], alpha=0.8, color=['red', 'orange'])
ax3.set_ylabel('Diferença R² (Train - Test)')
ax3.set_title('Análise de Overfitting\n(menor é melhor)')
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)

# ===== LINHA 2: PREDIÇÕES VS REAIS (TEST) =====

# RandomForest
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(y_test, y_pred_test_rf, alpha=0.5, s=10)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax4.set_xlabel('Valor Real')
ax4.set_ylabel('Predição')
ax4.set_title(f'RandomForest - Test\nR² = {metricas_rf_test["r2"]:.3f}')
ax4.grid(True, alpha=0.3)

# Gradient Boosting
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(y_test, y_pred_test_gb, alpha=0.5, s=10, color='orange')
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax5.set_xlabel('Valor Real')
ax5.set_ylabel('Predição')
ax5.set_title(f'Gradient Boosting - Test\nR² = {metricas_gb_test["r2"]:.3f}')
ax5.grid(True, alpha=0.3)

# Diferença de predições
ax6 = plt.subplot(3, 3, 6)
diff_pred = y_pred_test_gb - y_pred_test_rf
ax6.scatter(y_test, diff_pred, alpha=0.5, s=10, color='green')
ax6.axhline(y=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Valor Real')
ax6.set_ylabel('Predição GB - RF')
ax6.set_title('Diferença entre Modelos\n(verde = GB prediz mais)')
ax6.grid(True, alpha=0.3)

# ===== LINHA 3: RESÍDUOS =====

# RandomForest resíduos
ax7 = plt.subplot(3, 3, 7)
residuos_rf = y_test - y_pred_test_rf
ax7.scatter(y_pred_test_rf, residuos_rf, alpha=0.5, s=10)
ax7.axhline(y=0, color='r', linestyle='--', lw=2)
ax7.set_xlabel('Predição')
ax7.set_ylabel('Resíduo')
ax7.set_title('RandomForest - Resíduos')
ax7.grid(True, alpha=0.3)

# Gradient Boosting resíduos
ax8 = plt.subplot(3, 3, 8)
residuos_gb = y_test - y_pred_test_gb
ax8.scatter(y_pred_test_gb, residuos_gb, alpha=0.5, s=10, color='orange')
ax8.axhline(y=0, color='r', linestyle='--', lw=2)
ax8.set_xlabel('Predição')
ax8.set_ylabel('Resíduo')
ax8.set_title('Gradient Boosting - Resíduos')
ax8.grid(True, alpha=0.3)

# Distribuição de resíduos
ax9 = plt.subplot(3, 3, 9)
ax9.hist(residuos_rf, bins=50, alpha=0.6, label='RandomForest', edgecolor='black')
ax9.hist(residuos_gb, bins=50, alpha=0.6, label='Gradient Boosting', edgecolor='black', color='orange')
ax9.set_xlabel('Resíduo')
ax9.set_ylabel('Frequência')
ax9.set_title('Distribuição de Resíduos (Test)')
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparacao_modelos.png', dpi=100, bbox_inches='tight')
# plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

# RandomForest
importances_rf = pd.DataFrame({
    'feature': features_para_usar,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Gradient Boosting
importances_gb = pd.DataFrame({
    'feature': features_para_usar,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)


# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

importances_rf.head(12).plot(x='feature', y='importance', kind='barh', ax=axes[0], legend=False)
axes[0].set_title('Top 12 Features - RandomForest')
axes[0].set_xlabel('Importância')

importances_gb.head(12).plot(x='feature', y='importance', kind='barh', ax=axes[1], legend=False, color='orange')
axes[1].set_title('Top 12 Features - Gradient Boosting')
axes[1].set_xlabel('Importância')

plt.tight_layout()
plt.savefig('feature_importance_comparacao.png', dpi=100, bbox_inches='tight')
# plt.show()

# ==========================================
# RESUMO FINAL
# ==========================================

melhor_modelo = "Gradient Boosting" if metricas_gb_test['r2'] > metricas_rf_test['r2'] else "RandomForest"


print(f"""
RESULTADOS DO TEST SET:

RandomForest:
  • MAE:  {metricas_rf_test['mae']:.3f}
  • RMSE: {metricas_rf_test['rmse']:.3f}
  • R²:   {metricas_rf_test['r2']:.3f}
  • Overfitting (Train R² - Test R²): {overfitting_rf:.3f}

Gradient Boosting:
  • MAE:  {metricas_gb_test['mae']:.3f}
  • RMSE: {metricas_gb_test['rmse']:.3f}
  • R²:   {metricas_gb_test['r2']:.3f}
  • Overfitting (Train R² - Test R²): {overfitting_gb:.3f}

VENCEDOR: {melhor_modelo}
   Por quê: R² mais alto ({max(metricas_gb_test['r2'], metricas_rf_test['r2']):.3f} vs {min(metricas_gb_test['r2'], metricas_rf_test['r2']):.3f})
""")
