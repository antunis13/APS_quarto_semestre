import os
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path = os.getcwd() + '/dbqueimadas_CSV/df_final.csv'

df = pd.read_csv(path)

# Essa linha de código vai colocar as datas na ordem correta, isso tem que ser feito manualmente. 
# Ficará dessa forma só para testes.
df = df.sort_values('data').copy()

X = df.drop(columns=['focos_count', 'data'], axis=1)
y = df['focos_count']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

rf = RandomForestRegressor(
    n_estimators=400,           
    max_depth=None,             
    min_samples_split=2,        
    min_samples_leaf=1,         
    n_jobs=-1,                  
    random_state=1   
)

# rf = RandomForestRegressor(random_state=1)

rf.fit(X_train, y_train)

# Fazendo previsões
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_pred, y_test)
rmse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)

print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

joblib.dump(rf, 'modelo_frp.pkl')

# carregar depois
modelo = joblib.load('modelo_frp.pkl')
