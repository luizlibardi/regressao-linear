import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# Importando o train_test_split da biblioteca scikit-learn
from sklearn.model_selection import train_test_split

dados = pd.read_csv('arquivos/dados.csv', sep = ';')

# Criando uma Series (pandas) para armazenar o Consumo de Cerveja (y)
y = dados['consumo']

# Criando uma Series (pandas) para armazenar as variáveis explicativas (X)
X = dados[['temp_max', 'chuva', 'fds']]

# Modelo supervisionado, por ter treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2811)

# Função de Regressão com 3 variáveis explicativas

# Importando LinearRegression e metrics da biblioteca scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Instanciando a classe Linear Regression
modelo = LinearRegression()

# Utilizando o método fit() do objeto 'modelo' para estimar nosso modelo linear utilizando os dados de TREINO (X_train e y_train)
modelo.fit(X_train, y_train)

# Obtendo o coeficiente de determinação (R²) do modelo estimado com os dados de TREINO

# print(f'R² = {modelo.score(X_train, y_train).round(2)}')

# Gerando previsões para os dados de TESTE (X_test) utilizando método predict() do objeto 'modelo'
y_previsto = modelo.predict(X_test)

# Obtendo o coeficiente de determinação (R²) para as previsões do nosso modelo

# print(f'R² = {metrics.r2_score(y_test, y_previsto).round(2)}')

# Obtendo Previsões Pontuais

entrada = X_test[0:1]

# Gerando previsão pontual
modelo.predict(entrada)[0]

# Criando um simulador simples

temp_max = 40
chuva = 0
fds = 1
entrada = [[temp_max, chuva, fds]]

# print(f'{modelo.predict(entrada)[0]} litros')

## Obtendo o intercepto do modelo
modelo.intercept_

# Obtendo coeficientes de regressão
modelo.coef_

X.columns 
index= ['Intercepto', 'Temperatura Máxima', 'Chuva (mm)', 'Final de Semana']


pd.DataFrame(data = np.append(modelo.intercept_, modelo.coef_), index=index, columns=['Parâmetros'])

## Analises Graficas das Previsoes

# Gerando as previoes do modelo para os dados de TREINO
y_previsto_train = modelo.predict(X_train)

# Grafico de dispersão entre o valor estimado e o valor real
ax = sns.scatterplot(x = y_previsto_train, y = y_train)
ax.figure.set_size_inches(20, 6)
ax.set_title('Previsão X Real', fontsize = 18)
ax.set_xlabel('Consumo de Cerveja (litros) - Previsão', fontsize = 14)
ax.set_ylabel('Consumo de Cerveja (litros) - Real', fontsize = 14)
# plt.show()

# Obtendo residuo
residuo = y_train - y_previsto_train
# print(residuo)

# Grafico de dispersão entre valor estimado e residuos
ax = sns.scatterplot(x = y_previsto_train, y = residuo)
ax.figure.set_size_inches(20, 8)
ax.set_title('Residuos X Real', fontsize = 18)
ax.set_xlabel('Consumo de Cerveja (litros) - Previsão', fontsize = 14)
ax.set_ylabel('Residuos', fontsize = 14)
# plt.show()

# Plotando a distribuição de frequencia dos residuos
# ax = sns.distplot(residuo)
ax.set_title('Distribuição de frequencia dos Residuos', fontsize = 18)
ax.set_ylabel('Litros', fontsize = 14)
# plt.show()

## Comparando modelos

# Estimando um novo modelo

X2 = dados [['temp_media', 'chuva', 'fds']]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size = 0.3, random_state=2811)

modelo_2 = LinearRegression()
modelo_2.fit(X2_train, y2_train)
# print('Modelo com Temp. Média')
# print(f'R² = {modelo_2.score(X2_train, y2_train).round(2)}')
# print('Modelo com Temp. Máxima')
# print(f'R² = {modelo.score(X_train, y_train).round(2)}')

y_previsto = modelo.predict(X_test)
y_previsto_2 = modelo_2.predict(X2_test)


# print('Modelo com Temp. Média')
# print(f'R² = {metrics.r2_score(y2_test, y_previsto_2).round(2)}')

# print('Modelo com Temp. Máxima')
# print(f'R² = {metrics.r2_score(y_test, y_previsto).round(2)}')

# Obtendo métricas para modelo com Temperatura Média

EQM_2 = metrics.mean_squared_error(y2_test, y_previsto_2).round(2)
REQM_2 = np.sqrt(metrics.mean_squared_error(y2_test, y_previsto_2)).round(2)
R2_2 = metrics.r2_score(y2_test, y_previsto_2).round(2)

data = pd.DataFrame([EQM_2, REQM_2, R2_2], ['EQM', 'REQM', 'R²'], columns = ['Métricas'])
# print(data)

# Obtendo métricas para modelo com Temperatura Máxima

EQM_2 = metrics.mean_squared_error(y_test, y_previsto).round(2)
REQM_2 = np.sqrt(metrics.mean_squared_error(y_test, y_previsto)).round(2)
R2_2 = metrics.r2_score(y_test, y_previsto).round(2)

data = pd.DataFrame([EQM_2, REQM_2, R2_2], ['EQM', 'REQM', 'R²'], columns = ['Métricas'])
# print(data)
 

# Salvando o modelo estimado
import pickle

output = open('arquivos/modelo_consumo_cerveja', 'wb')
pickle.dump(modelo, output)
output.close()