import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


dados = pd.read_csv('arquivos/dados.csv', sep = ';')

## Analises Preliminares

# Estatisticas descritivas

estatistica_descritiva = dados.describe().round(2)
# print(estatistica_descritiva)

# Matriz de correlacao -> Medida de associação linear entre duas variáveis e situa-se entre -1 e +1 (associação negativa perfeita e associação positiva perfeita)

matriz_correlacao = dados.corr().round(4)
# print(matriz_correlacao)

## Análises Gráficas

# Plotando a variável dependente (y)

fig, ax = plt.subplots(1, 1, figsize=(30, 6))
ax.set_title('Consumo de Cerveja', fontsize = 20)
ax.set_ylabel('Litros', fontsize = 16)
ax.set_xlabel('Dias', fontsize = 16)
ax = dados['consumo'].plot(fontsize = 14)
# plt.show()

## Box Plot

# Outliers
ax = sns.boxplot(data=dados['consumo'], orient='v', width=0.2)
ax.figure.set_size_inches(12, 6)
# plt.show()

## Box Plot com 2 variáveis

# Mexendo com estilo no Seaborn
sns.set_palette("Accent")
sns.set_style('darkgrid')

ax = sns.boxplot(y = 'consumo', x = 'fds', data=dados, orient='h', width=0.5)
ax.figure.set_size_inches(12, 6)
ax.set_xlabel('Final de Semana', fontsize = 16)
# plt.show()

## Distribuições de Frequencia

ax = sns.distplot(dados['consumo'])
ax.set_title('Distribuição de frequência', fontsize = 20)
ax.set_ylabel('Consumo de Cerveja (Litros)', fontsize = 16)
# plt.show()
