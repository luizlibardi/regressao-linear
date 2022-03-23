import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


dados = pd.read_csv('arquivos/dados.csv', sep = ';')

## Variavel dependente x Variaveis Explicativas (pairplot)

# Mexendo com estilo no Seaborn
sns.set_palette("Accent")
sns.set_style('darkgrid')

# Pairplot

ax = sns.pairplot(dados, y_vars = 'consumo', x_vars=['temp_min', 'temp_media', 'temp_max', 'chuva', 'fds'], kind='reg') # Kind = reg mostra a linha de tendencia
ax.figure.set_size_inches(20, 4)
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize = 20, y=1.02)
plt.show()

# Joinplot -> Relacionamento entre duas variaveis e suas dist de freq

ax = sns.jointplot(x= 'temp_max', y='consumo', data=dados, kind='reg')
ax.figure.set_size_inches(20, 4)
ax.fig.suptitle('Dispersão - Consumo X Temperatura', fontsize = 20, y=1.02)
ax.set_axis_labels('Temperatura Máxima', 'Consumo de Cerveja', fontsize=14)
plt.show()

# Lmplot -> Reta de regressão entre duas variaveis e a dispersão entre elas

ax = sns.lmplot(x='temp_max', y='consumo', data=dados)
ax.figure.set_size_inches(20, 4)
ax.fig.suptitle('Reta de Regressão - Consumo X Temperatura', fontsize = 20, y=1.02)
ax.set_xlabels('Temperatura Máxima (°C)', fontsize=14)
ax.set_ylabels('Consumo de Cerveja (Litros)', fontsize=14)
plt.show()

# Plotando um lmplot utilizando uma 3ª variavel na análise

ax = sns.lmplot(x='temp_max', y='consumo', data=dados, hue='fds', markers=['o', '*'], legend=False)
ax.figure.set_size_inches(20, 4)
ax.fig.suptitle('Reta de Regressão - Consumo X Temperatura', fontsize = 20, y=1.02)
ax.set_xlabels('Temperatura Máxima (°C)', fontsize=14)
ax.set_ylabels('Consumo de Cerveja (Litros)', fontsize=14)
ax.add_legend(title='Fim de Semana')
plt.show()

# Separando por coluna

ax = sns.lmplot(x='temp_max', y='consumo', data=dados, col='fds')
ax.figure.set_size_inches(20, 4)
ax.fig.suptitle('Reta de Regressão - Consumo X Temperatura', fontsize = 20, y=1.02)
ax.set_xlabels('Temperatura Máxima (°C)', fontsize=14)
ax.set_ylabels('Consumo de Cerveja (Litros)', fontsize=14)
ax.add_legend(title='Fim de Semana')
plt.show()
