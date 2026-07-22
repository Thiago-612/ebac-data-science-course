import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('clientes-v3-preparado.csv')
print('\n', df.head().to_string(), '\n')

#GRÁFICO DE DISPERSÃO
sns.jointplot(x='idade', y='salario', data=df, kind="scatter") #hist hex kde reg resid
plt.show()

#GRÁFICO DE DENSIDADE
plt.figure(figsize=(10,6))
sns.kdeplot(df['salario'], fill=True, color='#863e9c')
plt.title('Densidade de Salários')
plt.xlabel('Salário')
plt.show()

#GRÁFICO DE PAIRPLOT - DISPERSÃO E HISTOGRAMA
sns.pairplot(df[['idade', 'salario', 'nivel_educacao', 'anos_experiencia']])
plt.show()

#GRÁFICO DE REGRESSÃO
sns.regplot(x='idade', y='salario', data=df, color='#278f65', scatter_kws={'alpha': 0.5, 'color': '#34c289'})
plt.title('Regressão de salário por idade')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.show()

#GRÁFICO COUNTPLOT COM HUE
sns.countplot(x='estado_civil', hue='nivel_educacao', data=df, palette='pastel')
plt.legend(title='Nível de Educação')
plt.xlabel('Estado Civil')
plt.ylabel('Quantidade de Clientes')
plt.show()
