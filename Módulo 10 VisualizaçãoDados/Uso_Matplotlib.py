import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('clientes-v3-preparado.csv')
print('\n', df.head().to_string(), '\n')

#GRÁFICO DE BARRAS
plt.figure(figsize = (10,6))
df['nivel_educacao'].value_counts().plot(kind='bar', color='#90ee70')
plt.title('Divisão de escolaridade - 1')
plt.xlabel('Nível de Educação')
plt.ylabel('Quantidade')
plt.xticks(rotation = 0)
plt.show()

x = df['nivel_educacao'].value_counts().index
y = df['nivel_educacao'].value_counts().values

plt.figure(figsize = (10,6))
plt.bar(x, y, color='#60aa65')
plt.title('Divisão de escolaridade - 2')
plt.xlabel('Nível de Educação')
plt.ylabel('Quantidade')

#GRÁFICO DE PIZZA
plt.figure(figsize = (10,6))
plt.pie(y, labels=x, autopct='%.1f%%', startangle=90)
plt.title('Distribuição de Nível de Educação')
plt.show()

#GRÁFICO DE DISPERSÃO
plt.hexbin(df['idade'], df['salario'], gridsize = 40, cmap = 'Blues')
plt.colorbar(label='Contagem dentro do bin')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.title('Dispersão de Idade e Salário')
plt.show()

