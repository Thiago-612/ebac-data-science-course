import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('clientes-v3-preparado.csv')
print('\n', df.head().to_string(), '\n')

#HISTOGRAMA
plt.hist(df['salario'])
plt.show()

#HISTOGRAMA PARÂMETROS
plt.figure(figsize = (10,6))
plt.hist(df['salario'], bins = 100, color = 'green', alpha = 0.8)
plt.title('Histograma - Distribuição de salários')
plt.xlabel('Salario')
plt.xticks(ticks = range(0,int(df['salario'].max()) + 2000, 2000))
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

#MÚLTIPLOS GRÁFICOS
plt.figure(figsize = (10,6))
plt.subplot(2,2,1) # 2 LINHAS 2 COLUNAS 1° GRÁFICO

#GRÁFICO DE DISPERSÃO
plt.scatter(df['salario'], df['salario'])
plt.title('Dispersão - Salários e salários')
plt.xlabel('Salario')
plt.ylabel('Salario')

plt.subplot(1,2,2)
plt.scatter(df['salario'], df['anos_experiencia'], color='#5883a8', alpha = 0.6, s=30)
plt.title('Dispersão - Idade e anos de experiência')
plt.xlabel('Salário')
plt.ylabel('Anos de experiência')

#MAPA DE CALOR
corr = df[['salario', 'anos_experiencia']].corr()
plt.subplot(2,2,3)
sns.heatmap(corr, annot = True, cmap = 'coolwarm')
plt.title('Correlação - Salário e Idade')

plt.tight_layout() #AJUSTAR ESPAÇAMENTOS
plt.show()


