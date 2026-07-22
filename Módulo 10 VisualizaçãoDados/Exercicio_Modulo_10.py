import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('ecommerce_preparados_Tarefa.csv')
print('\n', df.head().to_string(), '\n')

print('\nVerificação inicial: \n')
print(df.info())

print('\nAnálise de dados únicos: \n', df.nunique())

print('\nEstatísticas dos dados: \n', df.describe())

#DATAFRAME POSSUI MUITAS ALTERNATIVAS E PRECISA SER FILTRADO
df_filtrado = df[
    (df['Gênero'].isin(['Feminino', 'Masculino', 'Bebês'])) &
    (df['Temporada'].isin(['primavera/verão', 'outono/inverno', 'não definido']))
].copy()

#CODIFICAÇÃO ORDINAL PARA GÊNERO
genero_ordem = {'Masculino': 1, 'Feminino': 2, 'Bebês': 3}
df_filtrado['Gênero_ordem'] = df_filtrado['Gênero'].map(genero_ordem)

#GRÁFICO DE PIZZA PARA DETERMINAR A PROPORÇÃO DOS GÊNEROS
plt.figure(figsize = (10,10))
df_filtrado['Gênero'].value_counts().plot(
    kind='pie',
    color=['#ff9999','#66b3ff','#99ff99','#ffcc99'],
    autopct='%1.1f%%',
    startangle=90,
)
plt.title('Distribuição dos gêneros')
plt.ylabel('')
plt.show()

#HISTOGRAMA PARA CONCENTRAÇÃO DOS PREÇOS
plt.figure(figsize = (10,6))
plt.hist(df['Preço'], bins = 100, color = 'green', alpha = 0.8)
plt.title('Histograma - Distribuição dos preços')
plt.xlabel('Preço')
plt.xticks(ticks = range(0,int(df['Preço'].max()) + 10, 10))
plt.xticks(rotation = 90)
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

#MAPA DE CALOR PARA DETERMINAR QUEM GASTA MAIS
df_filtrado['genero_masc'] = df_filtrado['Gênero_ordem'].eq(1).astype(int)
df_filtrado['genero_fem'] = df_filtrado['Gênero_ordem'].eq(2).astype(int)
df_filtrado['genero_bebe'] = df_filtrado['Gênero_ordem'].eq(3).astype(int)

plt.figure(figsize=(10,6))

corr_pm = df_filtrado[['Preço', 'genero_masc']].corr()
corr_pf = df_filtrado[['Preço', 'genero_fem']].corr()
corr_pb = df_filtrado[['Preço', 'genero_bebe']].corr()

plt.subplot(2,2,1)
sns.heatmap(
    corr_pm,
    annot=True,              # mostra os valores dentro dos quadrados
    fmt=".2f",               # 2 casas decimais
    cmap="coolwarm",         # paleta mais contrastante
    center=0,                # centraliza a paleta em zero
    linewidths=0.5,          # linhas entre as células
    linecolor="white",       # cor das linhas
    cbar_kws={"shrink": .8, "label": "Correlação"}  # barra lateral ajustada
)
plt.title("Correlação - Gênero Masculino e Preço", fontsize=14, weight='bold')
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.subplot(2,2,2)
sns.heatmap(
    corr_pf,
    annot=True,              # mostra os valores dentro dos quadrados
    fmt=".2f",               # 2 casas decimais
    cmap="coolwarm",         # paleta mais contrastante
    center=0,                # centraliza a paleta em zero
    linewidths=0.5,          # linhas entre as células
    linecolor="white",       # cor das linhas
    cbar_kws={"shrink": .8, "label": "Correlação"}  # barra lateral ajustada
)
plt.title("Correlação - Gênero Feminino e Preço", fontsize=14, weight='bold')
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.subplot(2,2,3)
sns.heatmap(
    corr_pb,
    annot=True,              # mostra os valores dentro dos quadrados
    fmt=".2f",               # 2 casas decimais
    cmap="coolwarm",         # paleta mais contrastante
    center=0,                # centraliza a paleta em zero
    linewidths=0.5,          # linhas entre as células
    linecolor="white",       # cor das linhas
    cbar_kws={"shrink": .8, "label": "Correlação"}  # barra lateral ajustada
)
plt.title("Correlação - Gênero Bebê e Preço", fontsize=14, weight='bold')
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.show()

#GRÁFICO DE DISPERSÃO SCATTER PREÇO E PREÇO
plt.figure(figsize = (10,6))
plt.scatter(df['Preço'], df['Preço'])
plt.title('Dispersão - Preços e preços')
plt.xlabel('Preço')
plt.ylabel('Preço')
plt.show()

#GRÁFICO DE DISPERSÃO SCATTER VENDAS EM RELAÇÃO AO VALOR PARA IDENTIFICAR VALORES OUTLIERS
plt.figure(figsize = (10,6))
plt.scatter(df['Preço'], df['Qtd_Vendidos_Cod'],
            color='#5883a8',
            alpha = 0.6,
            s=30
)
plt.title('Dispersão - Preço e quantidade de vendas')
plt.xlabel('Preço')
plt.ylabel('Quantidade de vendas (log)')
plt.yscale('log')
plt.show()

#GRÁFICO DE DISPERSÃO HEXBIN VENDAS EM RELAÇÃO AO VALOR
plt.figure(figsize = (15,15))
plt.hexbin(df['Preço'], df['Qtd_Vendidos_Cod'],
           gridsize = 50,
           cmap = 'viridis',
           mincnt=1,
           edgecolors='black'
)
plt.colorbar(label='Contagem dentro do bin')
plt.xlabel('Preço')
plt.ylabel('Quantidade de vendidos')
plt.title('Dispersão de Quantidade de vendas em relação ao valor')
plt.xticks(np.arange(0, df['Preço'].max()+1, 10))#PONTOS EIXO
plt.xticks(rotation = 90)
plt.yticks(np.arange(0, df['Qtd_Vendidos_Cod'].max()+1, 1000))
plt.show()

#GRÁFICO DE REGRESSÃO
X = df_filtrado[['Preço']].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(df_filtrado['Qtd_Vendidos_Cod'], errors='coerce')

mask = X['Preço'].notna() & y.notna()# remove NaN nas duas colunas
X = X[mask]
y = y[mask]

modelo = LinearRegression()
modelo.fit(X, y)

print('\nDados da Regressão: \n')
print("Coeficiente:", modelo.coef_[0])
print("Intercepto:", modelo.intercept_)
print("R²:", modelo.score(X, y))

# cria uma faixa de X para desenhar a reta
x_min, x_max = X['Preço'].min(), X['Preço'].max()
x_lin = np.linspace(x_min, x_max, 200).reshape(-1, 1)
y_hat = modelo.predict(x_lin)


plt.figure(figsize=(10,8))
plt.scatter(X['Preço'], y, alpha=0.5, label='Dados')
plt.plot(x_lin.ravel(), y_hat, color='red', linewidth=2, label='Regressão Linear')
plt.title('Regressão Linear - Preço x Quantidade vendida')
plt.xlabel('Preço')
plt.ylabel('Qtd_Vendidos_Cod')
plt.legend()
plt.tight_layout()
plt.show()

#GRÁFICO DE DENSIDADE DOS DESCONTOS
plt.figure(figsize=(10,6))
sns.kdeplot(df['Desconto'], fill=True, color='#863e9c')
plt.title('Densidade dos descontos')
plt.xlabel('Desconto')
plt.show()

#HISTOGRAMA PARA DETERMINAR DESCONTO EM FAIXAS DE PREÇO
plt.figure(figsize = (15,6))
plt.hist(df['Desconto'], bins = 5, color = 'green', alpha = 0.8)
plt.title('Histograma - Distribuição dos valores dos descontos por faixas de preço')
plt.xlabel('Valor do Desconto em Reais')
plt.xticks(ticks = range(0,int(df['Desconto'].max()) + 5, 5))
plt.ylabel('Quantidade de descontos concedidos')
plt.grid(True)
plt.show()

#GRÁFICO DE BARRAS PARA ANALISAR QUEM GANHA MAIS DESCONTO
plt.figure(figsize = (10,10))
df_filtrado['Gênero'].value_counts().plot(kind='bar', color='#90ee70')
plt.title('Quantidade de desconto por gênero')
plt.xlabel('Gênero')
plt.ylabel('Desconto')
plt.xticks(rotation = 90)
plt.show()

#GRÁFICO COUNTPLOT COM HUE PARA IDENTIFICAR QUEM GANHA MAIS DESCONTO E EM QUAL TEMPORADA
sns.countplot(
    x='Temporada',
    hue='Gênero',
    data=df_filtrado,
    palette='pastel',
    hue_order=['Feminino', 'Masculino', 'Bebês'], # garante a ordem
    order=['primavera/verão', 'outono/inverno', 'não definido']
)
plt.legend(title='Quantidade de desconto por gênero em cada temporada')
plt.xlabel('Temporada')
plt.ylabel('Desconto')
plt.xticks(rotation = 90)
plt.show()



