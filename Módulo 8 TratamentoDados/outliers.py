import pandas as pd
from scipy import stats

pd.set_option('display.width', None)

df = pd.read_csv('clientes_limpeza.csv')

df_filtro_basico = df[df['idade'] > 100]
print('Filtro básico \n' , df_filtro_basico[['nome', 'idade']])

#IDENTIFICAR OUTLIERS COM Z-SCORE
z_score = stats.zscore(df['idade'].dropna())
outliers_z = df[z_score > 3]
print('Outliers pelo z-score: \n', outliers_z)

#FILTRAR OUTLIERS COM Z-SCORE
df_zscore = df[(stats.zscore(df['idade']) < 3)]

#IDENTIFICAR OUTLIERS COM IQR
Q1 = df['idade'].quantile(0.25)
Q3 = df['idade'].quantile(0.75)
IQR = Q3 - Q1

limite_baixo = Q1 - 1.5*IQR
limite_alto = Q3 + 1.5*IQR

print('Limites IQR: \n', limite_baixo,limite_alto)

#MOSTRAR OUTLIERS
outliers_iqr = df[(df['idade'] < limite_baixo) | (df['idade'] > limite_alto)]
print('Outliers IQR: \n', outliers_iqr)

#FILTRAR VALORES QUE NÃO SÃO OUTLIERS COM IQR
df_iqr = df[(df['idade'] >= limite_baixo) & (df['idade'] <= limite_alto)]

#FILTRAR ENDEREÇOS INVÁLIDOS
df['endereco'] = df['endereco'].apply(lambda x: 'Endereço inválido' if len(x.split('\n')) < 3 else x)
print('Quantidade de endereços inválidos: ', (df['endereco'] == 'Endereço inválido').sum())

#TRATAR CAMPOS DE TEXTO
df['nome'] = df['nome'].apply(lambda x: 'Nome inválido' if isinstance(x, str) and len(x) > 50 else x)
print('Quantidade de nomes inválidos: ', (df['nome'] == 'Nome inválido').sum())

print('Dados com outliers tratados: \n', df)

df.to_csv('clientes_remove_outliers.csv', index=False)







