import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.width', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print('\n', df.head(), '\n')

#TRANSFORMAÇÃO LOGARÍTMICA - USADA PARA AJUSTAR OUTLIERS - DISTRIBUIÇÃO ALTAMENTE ASSIMÉTRICA
df['salario_log'] = np.log1p(df['salario'])
print('\n DataFrame após transformação logarítmica no salário: \n', '\n',  df.head(), '\n')

#TRANSFORMAÇÃO BOX-COX - AJUSTA OS DADOS PARA UMA DISTRIBUIÇÃO NORMAL ELIMINANDO OUTLIERS.
df['salario_boxcox'], _ = stats.boxcox(df['salario'] + 1)
print('\n DataFrame após transformação box-cox no salário: \n', '\n',  df.head(), '\n')

#CODIFICAÇÃO DE FREQUÊNCIA PARA ESTADO
estado_freq = df['estado'].value_counts() / len(df)
df['estado_freq'] = df['estado'].map(estado_freq)
print('\n DataFrame após codificação de frequência para estado: \n', '\n',  df.head(), '\n')

#INTERAÇÕES
df['interacao_idade_filho'] = df['idade'] * df['numero_filhos']
print('\n DataFrame após criação de interações entre idade e número de filhos: \n', '\n',  df.head(), '\n')



