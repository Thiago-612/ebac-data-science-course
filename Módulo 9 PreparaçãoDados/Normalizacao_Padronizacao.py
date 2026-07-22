import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print('\n', df.head(), '\n')

df = df[['idade', 'salario']]

#NORMALIZAR SÓ ALTERA A ESCALA
#NORMALIZAÇÃO MIN/MAX SCALER
#NORMALIZA ENTRE 0 E 1

scaler = MinMaxScaler()
df['idadeMinMaxScaler'] = scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler'] = scaler.fit_transform(df[['salario']])

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
df['idadeMinMaxScaler_mm'] = min_max_scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler_mm'] = min_max_scaler.fit_transform(df[['salario']])

#PADRONIZAÇÃO STANDARD SCALER
#MUDA OS PADRÕES DOS DADO PARA MÉDIA 0 E DESVIO PADRÃO 1
#IDEAL GAUSS

scaler = StandardScaler()
df['idadeStandardScaler'] = scaler.fit_transform(df[['idade']])
df['salarioStandardScaler'] = scaler.fit_transform(df[['salario']])

#PADRONIZAÇÃO ROBUST SCALER
#MUDA OS PADRÕES DOS DADO E USA MEDIANA E IQR
#O INTERVALO ENTRE OS DADOS FICA MENOR POR CAUSA DA MEDIANA

scaler = RobustScaler()
df['idadeRobustScaler'] = scaler.fit_transform(df[['idade']])
df['salarioRobustScaler'] = scaler.fit_transform(df[['salario']])

print('\n', df.head(15), '\n')

print('\nMinMaxScaler (De 0 a 1): \n')
print('Idade - Min: {:.4f} Max: {:.4f} Mean(Média): {:.4f} Std(Desvio padrão): {:.4f}'.format(df['idadeMinMaxScaler'].min(), df['idadeMinMaxScaler'].max(), df['idadeMinMaxScaler'].mean(), df['idadeMinMaxScaler'].std()))
print('Salario - Min: {:.4f} Max: {:.4f} Mean(Média): {:.4f} Std(Desvio padrão): {:.4f}'.format(df['salarioMinMaxScaler'].min(), df['salarioMinMaxScaler'].max(), df['salarioMinMaxScaler'].mean(), df['salarioMinMaxScaler'].std()))

print('\nMinMaxScaler (De -1 a 1): \n')
print('Idade - Min: {:.4f} Max: {:.4f} Mean(Média): {:.4f} Std(Desvio padrão): {:.4f}'.format(df['idadeMinMaxScaler_mm'].min(), df['idadeMinMaxScaler_mm'].max(), df['idadeMinMaxScaler_mm'].mean(), df['idadeMinMaxScaler_mm'].std()))
print('Salario - Min: {:.4f} Max: {:.4f} Mean(Média): {:.4f} Std(Desvio padrão): {:.4f}'.format(df['salarioMinMaxScaler_mm'].min(), df['salarioMinMaxScaler_mm'].max(), df['salarioMinMaxScaler_mm'].mean(), df['salarioMinMaxScaler_mm'].std()))

print('\nStandardScaler (Ajusta a média a zero e o desvio padrâo a 1): \n')
print('Idade - Min: {:.4f} Max: {:.4f} Mean(Média): {:.18f} Std(Desvio padrão): {:.4f}'.format(df['idadeStandardScaler'].min(), df['idadeStandardScaler'].max(), df['idadeStandardScaler'].mean(), df['idadeStandardScaler'].std()))
print('Salario - Min: {:.4f} Max: {:.4f} Mean(Média): {:.18f} Std(Desvio padrão): {:.4f}'.format(df['salarioStandardScaler'].min(), df['salarioStandardScaler'].max(), df['salarioStandardScaler'].mean(), df['salarioStandardScaler'].std()))

print('\nRobustScaler (Ajuste a mediana e IQR): \n')
print('Idade - Min: {:.4f} Max: {:.4f} Mean(Média): {:.4f} Std(Desvio padrão): {:.4f}'.format(df['idadeRobustScaler'].min(), df['idadeRobustScaler'].max(), df['idadeRobustScaler'].mean(), df['idadeRobustScaler'].std()))
print('Salario - Min: {:.4f} Max: {:.4f} Mean(Média): {:.4f} Std(Desvio padrão): {:.4f}'.format(df['salarioRobustScaler'].min(), df['salarioRobustScaler'].max(), df['salarioRobustScaler'].mean(), df['salarioRobustScaler'].std()))


