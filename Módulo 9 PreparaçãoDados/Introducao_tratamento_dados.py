#Análise exploratória de dados
import pandas as pd

df = pd.read_csv('clientes-v2.csv')

print(df.head().to_string())
print(df.tail().to_string())
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')

print('\nVerificação inicial: \n')
print(df.info())

print('\nAnálise de dados nulos: \n', df.isnull().sum())
print('\nPorcentagem de dados nulos: \n', df.isnull().mean() * 100)
df.dropna(inplace=True)
print('\nConfirmar remoção de dados nulos: \n', df.isnull().sum().sum())

print('\nAnálise de dados duplicados: \n', df.duplicated().sum())

print('\nAnálise de dados únicos: \n', df.nunique())

print('\nEstatísticas dos dados: \n', df.describe())

df = df[['idade', 'data', 'estado', 'salario', 'nivel_educacao', 'numero_filhos', 'estado_civil', 'area_atuacao']]
print(df.head().to_string())

df.to_csv('clientes-v2-tratados.csv', index=False)
