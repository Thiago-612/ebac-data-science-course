import pandas as pd

df = pd.read_csv('clientes.csv')

#VERIFICAR OS PRIMEIROS REGISTROS
print(df.head().to_string())

#VERIFICAR A QUANTIDADE DE LINHAS E COLUNAS
print('Quantidade: ', df.shape)

#VERIFICA O TIPO DE DADOS
print('Tipagem: \n ', df.dtypes)

#CHECAR VALORES NULOS
print('Valores nulos: \n', df.isnull().sum())
