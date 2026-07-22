import pandas as pd

df = pd.read_csv('clientes.csv')

pd.set_option('display.width', None)
print(df.head())

#REMOVER DADOS
df.drop(labels='pais', axis=1, inplace=True) #COLUNA
df.drop(labels=2, axis=0, inplace=True) #LINHA

#NORMALIZAR CAMPOS DE TEXTO
df['nome'] = df['nome'].str.title()
df['endereco'] = df['endereco'].str.lower()
df['estado'] = df['estado'].str.strip().str.upper()

#CONVERTER TIPOS DE DADOS
df['idade'] = df['idade'].astype(int)

print(df.head())

#TRATAR VALORES NULOS AUSENTES
df_fillna = df.fillna(0) #SUBSTITUIR VALORES NULOS POR ZERO
df_dropna = df.dropna() #REMOVER REGISTROS COM VALORES NULOS
df_dropna4 = df.dropna(thresh=4) #MANTER REGISTROS COM NO MÍNIMO 4 VALORES NULOS
df = df.dropna(subset=['cpf']) #REMOVER REGISTRO COM CPF NULO

print('Valores nulos:\n', df.isnull().sum())
print('Quantidade de valores nulos com fillna: ', df_fillna.isnull().sum().sum())
print('Quantidade de valores nulos com dropna: ', df_dropna.isnull().sum().sum())
print('Quantidade de valores nulos com dropna4: ', df_dropna4.isnull().sum().sum())
print('Quantidade de valores nulos com CPF: ', df.isnull().sum().sum())

#TROCAR VALORES NULOS POR TEXTOS
#INPLACE=TRUE SERVE PARA SALVAR AS ALTERAÇÕES NO DF
df.fillna({'estado': 'desconhecido'}, inplace=True)#PODE USAR NOS PRÓXIMOS
df['endereco'] = df['endereco'].fillna('Endereço não informado')
df['idade_corrigida'] = df['idade'].fillna(df['idade'].mean())

#TRATAR FORMATO DE DADOS
df['data_corrigida'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')

#TRATAR DADOS DUPLICADOS
print('Quantidade de registros atuais: ', df.shape[0])
df.drop_duplicates()
df.drop_duplicates(subset=['cpf'], inplace=True)
print('Quantidade de registros removendo as duplicadas: ', len(df))

print('Dados limpos:\n ',df)

#SALVAR DATAFRAME
df['data'] = df['data_corrigida']
df['idade'] = df['idade_corrigida']

df_salvar = df[['nome', 'cpf', 'idade', 'data', 'endereco', 'estado']]
df_salvar.to_csv('clientes_limpeza.csv', index=False)

print('Novo DataFrame: \n', pd.read_csv('clientes_limpeza.csv'))
