import pandas as pd

#LISTA: UMA COLEÇÃO ORDENADADE ELEMENTOS QUE PODEM SER DE QUALQUER TIPO
lista_nomes = ['Ana', 'Marcos', 'Carlos']
print('Lista de nomes: \n', lista_nomes)
print('Primeiro elemento da lista: \n', lista_nomes[0])

#DICIONÁRIO: ESTRUTURA COMPOSTA DE CHAVES-VALORES
dicionario_pessoas = {
    'nome': 'Ana',
    'idade': 31,
    'Cidade': 'São Paulo'
}
print('Dicionário de uma pessoa: \n', dicionario_pessoas)
print('Atributo do dicionário: \n', dicionario_pessoas.get('nome'))

#LISTA DE DICIONÁRIOS
dados = [
    {'nome': 'Ana', 'idade':31, 'cidade': 'São Paulo'},
    {'nome': 'Marcos', 'idade':25, 'cidade': 'Rio de Janeiro'},
    {'nome': 'Carlos', 'idade':35, 'cidade': 'Brasília'},
]

#DATAFRAME: ESTRUTURA DE DADOS BIDIMENSIONAL
df = pd.DataFrame(dados)
print('DataFrame \n', df)

#SELECIONAR COLUNAS
print(df['nome'])

#SELECIONAR VÁRIAS COLUNAS
print(df[['nome', 'idade']])

#SELECIONAR LINHAS PELO ÍNDICE
print('Primeira linha: \n', df.iloc[0])

#ADICIONAR UMA NOVA COLUNA
df['salario'] = [4100, 3600, 2500]

#ADICIONAR UM NOVO REGISTRO
df.loc[len(df)] = {
    'nome': 'Pedro',
    'idade': 31,
    'cidade': 'Curitiba',
    'salario': 7000
}
print('DataFrame Atual \n', df)

#REMOVENDO UMA COLUNA
df.drop('salario', axis=1, inplace=True)

#FILTRANDO PESSOAS COM MAIS DE 29
filtro_idade = df[df['idade'] >= 30]
print('Filtro \n ', filtro_idade)

#SALVANDO O DATAFRAME EM CSV
df.to_csv('dados.csv', index=False)

#LENDO UM ARQUIVO CSV EM UM DATAFRAME
df_lido = pd.read_csv('dados.csv')
print('\n Leitura do CSV \n ', df_lido)
