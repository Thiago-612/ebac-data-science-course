import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print('\n', df.head(), '\n')

#CODIFICAÇÃO ONEHOT PARA ESTADO CIVIL
df = pd.concat([df, pd.get_dummies(df['estado_civil'], prefix='estado_civil')], axis=1)
print('\n DataFrame após codificação one-hot para estado_civil: \n', '\n', df.head(), '\n')

#CODIFICAÇÃO ORDINAL PARA NIVEL_EDUCACAO
educacao_ordem = {'Ensino Fundamental': 1, 'Ensino Médio': 2, 'Ensino Superior': 3, 'Pós-graduação': 4}
df['nivel_educacao_ordinal'] = df['nivel_educacao'].map(educacao_ordem)
print('\n DataFrame após codificação ordinal para nivel_educacao: \n', '\n', df.head(), '\n')

#TRANSFORMAR AREA_ATUACAO EM CATEGORIAS CODIFICADAS USANDO O METODO .CAT.CODES
df['area_atuacao'] = df['area_atuacao'].astype('category').cat.codes
print('\n DataFrame após transformar area_atuacao em códigos numéricos: \n', '\n', df.head(), '\n')

#LABEL ENCODER PARA ESTADO
#LABEL ENCODER CONVERTE CADA VALOR ÚNICO EM NÚMEROS DE 0 A N_CLASSES -1
Label_Encoder = LabelEncoder()
df['estado_cod'] = Label_Encoder.fit_transform(df['estado'])
print('\n DataFrame após aplicar LabelEncoder em estado: \n', '\n', df.head(), '\n')
