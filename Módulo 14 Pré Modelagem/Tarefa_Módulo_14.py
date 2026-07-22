import pandas as pd
from scipy.stats import pointbiserialr
import seaborn as sns
from translatepy import Translator
import time
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('CHURN_TELECON.csv',
    sep=';',
    encoding='utf-8',
    na_values=['', ' ', 'NA', 'None']
)

print('\n', df.head().to_string(), '\n')

print('\nVERIFICAÇÃO DO DATAFRAME ANTES DA TRADUÇÃO: \n')
print(df.info())

PAUSA = 0.03  # pausa leve entre traduções
COLUNA_ID = "customerID"  # não traduz valores desta coluna
COLUNAS_SEM_TRADUCAO_TITULO = ["StreamingTV", "Churn"]  # mantém título original

#LIMPEZA LEVE DE STRINGS SEM TRANSFORMAR NAN EM STRING
#sem a transformação os dados ficam viesados, pois nulo vira string e impede a quantificação
df.columns = df.columns.str.strip()
for c in df.select_dtypes(include="object"):
    # aplica strip só em strings; mantém NaN/None intocados
    df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

translator = Translator()

#TRADUZIR OS NOMES DAS COLUNAS EM CAIXA ALTA
colunas_traduzidas = {}
for col in df.columns:
    if col in COLUNAS_SEM_TRADUCAO_TITULO:
        t = col  # mantém título original
    else:
        try:
            t = translator.translate(col, "Portuguese").result
        except Exception:
            t = col  #fallback
    colunas_traduzidas[col] = t.upper()

df.rename(columns=colunas_traduzidas, inplace=True)

#Atualiza o nome da coluna ID, caso tenha mudado na tradução (para ignorar valores)
col_id_traduzido = colunas_traduzidas.get(COLUNA_ID, COLUNA_ID)

#Detectar strings puramente numéricas
def parece_numerico(s: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+([.,]\d+)?", s))

#Coletar valores únicos a traduzir (apenas colunas object, exceto ID)
valores_alvo = set()
for col in df.select_dtypes(include="object").columns:
    if col == col_id_traduzido:
        continue
    unicos = pd.Series(df[col].dropna().unique(), dtype="object")
    for v in unicos:
        if isinstance(v, str):
            vv = v.strip()
            if vv == "" or parece_numerico(vv):
                continue
            valores_alvo.add(vv)
        else:
            # não traduzir nulos/None e não-strings
            continue

#TRADUZIR COM CACHE PULANDO OS NULOS
#Tentei a biblioteca googletrans, mas pela demora excessiva, mesmo tendo internet rápida e máquina boa, tive que cancelar a execução
cache_traducao = {}
for v in valores_alvo:
    try:
        tr = translator.translate(v, "Portuguese").result
        cache_traducao[v] = tr
        time.sleep(PAUSA)
    except Exception:
        cache_traducao[v] = v  #fallback

#FUNÇÃO PARA NAO TRADUZIR NULOS E NAO CONVERTER PARA STRING
def traduzir_seguro(x, cache):
    if pd.isna(x):
        return x
    if not isinstance(x, str):
        return x
    key = x.strip()
    return cache.get(key, x)

# APLICAR TRADUÇÃO E CAIXAS
for col in df.select_dtypes(include="object").columns:
    if col == col_id_traduzido:
        continue  # não traduz valores do ID
    df[col] = df[col].apply(lambda x: traduzir_seguro(x, cache_traducao))# aplica tradução sem mexer nos nulos
    if col in ["StreamingTV", "Churn"]: # StreamingTV e Churn ficam em CAIXA ALTA; demais, em minúsculo (sem tocar nulos)
        df[col] = df[col].apply(lambda x: (x.upper() if isinstance(x, str) else x))
    else:
        df[col] = df[col].apply(lambda x: (x.lower() if isinstance(x, str) else x))

df.to_csv("CHURN_TELECON_TRADUZIDO.csv", index=False, encoding="utf-8-sig")

print("\n✅ Tradução concluída com sucesso!\n")
print("\nArquivo salvo como 'CHURN_TELECON_TRADUZIDO.csv'\n")

print('\nDataFrame após tradução: \n')
print('\n', df.head().to_string(), '\n')

#conferir as quantidades de dados em relação à verificação anterior para averiguar a quantidade de valores nulos
print('\nVERIFICAÇÃO DO DATAFRAME TRADUZIDO: \n')
print(df.info())

#ANALISAR OS DADOS ÚNICOS PARA CLASSIFICAR AS VARIÁVEIS EM CATEGÓRICAS OU NUMÉRICAS
#SE TIVER POUCOS VALORES ÚNICOS É CATEGÓRICA E SE TIVER MUITOS É NUMÉRICA
print('\nANÁLISE DE DADOS ÚNICOS: \n')
print(df.nunique())

#categóricas
colunas = ['GÊNERO', 'TIPO_CONTRATO', 'CHURN',
           'IDOSO', 'CASADO', 'DEPENDENTES',
           'SERVIÇO TELEFÔNICO', 'SERVICO_INTERNET',
           'SERVICO_SEGURANCA', 'SUPORTE_TECNICO', 'STREAMINGTV',
           'MÉTODO DE PAGAMENTO']

#conferindo os valores das categóricas
print('\nCONFERINDO OS VALORES INSERIDOS NAS COLUNAS DE VARIÁVEIS CATEGÓRICAS: \n')
for c in colunas:
    print(f"\n📊 Coluna: {c}")
    print(df[c].value_counts(dropna=False).to_frame(name='Contagem'))

#corrigindo as entradas da coluna gênero para não enviesar a análise
df['GÊNERO'] = df['GÊNERO'].replace({'f': 'fêmea', 'm': 'macho', 'F': 'fêmea', 'M': 'macho'})
print(f"\n📊 COLUNA GÊNERO APÓS PADRONIZAÇÃO:")
print(df['GÊNERO'].value_counts(dropna=False).to_frame(name='Contagem'))

#CHECAR VALORES NULOS
print('\nANÁLISE DE DADOS NULOS: \n')
print(df.isnull().sum())
print('\nPORCENTAGEM DE DADOS NULOS: \n')
print(df.isnull().mean() * 100)

#CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
#descobri um jeito mais fácil após ter feito uma por uma.

# df['GÊNERO_CATCODES'] = df['GÊNERO'].astype('category').cat.codes
# print(f"\n📊 COLUNA GÊNERO APÓS CODIFICAÇÃO:")
# print(df[['GÊNERO', 'GÊNERO_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['TIPO_CONTRATO_CATCODES'] = df['TIPO_CONTRATO'].astype('category').cat.codes
# print(f"\n📊 COLUNA TIPO_CONTRATO APÓS CODIFICAÇÃO:")
# print(df[['TIPO_CONTRATO', 'TIPO_CONTRATO_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['CHURN_CATCODES'] = df['CHURN'].astype('category').cat.codes
# print(f"\n📊 COLUNA CHURN APÓS CODIFICAÇÃO:")
# print(df[['CHURN', 'CHURN_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['IDOSO_CATCODES'] = df['IDOSO'].astype('category').cat.codes
# print(f"\n📊 COLUNA IDOSO APÓS CODIFICAÇÃO:")
# print(df[['IDOSO', 'IDOSO_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['CASADO_CATCODES'] = df['CASADO'].astype('category').cat.codes
# print(f"\n📊 COLUNA CASADO APÓS CODIFICAÇÃO:")
# print(df[['CASADO', 'CASADO_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['DEPENDENTES_CATCODES'] = df['DEPENDENTES'].astype('category').cat.codes
# print(f"\n📊 COLUNA DEPENDENTES APÓS CODIFICAÇÃO:")
# print(df[['DEPENDENTES', 'DEPENDENTES_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['SERVIÇO TELEFÔNICO_CATCODES'] = df['SERVIÇO TELEFÔNICO'].astype('category').cat.codes
# print(f"\n📊 COLUNA SERVIÇO TELEFÔNICO APÓS CODIFICAÇÃO:")
# print(df[['SERVIÇO TELEFÔNICO', 'SERVIÇO TELEFÔNICO_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['SERVICO_INTERNET_CATCODES'] = df['SERVICO_INTERNET'].astype('category').cat.codes
# print(f"\n📊 COLUNA SERVICO_INTERNET APÓS CODIFICAÇÃO:")
# print(df[['SERVICO_INTERNET', 'SERVICO_INTERNET_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['SERVICO_SEGURANCA_CATCODES'] = df['SERVICO_SEGURANCA'].astype('category').cat.codes
# print(f"\n📊 COLUNA SERVICO_SEGURANCA APÓS CODIFICAÇÃO:")
# print(df[['SERVICO_SEGURANCA', 'SERVICO_SEGURANCA_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['SUPORTE_TECNICO_CATCODES'] = df['SUPORTE_TECNICO'].astype('category').cat.codes
# print(f"\n📊 COLUNA SUPORTE_TECNICO APÓS CODIFICAÇÃO:")
# print(df[['SUPORTE_TECNICO', 'SUPORTE_TECNICO_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
# df['STREAMINGTV_CATCODES'] = df['STREAMINGTV'].astype('category').cat.codes
# print(f"\n📊 COLUNA STREAMINGTV APÓS CODIFICAÇÃO:")
# print(df[['STREAMINGTV', 'STREAMINGTV_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))
#
#METODO TEM ASCENTO, MAS FICA VERDE
# df['METODO DE PAGAMENTO_CATCODES'] = df['METODO DE PAGAMENTO'].astype('category').cat.codes
# print(f"\n📊 COLUNA METODO DE PAGAMENTO APÓS CODIFICAÇÃO:")
# print(df[['METODO DE PAGAMENTO', 'METODO DE PAGAMENTO_CATCODES']].value_counts(dropna=False).to_frame(name='Contagem'))

#NOVAS COLUNAS PARA ANÁLISE DA CORRELAÇÃO COM A COLUNA CHURN
#colunas categóricas
#df['GÊNERO_SEM_NULOS'] = df['GÊNERO'].fillna(pd.NA)
df['GÊNERO_MAIOR_GRUPO'] = df['GÊNERO'].fillna("macho")
df['GÊNERO_MENOR_GRUPO'] = df['GÊNERO'].fillna("fêmea")

# #tratamento de outlier nas colunas discretas
# #ignorar coluna idoso
# for col in df.select_dtypes(include=['float64', 'int64']).columns:
#     # Calcula limites pelo método IQR
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     limite_inferior = Q1 - 1.5 * IQR
#     limite_superior = Q3 + 1.5 * IQR
#
#     # Cria nova coluna sem outliers
#     #necessidade de preencher os valores dos outliers com NaN para não alterar a quantidade de linhas do dataframe
#     nova_coluna = f'{col}_SEM_OUTLIER'
#     df[nova_coluna] = df[col].apply(
#         lambda x: x if (limite_inferior <= x <= limite_superior) else pd.NA
#     )
#
#     print(f"✅ Coluna criada: {nova_coluna}")

#colunas discretas

df['TEMPO_COMO_CLIENTE_MEDIA'] = df['TEMPO_COMO_CLIENTE'].fillna(df['TEMPO_COMO_CLIENTE'].mean())
df['TEMPO_COMO_CLIENTE_MEDIANA'] = df['TEMPO_COMO_CLIENTE'].fillna(df['TEMPO_COMO_CLIENTE'].median())

# df['TEMPO_COMO_CLIENTE_SEM_OUTLIER_SEM_NULOS'] = df['TEMPO_COMO_CLIENTE_SEM_OUTLIER'].fillna(pd.NA)
# df['TEMPO_COMO_CLIENTE_SEM_OUTLIER_MEDIA'] = df['TEMPO_COMO_CLIENTE_SEM_OUTLIER'].fillna(df['TEMPO_COMO_CLIENTE_SEM_OUTLIER'].mean())
# df['TEMPO_COMO_CLIENTE_SEM_OUTLIER_MEDIANA'] = df['TEMPO_COMO_CLIENTE_SEM_OUTLIER'].fillna(df['TEMPO_COMO_CLIENTE_SEM_OUTLIER'].median())

#df['PAGAMENTO_MENSAL_SEM_NULOS'] = df['PAGAMENTO_MENSAL'].fillna(pd.NA)
df['PAGAMENTO_MENSAL_MEDIA'] = df['PAGAMENTO_MENSAL'].fillna(df['PAGAMENTO_MENSAL'].mean())
df['PAGAMENTO_MENSAL_MEDIANA'] = df['PAGAMENTO_MENSAL'].fillna(df['PAGAMENTO_MENSAL'].median())

# df['PAGAMENTO_MENSAL_SEM_OUTLIER_SEM_NULOS'] = df['PAGAMENTO_MENSAL_SEM_OUTLIER'].fillna(pd.NA)
# df['PAGAMENTO_MENSAL_SEM_OUTLIER_MEDIA'] = df['PAGAMENTO_MENSAL_SEM_OUTLIER'].fillna(df['PAGAMENTO_MENSAL_SEM_OUTLIER'].mean())
# df['PAGAMENTO_MENSAL_SEM_OUTLIER_MEDIANA'] = df['PAGAMENTO_MENSAL_SEM_OUTLIER'].fillna(df['PAGAMENTO_MENSAL_SEM_OUTLIER'].median())

#df['TOTAL_PAGAMENTO_SEM_NULOS'] = df['TOTAL_PAGAMENTO'].fillna(pd.NA)
df['TOTAL_PAGAMENTO_MEDIA'] = df['TOTAL_PAGAMENTO'].fillna(df['TOTAL_PAGAMENTO'].mean())
df['TOTAL_PAGAMENTO_MEDIANA'] = df['TOTAL_PAGAMENTO'].fillna(df['TOTAL_PAGAMENTO'].median())

# df['TOTAL_PAGAMENTO_SEM_OUTLIER_SEM_NULOS'] = df['TOTAL_PAGAMENTO_SEM_OUTLIER'].fillna(pd.NA)
# df['TOTAL_PAGAMENTO_SEM_OUTLIER_MEDIA'] = df['TOTAL_PAGAMENTO_SEM_OUTLIER'].fillna(df['TOTAL_PAGAMENTO_SEM_OUTLIER'].mean())
# df['TOTAL_PAGAMENTO_SEM_OUTLIER_MEDIANA'] = df['TOTAL_PAGAMENTO_SEM_OUTLIER'].fillna(df['TOTAL_PAGAMENTO_SEM_OUTLIER'].median())

#CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS

# Lista de colunas para não codificar
colunas_excluidas = ['ID DO CLIENTE', 'IDOSO', 'TEMPO_COMO_CLIENTE','TEMPO_COMO_CLIENTE_MEDIA', 'TEMPO_COMO_CLIENTE_MEDIANA',
                     'PAGAMENTO_MENSAL', 'PAGAMENTO_MENSAL_MEDIA', 'PAGAMENTO_MENSAL_MEDIANA',
                     'TOTAL_PAGAMENTO', 'TOTAL_PAGAMENTO_MEDIA', 'TOTAL_PAGAMENTO_MEDIANA']


# colunas_excluidas = ['ID DO CLIENTE', 'IDOSO', 'TEMPO_COMO_CLIENTE', 'PAGAMENTO_MENSAL','TOTAL_PAGAMENTO','PAGAMENTO_MENSAL_SEM_OUTLIER',
#                      'PAGAMENTO_MENSAL_MEDIA', 'PAGAMENTO_MENSAL_MEDIANA', 'IDOSO_SEM_OUTLIER', 'PAGAMENTO_MENSAL_SEM_OUTLIER_SEM_NULOS',
#                      'PAGAMENTO_MENSAL_SEM_OUTLIER_MEDIA', 'PAGAMENTO_MENSAL_SEM_OUTLIER_MEDIANA']

# Converter ambas as listas para maiúsculas (por segurança)
colunas_excluidas = [c.upper() for c in colunas_excluidas]

# Identificar colunas categóricas elegíveis
colunas_categoricas = [
    col for col in df.select_dtypes(include='object').columns
    if col.upper() not in colunas_excluidas
]

# Criar códigos para cada coluna categórica
for col in colunas_categoricas:
    nova_coluna = f"{col}_CATCODES"
    df[nova_coluna] = df[col].astype('category').cat.codes
    print(f"\n✅ Codificada: {col} → {nova_coluna}")

for col in colunas_categoricas:
    print(f"\n🔎 Mapeamento da coluna {col}:")
    print(dict(enumerate(df[col].astype('category').cat.categories)))

print('\nANÁLISE DE DADOS NULOS DO DATAFRAME ORIGINAL: \n')
print(df.isnull().sum())

#necessidade de criar novos dataframes sem os nulos
df_sem_nulos_geral = df.dropna()
df_genero_sem_nulo = df.dropna(subset=['GÊNERO'])
df_serviço_telefonico_sem_nulo = df.dropna(subset=['SERVIÇO TELEFÔNICO'])
df_pagamento_mensal_sem_nulos = df.dropna(subset=['PAGAMENTO_MENSAL'])

#sem nulos geral
colunas_categoricas1 = [
    col for col in df_sem_nulos_geral.select_dtypes(include='object').columns
    if col.upper() not in colunas_excluidas
]
for col in colunas_categoricas1:
    nova_coluna = f"{col}_CATCODES"
    df_sem_nulos_geral[nova_coluna] = df_sem_nulos_geral[col].astype('category').cat.codes

#genero sem nulo
colunas_categoricas2 = [
    col for col in df_genero_sem_nulo.select_dtypes(include='object').columns
    if col.upper() not in colunas_excluidas
]
for col in colunas_categoricas2:
    nova_coluna = f"{col}_CATCODES"
    df_genero_sem_nulo[nova_coluna] = df_genero_sem_nulo[col].astype('category').cat.codes

#serviço telefonico sem nulo
colunas_categoricas3 = [
    col for col in df_serviço_telefonico_sem_nulo.select_dtypes(include='object').columns
    if col.upper() not in colunas_excluidas
]
for col in colunas_categoricas3:
    nova_coluna = f"{col}_CATCODES"
    df_serviço_telefonico_sem_nulo[nova_coluna] = df_serviço_telefonico_sem_nulo[col].astype('category').cat.codes

#pagamento mensal sem nulo
colunas_categoricas4 = [
    col for col in df_pagamento_mensal_sem_nulos.select_dtypes(include='object').columns
    if col.upper() not in colunas_excluidas
]
for col in colunas_categoricas4:
    nova_coluna = f"{col}_CATCODES"
    df_pagamento_mensal_sem_nulos[nova_coluna] = df_pagamento_mensal_sem_nulos[col].astype('category').cat.codes

print('\nVERIFICAÇÃO FINAL DO DATAFRAME ORIGINAL: \n')
print(df.head().to_string(), '\n')
print(df.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME ORIGINAL: \n')
print(df.isnull().sum())

print('\nVERIFICAÇÃO FINAL DO DATAFRAME SEM TODOS OS NULOS: \n')
print(df.head().to_string(), '\n')
print(df.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME SEM TODOS OS NULOS : \n')
print(df.isnull().sum())

print('\nVERIFICAÇÃO FINAL DO DATAFRAME SEM NULO GÊNERO: \n')
print(df.head().to_string(), '\n')
print(df.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME SEM NULO GÊNERO: \n')
print(df.isnull().sum())

print('\nVERIFICAÇÃO FINAL DO DATAFRAME SEM NULO SERVIÇO TELEFÔNICO: \n')
print(df.head().to_string(), '\n')
print(df.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME SEM NULO SERVIÇO TELEFÔNICO: \n')
print(df.isnull().sum())

print('\nVERIFICAÇÃO FINAL DO DATAFRAME SEM NULO PAGAMENTO MENSAL: \n')
print(df.head().to_string(), '\n')
print(df.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME SEM NULO PAGAMENTO MENSAL: \n')
print(df.isnull().sum())

#TEMOS TODAS AS COLUNAS NECESSÁRIAS PARA CORRELACIONAR COM A COLUNA CHURN E DECIDIR QUAL ESTRATÉGIA USAR
#correlação ponto-bisseria
# Filtra apenas colunas numéricas (exceto a própria CHURN_CATCODES)
numeric_cols = [
    col for col in df.select_dtypes(include=['float64', 'int64', 'int8']).columns
    if col != 'CHURN_CATCODES'
]

# # Calcula correlação point-biserial (Pearson para variável binária)
# resultados = []
# for col in numeric_cols:
#     # Remove valores nulos antes da correlação
#     subset = df[['CHURN_CATCODES', col]]#.dropna()
#     if subset[col].nunique() > 1:  # evita erro em colunas constantes
#         corr, p = pointbiserialr(subset['CHURN_CATCODES'], subset[col])
#         resultados.append({'Variável': col, 'Correlação': corr})#, 'p-valor': p})
#
# # Cria DataFrame ordenado
# corr_df = pd.DataFrame(resultados).sort_values('Correlação', ascending=False)
#
# # Exibe tabela
# print("\n📊 Correlação point-biserial com CHURN:\n")
# print(corr_df.to_string(index=False))

# # Plot opcional
# plt.figure(figsize=(16, 12))
# sns.heatmap(corr_df[['Correlação']].set_index(corr_df['Variável']).T,
#             annot=True, cmap='coolwarm', center=0)
# plt.title('Correlação Point-Biserial com CHURN', fontsize=14, weight='bold')
# plt.yticks([])
# plt.tight_layout()
# plt.show()

