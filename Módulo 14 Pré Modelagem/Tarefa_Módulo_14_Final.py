import pandas as pd
from scipy.stats import pointbiserialr
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

#NOVAS COLUNAS PARA ANÁLISE DA CORRELAÇÃO COM A COLUNA CHURN
#colunas categóricas
df['GÊNERO_MAIOR_GRUPO'] = df['GÊNERO'].fillna("macho")
df['GÊNERO_MENOR_GRUPO'] = df['GÊNERO'].fillna("fêmea")
df['SERVIÇO TELEFÔNICO_MAIOR_GRUPO'] = df['SERVIÇO TELEFÔNICO'].fillna("sim")
df['SERVIÇO TELEFÔNICO_MENOR_GRUPO'] = df['SERVIÇO TELEFÔNICO'].fillna("não")
#colunas discretas
#df['TEMPO_COMO_CLIENTE_MEDIA'] = df['TEMPO_COMO_CLIENTE'].fillna(df['TEMPO_COMO_CLIENTE'].mean())
#df['TEMPO_COMO_CLIENTE_MEDIANA'] = df['TEMPO_COMO_CLIENTE'].fillna(df['TEMPO_COMO_CLIENTE'].median())
df['PAGAMENTO_MENSAL_MEDIA'] = df['PAGAMENTO_MENSAL'].fillna(df['PAGAMENTO_MENSAL'].mean())
df['PAGAMENTO_MENSAL_MEDIANA'] = df['PAGAMENTO_MENSAL'].fillna(df['PAGAMENTO_MENSAL'].median())
#df['TOTAL_PAGAMENTO_MEDIA'] = df['TOTAL_PAGAMENTO'].fillna(df['TOTAL_PAGAMENTO'].mean())
#df['TOTAL_PAGAMENTO_MEDIANA'] = df['TOTAL_PAGAMENTO'].fillna(df['TOTAL_PAGAMENTO'].median())

#CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS

# Lista de colunas para não codificar
colunas_excluidas = ['ID DO CLIENTE', 'IDOSO', 'TEMPO_COMO_CLIENTE',
                     'PAGAMENTO_MENSAL', 'PAGAMENTO_MENSAL_MEDIA', 'PAGAMENTO_MENSAL_MEDIANA',
                     'TOTAL_PAGAMENTO']

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
    df_sem_nulos_geral.loc[:, nova_coluna] = df_sem_nulos_geral[col].astype('category').cat.codes

#genero sem nulo
colunas_categoricas2 = [
    col for col in df_genero_sem_nulo.select_dtypes(include='object').columns
    if col.upper() not in colunas_excluidas
]
for col in colunas_categoricas2:
    nova_coluna = f"{col}_CATCODES"
    df_genero_sem_nulo.loc[:, nova_coluna] = df_genero_sem_nulo[col].astype('category').cat.codes

#serviço telefonico sem nulo
colunas_categoricas3 = [
    col for col in df_serviço_telefonico_sem_nulo.select_dtypes(include='object').columns
    if col.upper() not in colunas_excluidas
]
for col in colunas_categoricas3:
    nova_coluna = f"{col}_CATCODES"
    df_serviço_telefonico_sem_nulo.loc[:, nova_coluna] = df_serviço_telefonico_sem_nulo[col].astype('category').cat.codes

#pagamento mensal sem nulo
colunas_categoricas4 = [
    col for col in df_pagamento_mensal_sem_nulos.select_dtypes(include='object').columns
    if col.upper() not in colunas_excluidas
]
for col in colunas_categoricas4:
    nova_coluna = f"{col}_CATCODES"
    df_pagamento_mensal_sem_nulos.loc[:, nova_coluna] = df_pagamento_mensal_sem_nulos[col].astype('category').cat.codes

print('\nVERIFICAÇÃO FINAL DO DATAFRAME ORIGINAL: \n')
print(df.head().to_string(), '\n')
print(df.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME ORIGINAL: \n')
print(df.isnull().sum())

print('\nVERIFICAÇÃO FINAL DO DATAFRAME SEM TODOS OS NULOS: \n')
print(df_sem_nulos_geral.head().to_string(), '\n')
print(df_sem_nulos_geral.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME SEM TODOS OS NULOS : \n')
print(df_sem_nulos_geral.isnull().sum())

print('\nVERIFICAÇÃO FINAL DO DATAFRAME SEM NULO GÊNERO: \n')
print(df_genero_sem_nulo.head().to_string(), '\n')
print(df_genero_sem_nulo.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME SEM NULO GÊNERO: \n')
print(df_genero_sem_nulo.isnull().sum())

print('\nVERIFICAÇÃO FINAL DO DATAFRAME SEM NULO SERVIÇO TELEFÔNICO: \n')
print(df_serviço_telefonico_sem_nulo.head().to_string(), '\n')
print(df_serviço_telefonico_sem_nulo.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME SEM NULO SERVIÇO TELEFÔNICO: \n')
print(df_serviço_telefonico_sem_nulo.isnull().sum())

print('\nVERIFICAÇÃO FINAL DO DATAFRAME SEM NULO PAGAMENTO MENSAL: \n')
print(df_pagamento_mensal_sem_nulos.head().to_string(), '\n')
print(df_pagamento_mensal_sem_nulos.info(),'\n')
print('\nANÁLISE DE DADOS NULOS DO DATAFRAME SEM NULO PAGAMENTO MENSAL: \n')
print(df_pagamento_mensal_sem_nulos.isnull().sum())

#TEMOS TODAS AS COLUNAS NECESSÁRIAS PARA CORRELACIONAR COM A COLUNA CHURN E DECIDIR QUAL ESTRATÉGIA USAR

#correlação ponto-bisserial geral
# Filtra apenas colunas numéricas (exceto a própria CHURN_CATCODES)
numeric_cols = [
    col for col in df.select_dtypes(include=['float64', 'int64', 'int8']).columns
    if col != 'CHURN_CATCODES'
]

# Calcula correlação point-biserial (Pearson para variável binária)
resultados = []
for col in numeric_cols:
    # Remove valores nulos antes da correlação
    subset = df[['CHURN_CATCODES', col]].dropna()
    if subset[col].nunique() > 1:  # evita erro em colunas constantes
        corr, p = pointbiserialr(subset['CHURN_CATCODES'], subset[col])
        resultados.append({'Variável': col, 'Correlação': corr})#, 'p-valor': p})

# Cria DataFrame ordenado
corr_df = pd.DataFrame(resultados).sort_values('Correlação', ascending=False)

#sem nulos
numeric_cols = [
    col for col in df_sem_nulos_geral.select_dtypes(include=['float64', 'int64', 'int8']).columns
    if col != 'CHURN_CATCODES'
]
resultados = []
for col in numeric_cols:
    subset = df_sem_nulos_geral[['CHURN_CATCODES', col]].dropna()
    if subset[col].nunique() > 1:
        corr, p = pointbiserialr(subset['CHURN_CATCODES'], subset[col])
        resultados.append({'Variável': col, 'Correlação': corr})

# Cria DataFrame ordenado
corr_df_sem_nulos_geral = pd.DataFrame(resultados).sort_values('Correlação', ascending=False)

#genero sem nulos
numeric_cols = [
    col for col in df_genero_sem_nulo.select_dtypes(include=['float64', 'int64', 'int8']).columns
    if col != 'CHURN_CATCODES'
]
resultados = []
for col in numeric_cols:
    subset = df_genero_sem_nulo[['CHURN_CATCODES', col]].dropna()
    if subset[col].nunique() > 1:
        corr, p = pointbiserialr(subset['CHURN_CATCODES'], subset[col])
        resultados.append({'Variável': col, 'Correlação': corr})

# Cria DataFrame ordenado
corr_df_genero_sem_nulo = pd.DataFrame(resultados).sort_values('Correlação', ascending=False)

#serviço telefonico sem nulos
numeric_cols = [
    col for col in df_serviço_telefonico_sem_nulo.select_dtypes(include=['float64', 'int64', 'int8']).columns
    if col != 'CHURN_CATCODES'
]
resultados = []
for col in numeric_cols:
    subset = df_serviço_telefonico_sem_nulo[['CHURN_CATCODES', col]].dropna()
    if subset[col].nunique() > 1:
        corr, p = pointbiserialr(subset['CHURN_CATCODES'], subset[col])
        resultados.append({'Variável': col, 'Correlação': corr})

# Cria DataFrame ordenado
corr_df_serviço_telefonico_sem_nulo = pd.DataFrame(resultados).sort_values('Correlação', ascending=False)

#pagamento mensal sem nulos
numeric_cols = [
    col for col in df_pagamento_mensal_sem_nulos.select_dtypes(include=['float64', 'int64', 'int8']).columns
    if col != 'CHURN_CATCODES'
]
resultados = []
for col in numeric_cols:
    subset = df_pagamento_mensal_sem_nulos[['CHURN_CATCODES', col]].dropna()
    if subset[col].nunique() > 1:
        corr, p = pointbiserialr(subset['CHURN_CATCODES'], subset[col])
        resultados.append({'Variável': col, 'Correlação': corr})

# Cria DataFrame ordenado
corr_df_pagamento_mensal_sem_nulos = pd.DataFrame(resultados).sort_values('Correlação', ascending=False)

#TABELA PARA DECISÃO
#Anexar um rótulo de cenário e concatenar tudo
todas = pd.concat([
    corr_df.assign(Cenário='Original'),
    corr_df_sem_nulos_geral.assign(Cenário='Sem nulos (geral)'),
    corr_df_genero_sem_nulo.assign(Cenário='Sem nulos (GÊNERO)'),
    corr_df_serviço_telefonico_sem_nulo.assign(Cenário='Sem nulos (SERVIÇO TEL)'),
    corr_df_pagamento_mensal_sem_nulos.assign(Cenário='Sem nulos (PAGAMENTO)')
], ignore_index=True)

#pivot: Variável como índice, Cenário como colunas, Correlação como valores
tabela_final = todas.pivot_table(
    index='Variável', columns='Cenário', values='Correlação', aggfunc='first'
)

#Ordenar pelas maiores correlações (em módulo) em qualquer cenário
ordem = tabela_final.abs().max(axis=1).sort_values(ascending=False).index
tabela_final = tabela_final.loc[ordem]

#Arredondar e mostrar/exports
tabela_final = tabela_final.round(4)
print("\n📊 TABELA FINAL DE CORRELAÇÕES (variáveis × cenários):\n")
print(tabela_final.to_string())
df.to_csv("CHURN_TELECON_FINAL.csv", index=False, encoding="utf-8-sig")
