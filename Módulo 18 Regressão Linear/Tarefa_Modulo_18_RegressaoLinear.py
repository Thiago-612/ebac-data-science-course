import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('ALUGUEL_MOD12.csv',
    sep=';',
    encoding='utf-8',
    na_values=['', ' ', 'NA', 'None']
)
print('\nVERIFICAÇÃO DO DATAFRAME: \n')
print('\n', df.head(20).to_string(), '\n')
print(df.info())
print(df.describe().to_string(), '\n')

#CONFERENCIA DE VALORES DAS COLUNAS

"""
Análise:
Ausência de valores nulos, valores corretos, média não muito distante da mediana (normal cauda a direita em aluguéis).
Correlações positivas entre as variáveis comparadas com o valor do aluguel.
Ausência de outliers, valores altos de aluguel são dados importantes.
Variáveis com relação linear positiva.
"""

colunas = ['Valor_Aluguel', 'Valor_Condominio', 'Metragem', 'N_Quartos', 'N_banheiros',
    'N_Suites', 'N_Vagas']

print('\nCONFERINDO OS VALORES INSERIDOS NAS COLUNAS: \n')
for c in colunas:
    print(f"\n📊 Coluna: {c}")
    print(df[c].value_counts(dropna=False).to_frame(name='Contagem'))

#Análise da média com a mediana(50%)

colunas_numericas = [
    col for col in df.select_dtypes(include=['int64']).columns
]

# Listas para armazenar resultados
resultados = []

for col in colunas_numericas:
    media = df[col].mean()
    mediana = df[col].median()

    if pd.notna(media) and pd.notna(mediana):  # ignora colunas vazias
        diff = media - mediana
        resultados.append({
            'Coluna': col,
            'Média': media,
            'Mediana': mediana,
            'Diferença (Média - Mediana)': diff
        })

tabela = pd.DataFrame(resultados)

# Separar por tipo de assimetria
tabela_maior_media = tabela[tabela['Diferença (Média - Mediana)'] > 0].sort_values('Diferença (Média - Mediana)',ascending=False)
tabela_menor_media = tabela[tabela['Diferença (Média - Mediana)'] < 0].sort_values('Diferença (Média - Mediana)',ascending=True)

print('\nCOMPARANDO OS VALORES DA MÉDIA E MEDIANA INSERIDOS NAS COLUNAS DE VARIÁVEIS NUMÉRICAS:')
print('Valores altos na coluna Diferença indicam possíveis outliers')
print("\n📊 Colunas com MÉDIA MAIOR que MEDIANA (cauda à direita):")
print(tabela_maior_media.to_string(index=False))
print("\n📉 Colunas com MÉDIA MENOR que MEDIANA (cauda à esquerda):")
print(tabela_menor_media.to_string(index=False))

#Conferindo outliers
for col in colunas_numericas:
    fig = px.box(
        data_frame=df,
        y=col,
        points="outliers",
        title=f"Boxplot da variável {col}",
        template="plotly_white"
    )

    fig.update_layout(
        yaxis_title=col,
        xaxis_title="",
        title_font_size=20
        )

    fig.show()

for col in colunas:
    if col != 'Valor_Aluguel':
        fig = px.scatter(
            df,
            x=col,
            y='Valor_Aluguel',
            trendline='ols',
            title=f'Valor do Aluguel vs {col}',
            template='plotly_white'
        )
        fig.show()

print('\nCORRELAÇÃO:\n')
print(df[colunas].corr()['Valor_Aluguel'].sort_values(ascending=False))

#SEPARAÇÃO BASE DE TREINO E TESTE

x = df.drop('Valor_Aluguel', axis=1)
y = df['Valor_Aluguel']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('\nCONFERÊNCIA DAS BASES DE TREINO E TESTE:\n')
print('\n Tamanho do x_train:', x_train.shape)
print('\n Tamanho do y_train:', y_train.shape)
print('\n Tamanho do x_test:', x_test.shape)
print('\n Tamanho do y_test:', y_test.shape)

#REGRESSÃO LINEAR SIMPLES
"""
Análise:
Stats models para uma análise inicial e sklearn após para treinar e validar.
Valor do R² utilizando somente a variável metragem demonstra que o modelo explica 52% da variação do aluguel.
Valor do R² ajustado igual do R² indica que o modelo não tem variáveis inúteis.
Os coeficientes mostram quanto o valor do aluguel varia quando a variável aumenta uma unidade.
A cada m² o valor do aluguel aumenta R$21, utilizando todas as variáveis.
A cada m² o valor do aluguel aumenta R$35, utilizando somente a variável metragem. Metragem puxa o efeito de outras variáveis.
O erro padrão do coeficiente (std err) mede o grau de precisão com que o modelo de regressão
estima o valor desconhecido do coeficiente.
Metragem possui std err baixo.
Condition number é alto, o que indica multicolinearidade.
A reta de regressão do teste explica bem o modelo até 200m², após essa metragem o aluguel distancia muito da reta,
porém isso não significa que temos outliers. Possivelmente são imóveis de luxo em bairros nobres.
Separar os dados por bairro tornaria o modelo mais realista.
"""

#Stats models
# Base de Treino
# Variáveis explicativas
x1_train = x_train[['Metragem']]

# Variável alvo
y1_train = y_train

# Constante adicionada
x1_train = sm.add_constant(x1_train)

# Ajuste do modelo
modelo_treino = sm.OLS(y1_train, x1_train).fit()

print('\n___TREINO___')
print('RESULTADOS DA REGRESSÃO LINEAR SIMPLES STATS MODELS:\n')
print(modelo_treino.summary())

#Equação da reta
coef_treino = modelo_treino.params

equacao_treino = f"""
Valor_Aluguel =
{coef_treino['const']:.2f} + {coef_treino['Metragem']:.4f} * Metragem
"""

print('\n___TREINO___')
print('EQUAÇÃO DA RETA DE REGRESSÃO:')
print(equacao_treino)

#Gráfico da reta de regressão
fig = px.scatter(
    df,
    x="Metragem",
    y="Valor_Aluguel",
    trendline="ols",
    title="Reta de regressão de todo o dataframe"
)

fig.update_layout(
    xaxis_title="Metragem",
    yaxis_title="Aluguel",
    title_font_size=20,
    template="plotly_white"
)

fig.show()


# Base de Teste
x1_teste = sm.add_constant(x_test[['Metragem']], has_constant='add')
y_previsto = modelo_treino.predict(x1_teste)

print('\n___TESTE___')
print('VERIFICAÇÃO DAS PREVISÕES STATS MODELS:\n')
print("R2 teste:", r2_score(y_test, y_previsto))
print("MAE teste:", mean_absolute_error(y_test, y_previsto))
print("RMSE teste:", np.sqrt(mean_squared_error(y_test, y_previsto)))

#Sklearn
#Treino
x2_train = x_train[['Metragem']]
y2_train = y_train

modelo_sk = LinearRegression()
modelo_sk.fit(x2_train, y2_train)

intercepto = modelo_sk.intercept_
coeficiente = modelo_sk.coef_[0]

print('\n___TREINO___')
print('RESULTADOS SKLEARN:\n')
print("\nParâmetros do modelo:")
print("Intercepto:", intercepto)
print("Coeficiente (Metragem):", coeficiente)

equacao = f"""
Valor_Aluguel =
{intercepto:.2f} + {coeficiente:.4f} * Metragem
"""
print("\nEquação da regressão:")
print(equacao)

r2_treino = modelo_sk.score(x2_train, y2_train)
print("\nCoeficiente de Determinação (R²) nos Dados de Treino:", r2_treino)

print('\n___TESTE___')
print('RESULTADOS SKLEARN:\n')
x2_teste = x_test[['Metragem']]
y2_teste = y_test
previsoes = modelo_sk.predict(x2_teste)
r2_teste = modelo_sk.score(x2_teste, y2_teste)
print("\nCoeficiente de Determinação (R²) nos Dados de Teste:", r2_teste)
"""
Análise:
O R² dos dados de teste demonstra que o modelo explica 63% da variação no preço do aluguel, enquanto
o R² dos dados do treino explica 59%. Desta forma, o modelo não se ajusta aos dados de treino e não 
consegue obter a relação que os dados independentes geram na variável dependente (Aluguel).
Variáveis adicionais, como preço do imóvel e bairro, tornariam o modelo mais eficaz com a possibilidade
de separar melhor o conjunto dos imóveis.
"""

#REGRESSÃO LINEAR MÚLTIPLA
"""
Análise:
O R² dos dados de teste demonstra que o modelo explica 59% da variação no preço do aluguel, enquanto
o R² dos dados do treino explica 5%. Desta forma, o modelo não se ajusta aos dados de treino e não 
consegue obter a relação que os dados independentes geram na variável dependente (Aluguel).
Variáveis adicionais, como preço do imóvel e bairro, tornariam o modelo mais eficaz com a possibilidade
de separar melhor o conjunto dos imóveis.
Valor do R² ajustado igual do R² indica que o modelo não tem variáveis inúteis.
Cada quarto adicional diminui o aluguel, isso ocorre pois metragem/quartos/suítes competem entre si.
O número de quartos pode aumentar sem aumentar a metragem.
"""
#Stats models
# Base de Treino
# Variáveis explicativas
x3_train = x_train[['Valor_Condominio', 'Metragem', 'N_Quartos', 'N_banheiros', 'N_Suites', 'N_Vagas']]

# Variável alvo
y3_train = y_train

# Constante adicionada
x3_train = sm.add_constant(x3_train)

# Ajuste do modelo
modelo_treino_multi = sm.OLS(y3_train, x3_train).fit()

print('\n___TREINO___')
print('RESULTADOS DA REGRESSÃO LINEAR MÚLTIPLA STATS MODELS:\n')
print(modelo_treino_multi.summary())

#Sklearn
#Treino
x4_train = x_train[['Valor_Condominio', 'Metragem', 'N_Quartos', 'N_banheiros', 'N_Suites', 'N_Vagas']]
y4_train = y_train

modelo_sk_multi = LinearRegression()
modelo_sk_multi.fit(x4_train, y4_train)

intercepto_multi = modelo_sk_multi.intercept_

print('\n___TREINO___')
print('RESULTADOS DA REGRESSÃO LINEAR MÚLTIPLA SKLEARN:\n')
print("\nParâmetros do modelo:")
print("Intercepto:", intercepto_multi)
print("\nCoeficientes por variável:")
for nome, coef in zip(x4_train.columns, modelo_sk_multi.coef_):
    print(f"{nome}: {coef:.4f}")

r2_treino_multi = modelo_sk_multi.score(x4_train, y4_train)
print("\nCoeficiente de Determinação (R²) nos Dados de Treino:", r2_treino_multi)

print('\n___TESTE___')
print('RESULTADOS SKLEARN:\n')
x3_teste = x_test[['Valor_Condominio', 'Metragem', 'N_Quartos', 'N_banheiros', 'N_Suites', 'N_Vagas']]
y3_teste = y_test
previsoes_multi = modelo_sk_multi.predict(x3_teste)
r2_teste_multi = modelo_sk_multi.score(x3_teste, y3_teste)
print("\nCoeficiente de Determinação (R²) nos Dados de Teste:", r2_teste_multi)

#MELHORIAS PROPOSTAS PELO TUTOR
"""
Como você notou que o Condition Number é alto e que variáveis como "N_Quartos" apresentam coeficientes negativos 
(devido à competição com a Metragem), tente calcular o VIF (Variance Inflation Factor). Isso ajudaria a decidir quais
variáveis remover para tornar o modelo mais estável e interpretável sem que uma "puxe" o efeito da outra.

Dado que imóveis de luxo distanciam-se muito da reta (cauda à direita), a dica é aplicar o Logaritmo na variável Valor_Aluguel.
Isso costuma linearizar a relação e reduzir o impacto dos valores extremos, resultando em um R² mais robusto e 
resíduos melhor distribuídos (mais próximos de uma distribuição normal).

Após identificar multicolinearidade no modelo completo por meio do Condition Number e do VIF, foi ajustado um modelo reduzido 
com variáveis menos correlacionadas. Sobre essa mesma estrutura, aplicou-se a transformação logarítmica na variável 
dependente (Valor_Aluguel), com o objetivo de reduzir a assimetria causada por imóveis de alto padrão.
O modelo com log apresentou resíduos mais homogêneos e métricas mais robustas, mantendo a interpretabilidade econômica dos 
coeficientes.
"""

# CALCULO DO VIF
variaveis_multi = ['Valor_Condominio', 'Metragem', 'N_Quartos', 'N_banheiros', 'N_Suites', 'N_Vagas']

X_train_multi = x_train[variaveis_multi].copy()
X_test_multi  = x_test[variaveis_multi].copy()

X_train_multi_const = sm.add_constant(X_train_multi, has_constant='add')

vif = pd.DataFrame({
    "Variável": X_train_multi_const.columns,
    "VIF": [variance_inflation_factor(X_train_multi_const.values, i)
            for i in range(X_train_multi_const.shape[1])]
}).sort_values("VIF", ascending=False)

print("\n====================")
print("VIF (MODELO MÚLTIPLO)")
print("====================")
print(vif.to_string(index=False))

# MODELO REDUZIDO COM BASE NO VIF
variaveis_reduzidas = ['Valor_Condominio', 'Metragem', 'N_banheiros', 'N_Vagas', 'N_Suites']

X_train_red = x_train[variaveis_reduzidas].copy()
X_test_red  = x_test[variaveis_reduzidas].copy()

# Statsmodels (treino)
X_train_red_const = sm.add_constant(X_train_red, has_constant='add')
modelo_red_sm = sm.OLS(y_train, X_train_red_const).fit()

print("\n====================")
print("MODELO REDUZIDO (STATS MODELS) — TREINO")
print("====================")
print(modelo_red_sm.summary())

# Statsmodels (teste)
X_test_red_const = sm.add_constant(X_test_red, has_constant='add')
pred_red_sm = modelo_red_sm.predict(X_test_red_const)

# Métricas (reduzido)
r2_red_sm = r2_score(y_test, pred_red_sm)
mae_red_sm = mean_absolute_error(y_test, pred_red_sm)
rmse_red_sm = np.sqrt(mean_squared_error(y_test, pred_red_sm))

print("\n___TESTE___")
print("MÉTRICAS — REDUZIDO (STATS MODELS)")
print("R2:", r2_red_sm)
print("MAE:", mae_red_sm)
print("RMSE:", rmse_red_sm)

#MODELO REDUZIDO COM LOG NO ALUGUEL
# Treino
y_train_log = np.log(y_train)

# X reduzido
x_log_train = x_train[['Valor_Condominio', 'Metragem', 'N_banheiros', 'N_Vagas', 'N_Suites']]
x_log_train = sm.add_constant(x_log_train)

# Ajuste
modelo_log = sm.OLS(y_train_log, x_log_train).fit()

print("\n====================")
print("MODELO REDUZIDO COM LOG(Valor_Aluguel) (STATS MODELS) — TREINO")
print("====================")
print(modelo_log.summary())

# Teste
x_log_test = sm.add_constant(
    x_test[['Valor_Condominio', 'Metragem', 'N_banheiros', 'N_Vagas', 'N_Suites']],
    has_constant='add'
)

# Previsões em log
y_pred_log = modelo_log.predict(x_log_test)

# Métricas EM LOG
r2_log = r2_score(np.log(y_test), y_pred_log)
mae_log = mean_absolute_error(np.log(y_test), y_pred_log)
rmse_log = np.sqrt(mean_squared_error(np.log(y_test), y_pred_log))

# Correção de viés com smearing para comparar o RMSE
# resíduos do treino (em log)
residuos_log = modelo_log.resid

# fator de correção
fator_smearing = np.mean(np.exp(residuos_log))

# previsão corrigida em escala real
y_pred_corrigido = np.exp(y_pred_log) * fator_smearing

# Métricas em R$
r2_real = r2_score(y_test, y_pred_corrigido)
mae_real = mean_absolute_error(y_test, y_pred_corrigido)
rmse_real = np.sqrt(mean_squared_error(y_test, y_pred_corrigido))

print("\n___TESTE___")
print("MÉTRICAS — REDUZIDO + LOG(y) (STATS MODELS)")
print("R2 (Log):", r2_log)
print("MAE (Log):", mae_log)
print("RMSE (Log):", rmse_log)
print("R2 (Real):", r2_real)
print("MAE (Real):", mae_real)
print("RMSE (Real):", rmse_real)

# Tabela de comparação
X_test_multi_const = sm.add_constant(X_test_multi, has_constant='add')
X_train_multi_const = sm.add_constant(X_train_multi, has_constant='add')
modelo_base_sm = sm.OLS(y_train, X_train_multi_const).fit()
pred_base_sm = modelo_base_sm.predict(X_test_multi_const)

r2_base = r2_score(y_test, pred_base_sm)
mae_base = mean_absolute_error(y_test, pred_base_sm)
rmse_base = np.sqrt(mean_squared_error(y_test, pred_base_sm))

tabela_comp = pd.DataFrame([
    {"Modelo": "Múltiplo (Normal)", "R2": r2_base, "MAE": mae_base, "RMSE": rmse_base},
    {"Modelo": "Reduzido (Normal)", "R2": r2_red_sm, "MAE": mae_red_sm, "RMSE": rmse_red_sm},
    {"Modelo": "Reduzido (LOG y)", "R2": r2_log, "MAE": mae_log, "RMSE": rmse_log},
]).sort_values("R2", ascending=False)

print("\n====================")
print("COMPARAÇÃO (TESTE)")
print("====================")
print(tabela_comp.to_string(index=False))


#Gráficos Plotly
def graficos_diagnostico(y_true, y_pred, titulo):
    resid = y_true - y_pred
    df_plot = pd.DataFrame({"Real": y_true, "Previsto": y_pred, "Resíduo": resid})

    # Real vs Previsto
    fig1 = px.scatter(df_plot, x="Real", y="Previsto", title=f"{titulo} — Real vs Previsto", trendline="ols")
    fig1.update_layout(template="plotly_white")
    fig1.show()

    # Resíduo vs Previsto
    fig2 = px.scatter(df_plot, x="Previsto", y="Resíduo", title=f"{titulo} — Resíduo vs Previsto")
    fig2.add_hline(y=0)
    fig2.update_layout(template="plotly_white")
    fig2.show()

    # Histograma de resíduos
    fig3 = px.histogram(df_plot, x="Resíduo", nbins=40, title=f"{titulo} — Histograma dos resíduos")
    fig3.update_layout(template="plotly_white")
    fig3.show()

print("\n=== GRÁFICOS — MÚLTIPLO (NORMAL) ===")
graficos_diagnostico(y_test, pred_base_sm, "Múltiplo (Normal)")

print("\n=== GRÁFICOS — REDUZIDO (NORMAL) ===")
graficos_diagnostico(y_test, pred_red_sm, "Reduzido (Normal)")

print("\n=== GRÁFICOS — REDUZIDO (LOG y) ===")
graficos_diagnostico(y_test, pred_log_real, "Reduzido (LOG y)")
