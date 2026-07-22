import pandas as pd
from scipy.stats import ttest_ind, t, norm
import plotly.graph_objects as go
import numpy as np

"""
Análise:
Hipótese nula (H0): A média das notas dos alunos na estratégia A é igual à média das notas dos alunos na estratégia B.
Hipótese alternativa (H1): A média das notas na Estratégia B é maior do que a média das notas na Estratégia A.

Se a hipótese nula é somente igual, então a hipótese alternativa deve ser diferente. 
Teste bicaudal.
Intervalo de confiança de 95% e nível de significância de 5%.
Zona de rejeição de 2,5% em cada cauda.
"""

# Parâmetros populacionais
media_estrategia_A = 70
desvio_padrao_estrategia_A = 10

media_estrategia_B = 75
desvio_padrao_estrategia_B = 12

# Gerando as amostras de notas para cada estratégia de ensino da nossa base
np.random.seed(0)  # Para garantir a reprodutibilidade dos resultados
amostra_estrategia_A = np.random.normal(loc=media_estrategia_A, scale=desvio_padrao_estrategia_A, size=50)
amostra_estrategia_B = np.random.normal(loc=media_estrategia_B, scale=desvio_padrao_estrategia_B, size=50)

print('\nNOTAS: \n')
print("Notas da amostra da Estratégia A:", amostra_estrategia_A[:5])
print("Notas da amostra da Estratégia B:", amostra_estrategia_B[:5])

# Estimadores amostrais

media_amostra_A = np.mean(amostra_estrategia_A)
variancia_amostra_A = np.var(amostra_estrategia_A, ddof=1)  # ddof=1 para variância amostral
DesvioPadrao_amostra_A = variancia_amostra_A ** 0.5
media_amostra_B = np.mean(amostra_estrategia_B)
variancia_amostra_B = np.var(amostra_estrategia_B, ddof=1)
DesvioPadrao_amostra_B = variancia_amostra_B ** 0.5

# Tabela comparativa lado a lado
tabela_comparacao = pd.DataFrame({
    "Métrica": [
        "Média populacional",
        "Desvio padrão populacional",
        "Média amostral",
        "Variância amostral",
        "Desvio padrão amostral"
    ],
    "Estratégia A": [
        media_estrategia_A,
        desvio_padrao_estrategia_A,
        media_amostra_A,
        variancia_amostra_A,
        DesvioPadrao_amostra_A
    ],
    "Estratégia B": [
        media_estrategia_B,
        desvio_padrao_estrategia_B,
        media_amostra_B,
        variancia_amostra_B,
        DesvioPadrao_amostra_B
    ]
})

print("\nCOMPARAÇÃO ENTRE PARÂMETROS POPULACIONAIS E ESTIMADORES AMOSTRAIS\n")
print(tabela_comparacao.to_string(index=False))

"""
Análise: 
As duas populações são parecidas e suas respectivas amostras possuem estimadores parecidos com os parâmetros populacionais e
parecidos entre si.
"""

# Hipóteses:
# H0: mu_A = mu_B
# H1: mu_A != mu_B   (bicaudal)

alpha = 0.05

# Dados da amostra
nA = len(amostra_estrategia_A)
nB = len(amostra_estrategia_B)

xA = np.mean(amostra_estrategia_A)
xB = np.mean(amostra_estrategia_B)

# Parâmetros populacionais conhecidos (sigma)
sigma_A = desvio_padrao_estrategia_A
sigma_B = desvio_padrao_estrategia_B

# Erro-padrão da diferença de médias (com sigmas conhecidos)
se = np.sqrt((sigma_A**2)/nA + (sigma_B**2)/nB)

# Estatística Z (diferença esperada sob H0 = 0)
z = (xA - xB) / se

# p-value bicaudal
p_value = 2 * (1 - norm.cdf(abs(z)))

# Região crítica (equivalente pela abordagem do valor crítico)
z_crit = norm.ppf(1 - alpha/2)

print("\n====================")
print("TESTE Z (bicaudal) — 2 médias independentes (σ conhecidos)")
print("====================")
print(f"nA={nA}, nB={nB}")
print(f"Média A = {xA:.4f}")
print(f"Média B = {xB:.4f}")
print(f"Diferença (A - B) = {xA - xB:.4f}")
print(f"Z = {z:.4f}")
print(f"p-value = {p_value:.6f}")
print(f"z crítico (±) = {z_crit:.4f}")

# Decisão
if p_value < alpha:
    print("\n❌ Rejeitamos H0 (p < 0.05): há diferença estatisticamente significativa entre as médias.")
else:
    print("\n✅ Não rejeitamos H0 (p ≥ 0.05): não há evidência suficiente de diferença entre as médias.")

# Parâmetros do teste
alpha = 0.05
z_obs = z  # estatística Z calculada anteriormente

# Eixo X da normal padrão
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)

# Valores críticos
z_crit = norm.ppf(1 - alpha / 2)

# Criando o gráfico
fig = go.Figure()

# Curva normal padrão
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='lines',
    name='Distribuição Normal Padrão',
    line=dict(color='black')
))

# Região crítica esquerda
fig.add_trace(go.Scatter(
    x=x[x <= -z_crit],
    y=y[x <= -z_crit],
    fill='tozeroy',
    name='Região crítica (α/2)',
    fillcolor='rgba(255,0,0,0.3)',
    line=dict(color='rgba(255,0,0,0)')
))

# Região crítica direita
fig.add_trace(go.Scatter(
    x=x[x >= z_crit],
    y=y[x >= z_crit],
    fill='tozeroy',
    name='Região crítica (α/2)',
    fillcolor='rgba(255,0,0,0.3)',
    line=dict(color='rgba(255,0,0,0)')
))

# Linha do Z observado
fig.add_vline(
    x=z_obs,
    line=dict(color='blue', width=3),
    annotation_text=f"Z observado = {z_obs:.2f}",
    annotation_position="top"
)

# Layout
fig.update_layout(
    title="Teste Z bicaudal — Comparação entre Estratégias A e B",
    xaxis_title="Z",
    yaxis_title="Densidade de probabilidade",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5
    )
)

fig.show()


"""
Análise:
Teste de hipótese de amostras independentes, usando a diferença entre as médias e não os valores separados.
Z padronizado não está na região de rejeição, -1,96<Z>1,96.
P-value maior que o nível de significância.
Não rejeitamos a H0 por esses motivos.
As duas estratégias são eficazes, pois não há evidência de diferença entre as médias.
"""

#MELHORIAS PROPOSTAS PELO TUTOR
"""
Você utilizou o Teste Z porque conhecia os desvios padrões populacionais (sigma). Contudo, em cenários reais, 
raramente conhecemos o sigma. Uma dica é praticar o uso do Teste T (t-test) através da função scipy.stats.ttest_ind, 
que é mais robusto quando trabalhamos apenas com dados amostrais e tamanhos de amostra menores.
"""

# ====================
# TESTE t (bicaudal) — 2 amostras independentes (sigma desconhecido)
# ====================

alpha = 0.05

# t-test de Welch (mais robusto: não assume variâncias iguais)
t_stat, p_value_t = ttest_ind(amostra_estrategia_A, amostra_estrategia_B, equal_var=False)

# Estatísticas
nA = len(amostra_estrategia_A)
nB = len(amostra_estrategia_B)

xA = np.mean(amostra_estrategia_A)
xB = np.mean(amostra_estrategia_B)

sA = np.std(amostra_estrategia_A, ddof=1)
sB = np.std(amostra_estrategia_B, ddof=1)

# Welch-Satterthwaite df (graus de liberdade aproximados)
se_welch = np.sqrt((sA**2)/nA + (sB**2)/nB)
df_welch = ( (sA**2/nA + sB**2/nB)**2 ) / ( ((sA**2/nA)**2)/(nA-1) + ((sB**2/nB)**2)/(nB-1) )

# valor crítico t (bicaudal)
t_crit = t.ppf(1 - alpha/2, df_welch)

# IC 95% para (A - B)
diff = xA - xB
ic_inf = diff - t_crit * se_welch
ic_sup = diff + t_crit * se_welch

print("\n====================")
print("TESTE t (Welch) bicaudal — 2 médias independentes (σ desconhecido)")
print("====================")
print(f"nA={nA}, nB={nB}")
print(f"Média A = {xA:.4f} | Desvio amostral A = {sA:.4f}")
print(f"Média B = {xB:.4f} | Desvio amostral B = {sB:.4f}")
print(f"Diferença (A - B) = {diff:.4f}")
print(f"t = {t_stat:.4f}")
print(f"df (Welch) ≈ {df_welch:.2f}")
print(f"p-value = {p_value_t:.6f}")
print(f"t crítico (±) = {t_crit:.4f}")
print(f"IC 95% (A - B) = [{ic_inf:.4f}, {ic_sup:.4f}]")

# Decisão
if p_value_t < alpha:
    print("\n❌ Rejeitamos H0 (p < 0.05): há diferença estatisticamente significativa entre as médias.")
else:
    print("\n✅ Não rejeitamos H0 (p ≥ 0.05): não há evidência suficiente de diferença entre as médias.")

# Criando o gráfico
# eixo x baseado na t com df_welch
x = np.linspace(-5, 5, 1200)
y = t.pdf(x, df_welch)

fig = go.Figure()

# curva t
fig.add_trace(go.Scatter(
    x=x, y=y, mode="lines",
    name=f"t(df≈{df_welch:.1f})",
    line=dict(color="black")
))

# região crítica esquerda
fig.add_trace(go.Scatter(
    x=x[x <= -t_crit],
    y=y[x <= -t_crit],
    fill="tozeroy",
    name="Região crítica (α/2)",
    fillcolor="rgba(255,0,0,0.25)",
    line=dict(color="rgba(255,0,0,0)")
))

# região crítica direita
fig.add_trace(go.Scatter(
    x=x[x >= t_crit],
    y=y[x >= t_crit],
    fill="tozeroy",
    name="Região crítica (α/2)",
    fillcolor="rgba(255,0,0,0.25)",
    line=dict(color="rgba(255,0,0,0)")
))

# linha do t observado
fig.add_vline(
    x=t_stat,
    line=dict(color="blue", width=3),
    annotation_text=f"t observado = {t_stat:.2f}",
    annotation_position="top"
)

# layout
fig.update_layout(
    title="Teste t bicaudal (Welch) — Estratégia A vs B",
    xaxis_title="t",
    yaxis_title="Densidade",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
)

fig.show()
