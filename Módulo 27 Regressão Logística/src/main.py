from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

"""
REGRESSÃO LOGÍSTICA UTILIZA A FUNÇÃO SIGMOIDE PARA GERAR UMA PROBABILIDADE ENTRE 0 E 1 TENDO COMO BASE VARIÁVEIS INDEPENDENTES.
DEPENDENDO DO VALOR DA PROBABILIDADE O MODELO CLASSIFICA COMO 0(FRACASSO) OU 1(SUCESSO).
REGRESSÃO LINEAR É DIFERENTE DA LOGÍSTICA.
LINEAR É MODELO DE REGRESSÃO, SAÍDA É UMA VARIÁVEL CONTÍNUA.
LOGÍSTICA É MODELO DE CLASSIFICAÇÃO, SAÍDA É UMA VARIÁVEL CATEGÓRICA.
AMBAS ANALISAM AS VARIÁVEIS INDEPENDENTES PARA ENTENDER A VARIÁVEL DEPENDENTE.
"""

def load_data():
    """
    EN: Load datasets from /data (project root is one level above /src).
    PT: Carrega os datasets da pasta /data (raiz do projeto fica 1 nivel acima de /src).
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    df = pd.read_csv(data_dir / "CARDIO_BASE.csv", sep=";", decimal=",", encoding="utf-8")

    return df

def remove_outliers_iqr(df, colunas, fator=1.5):
    df_limpo = df.copy()

    for col in colunas:
        Q1 = df_limpo[col].quantile(0.25)
        Q3 = df_limpo[col].quantile(0.75)
        IQR = Q3 - Q1

        limite_baixo = Q1 - fator * IQR
        limite_alto = Q3 + fator * IQR

        mask_outlier = (df_limpo[col] < limite_baixo) | (df_limpo[col] > limite_alto)
        qtd_outliers = int(mask_outlier.sum())
        total_antes = df_limpo.shape[0]

        # agora sim filtra removendo outliers dessa coluna
        df_limpo = df_limpo[~mask_outlier]

        print(f"\nCOLUNA: {col}")
        print(f"Limite inferior: {limite_baixo}")
        print(f"Limite superior: {limite_alto}")
        print(f"Outliers na coluna: {qtd_outliers} ({qtd_outliers/total_antes:.2%})")
        print(f"Linhas restantes após remover: {df_limpo.shape[0]}")

    return df_limpo

def remove_outliers_fisiologicos(df, limites):
    """
    limites: dicionário no formato:
    {
        "height": (140, 210),
        "weight": (40, 200)
    }
    """

    df_limpo = df.copy()
    total_inicial = df_limpo.shape[0]

    for col, (min_val, max_val) in limites.items():
        mask_outlier = (df_limpo[col] < min_val) | (df_limpo[col] > max_val)
        qtd_outliers = int(mask_outlier.sum())
        total_antes = df_limpo.shape[0]

        df_limpo = df_limpo[~mask_outlier]

        print(f"\nCOLUNA: {col}")
        print(f"Limite fisiológico: {min_val} - {max_val}")
        print(f"Outliers removidos: {qtd_outliers} ({qtd_outliers / total_antes:.2%})")
        print(f"Linhas restantes: {df_limpo.shape[0]}")

    print("\nTotal removido no processo:",
          total_inicial - df_limpo.shape[0])

    return df_limpo

def main():
    """
    EN: Load data.
    PT: Carregar dados.
    """
    df = load_data()

    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["height", "weight"])

    print('\n==== VERIFICAÇÃO DE DADOS: ====\n')
    print(df.head().to_string())

    print('\n==== QUANTIDADE DE DADOS: ====\n')
    print(df.shape)

    print('\n==== TIPOS DE DADOS: ====\n')
    print(df.dtypes)

    print('\n==== VERIFICAÇÃO DE VALORES NULOS: ====\n')
    print(df.isnull().sum())

    print('\n==== ANÁLISE DE DADOS ÚNICOS: ====\n')
    print(df.nunique())



    print('\n==== VERIFICAÇÃO DOS OUTLIERS COM IQR: ====')
    colunas_numericas = df.select_dtypes(include=['int64','float64']).columns
    colunas_numericas = [c for c in colunas_numericas if df[c].nunique() > 3]
    df_sem_outliers = remove_outliers_iqr(df, colunas_numericas)
    print('\nDATAFRAME COM OUTLIERS:')
    print(df.describe())
    print("\nDATAFRAME SEM OUTLIERS (IQR):")
    print(df_sem_outliers.describe())

    print('\n==== VERIFICAÇÃO DOS OUTLIERS COM LIMITES FISIOLÓGICOS: ====')
    limites = {
        "height": (140, 210),
        "weight": (40, 200)
    }

    df_sem_outliers_lim = remove_outliers_fisiologicos(df, limites)
    print("\nDATAFRAME SEM OUTLIERS UTILIZANDO LIMITES NATURAIS:")
    print(df_sem_outliers_lim.describe())

    #GRÁFICOS
    fig = px.box(df, y="height", points="outliers", title="Boxplot - Altura")
    fig.show()

    fig = px.box(df, y="weight", points="outliers", title="Boxplot - Peso")
    fig.show()

    fig = px.histogram(df, x="height", nbins=50, title="Distribuição de Altura")
    fig.show()

    fig = px.histogram(df, x="weight", nbins=50, title="Distribuição de Peso")
    fig.show()

    fig = px.scatter(
        df,
        x="height",
        y="weight",
        opacity=0.5,
        trendline="ols",
        title="Altura x Peso com tendência"
    )
    fig.show()

    fig = px.scatter(
        df_sem_outliers_lim,
        x="height",
        y="weight",
        opacity=0.5,
        trendline="ols",
        title="Altura x Peso com tendência sem outliers"
    )
    fig.show()

    #correlação
    corr = df_sem_outliers_lim.corr(numeric_only=True)

    fig = px.imshow(corr, text_auto=True, title="Heatmap de correlação (numérico)")
    fig.show()

    #parcats
    df_viz = df_sem_outliers_lim.copy()

    # Binning das numéricas
    df_viz["age_bin"] = pd.qcut(df_viz["age"], q=3, duplicates="drop").astype(str)
    df_viz["height_bin"] = pd.qcut(df_viz["height"], q=3, duplicates="drop").astype(str)
    df_viz["weight_bin"] = pd.qcut(df_viz["weight"], q=3, duplicates="drop").astype(str)

    # Mapear categorias
    df_viz["gender"] = df_viz["gender"].map({1: "Homem", 2: "Mulher"})
    df_viz["smoke"] = df_viz["smoke"].map({0: "Não", 1: "Sim"})
    df_viz["alco"] = df_viz["alco"].map({0: "Não", 1: "Sim"})
    df_viz["active"] = df_viz["active"].map({0: "Não", 1: "Sim"})
    df_viz["cardio_disease"] = df_viz["cardio_disease"].map({0: "Não", 1: "Sim"})

    nivel = {1: "Normal", 2: "Acima", 3: "Muito acima"}
    df_viz["gluc"] = df_viz["gluc"].map(nivel)
    df_viz["cholesterol"] = df_viz["cholesterol"].map(nivel)

    df_viz["cardio_label"] = df_viz["cardio_disease"].map({0: "Não", 1: "Sim"})

    dimensoes = [
        "age_bin",
        "gender",
        "smoke",
        "alco",
        "active",
        "height_bin",
        "weight_bin",
        "cholesterol",
        "gluc",
        "cardio_disease"
    ]

    fig = px.parallel_categories(
        df_viz,
        dimensions=dimensoes,
        color=df_sem_outliers_lim["cardio_disease"],
        title="Parcats - Todas as Dimensões"
    )

    fig.show()

    """
    IQR PODE ELIMINAR PESSOAS MUITO ALTAS OU MUITO MAGRAS QUE SÃO REAIS.
    NECESSIDADE DE IMPOR LIMITES FISIOLÓGICOS.
    COLUNA ALTURA E PESO COM VALORES SEM SENTIDO.
    A IDADE MÍNIMA É DE 30 ANOS E O PESO MÍNIMO É DE 30KG, POSSIVELMENTE TEMOS PESOS ERRADOS.
    A IDADE MÍNIMA É DE 30 ANOS E A ALTURA MÍNIMA É DE 70 CM E A ALTURA MÁXIMA É DE 250 CM.
    A ALTURA MÉDIA DE UMA PESSOA COM NANISMO VARIA DE 120 CM A 132 CM.
    CONSIDERANDO QUE A FORMA EXTREMAMENTE RARA DE NANISMO, PRIMORDIAL, TEM UMA INCIDÊNCIA 
    DE 1 EM 1 MILHÃO DE NASCIMENTOS, PODE-SE CONCLUIR REGISTROS ERRADOS.
    OUTLIERS EXCLUÍDOS COM BASE EM LIMITES FISIOLÓGICOS.
    CORRELAÇÃO NÃO SIGNIFICA CAUSALIDADE, REGRESSÃO LOGÍSTICA É O TESTE REAL.
    ALTURA TEVE CORRELAÇÃO NEGATIVA, IMC DIMINUI COM O AUMENTO DA ALTURA.
    ATIVIDADE FÍSICA TEVE CORRELAÇÃO NEGATIVA.
    FUMAR E BEBER TEVE CORRELAÇÃO NEGATIVA, PROVAVELMENTE SÃO DE PESSOAS MAIS JOVENS E A IDADE
    TEM UMA CORRELAÇÃO FORTE COM A TARGET. LOGO, JOVENS QUE FUMAM E BEBE TEM O RISCO BAIXO DE DOENÇAS CARDIOVASCULARES.
    IDADE, PESO E COLESTEROL POSSUEM CORRELAÇÕES FORTES COM A TARGET.
    PARCATS MOSTRA ASSOCIAÇÃO MAS NÃO É CAUSA.
    PESSOAS COM MAIS DE 50 ANOS POSSUEM MAIS CHANCES DE DOENÇAS CARDIOVASCULARES.
    GÊNERO ESTÁ DESBALANCEADO E CADA SEXO TEM 50% DE CHANCES.
    FUMAR E BEBER SÃO IRRELEVANTES NO PARCATS.
    FAZER ATIVIDADE NÃO DIMINUI A CHANCE DE DOENÇAS CARDIACAS E PESSOAS COM MAIS DE 50 ANOS POSSUEM O RISCO MESMO FAZENDO EXERCÍCIOS.
    EM CADA GRUPO DE ALTURA TEMOS 50%.
    GANHO DE PESO É UM FATOR QUE AUMENTA AS CHANCES.
    COLESTEROL ALTO E MUITO ALTO POSSUEM MAIS INCIDÊNCIAS, PORÉM COLESTEROL NORMAL NÃO IMPEDE DE TER DOENÇAS DEVIDO A OUTROS FATORES.
    MESMA ANÁLISE PARA OS NÍVEIS DE GLICOSE.
    """

    #MODELO

    """
    A FORMA PADRÃO DA FUNÇÃO SIGMOIDE USADA PARA CALCULAR A PROBABILIDADE DE UM EVENTO BINÁRIO POSSUI NA FÓRMULA O COEFICIENTE BETA.
    NA FÓRMULA DA REGRESSÃO LOGÍSTICA SEM FAZER AS SIMPLIFICAÇÕES QUE LEVAM À FUNÇÃO SIGMOIDE, CADA COEFICIENTE BETA MEDE QUANTO O LOG-RISCO
    MUDA QUANDO X AUMENTA UMA UNIDADE.
    LOG(P/1-P) = BetaZero + (BetaUm x X1) + (BetaDois x X2) + ...
    SEM A PADRONIZAÇÃO, CADA VARIÁVEL UTILIZADA NO MODELO VAI TER UM COEFICIENTE BASEADO EM SUA PRÓPRIA UNIDADE DE MEDIDA.
    O COEFICIENTE MOSTRA O QUANTO UMA VARIÁVEL INFLUÊNCIA MAIS QUE A OUTRA.
    DESTA FORMA, O MODELO FICA ENVIESADO PORQUE VARIÁVEIS COM ESCALA MAIORES SÃO PENALIZADAS.
    UTILIZANDO O ODDS-RATIO FICA NÍTIDO O VIÉS GERADO PELA FALTA DE PADRONIZAÇÃO.
    O LOG-RISCO SERIA DIFERENTE SEM A PADRONIZAÇÃO.
    MESMO QUE OS DADOS DA BASE DE TREINO NÃO ESTEJAM DIVIDIDOS IGUAIS, ISSO NÃO SIGNIFICA DESBALANCEAMENTO.
    BALANCEAMENTO SOMENTE NA VARIÁVEL TARGET, PORÉM NÃO TEM NECESSIDADE NESTA BASE DE DADOS.
    FEATURES NÃO PRECISAM DE BALANCEAMENTO, POIS MOSTRAM A REALIDADE POPULACIONAL.
    TEMOS POUCOS FUMANTES, SE FOSSE BALANCEADO O MODELO ENTENDERIA QUE NO MUNDO EXISTE A MESMA QUANTIDADE DE FUMANTES E NÃO FUMANTES,
    CRIANDO UMA POPULAÇÃO ARTIFICIAL.
    """

    # separar a target para evitar data leakage, treino e teste
    #x_raw é sem a padronização

    print("\n==== CONFERÊNCIA DOS DADOS ====\n")

    colunas = ['gender' , 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio_disease' ]

    for c in colunas:
        print(f"\n📊 Coluna: {c}")
        print(df_sem_outliers_lim[c].value_counts(dropna=False).to_frame(name='Contagem'))

    #sem padronizar
    x_raw = df_sem_outliers_lim.drop("cardio_disease", axis=1)
    y = df_sem_outliers_lim["cardio_disease"]

    x_train_r, x_test_r, y_train, y_test = train_test_split(
        x_raw, y, test_size=0.25, random_state=42, stratify=y
    )

    print('\n==== CONFERÊNCIA DAS BASES DE TREINO E TESTE SEM PADRONIZAÇÃO: ====')
    print('\n Tamanho do x_train:', x_train_r.shape)
    print('\n Tamanho do y_train:', y_train.shape)
    print('\n Tamanho do x_test:', x_test_r.shape)
    print('\n Tamanho do y_test:', y_test.shape)

    #padronizado
    x_scaled = x_raw.copy()
    cols_continuas = ["age", "height", "weight"]

    x_train_s, x_test_s, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()

    x_train_s[cols_continuas] = scaler.fit_transform(
        x_train_s[cols_continuas]
    )

    x_test_s[cols_continuas] = scaler.transform(
        x_test_s[cols_continuas]
    )

    print('\n==== CONFERÊNCIA DAS BASES DE TREINO E TESTE PADRONIZADAS: ====')
    print('\n Tamanho do x_train:', x_train_s.shape)
    print('\n Tamanho do y_train:', y_train.shape)
    print('\n Tamanho do x_test:', x_test_s.shape)
    print('\n Tamanho do y_test:', y_test.shape)

    #treinamento do modelo sem padronizar
    model_raw = LogisticRegression(max_iter=1000)
    model_raw.fit(x_train_r, y_train)

    odds_raw = pd.DataFrame({
        "Variavel": x_raw.columns,
        "OddsRatio_Sem_Padronizacao": np.exp(model_raw.coef_[0])
    })

    #treinamento do modelo padronizado
    model_scaled = LogisticRegression(max_iter=1000)
    model_scaled.fit(x_train_s, y_train)

    odds_scaled = pd.DataFrame({
        "Variavel": x_scaled.columns,
        "OddsRatio_Padronizado": np.exp(model_scaled.coef_[0])
    })

    comparacao_odds = odds_raw.merge(
        odds_scaled,
        on="Variavel"
    )

    print("\n==== COMPARAÇÃO ODDS RATIO ====\n")
    print(comparacao_odds.sort_values(
        "OddsRatio_Padronizado",
        ascending=False
    ))

    # analise do treino
    #statsmodels
    x_sm = x_train_s.copy()
    x_sm = sm.add_constant(x_sm)

    logit_model = sm.Logit(y_train, x_sm)
    result = logit_model.fit()

    print('\n==== TREINO ====\n')
    print('RESULTADOS DA REGRESSÃO LOGÍSTICA (STATSMODELS):')
    print('\nParâmetros do modelo:\n')
    print(result.summary())

    #Odds Ratio e intervalo de confiança
    params = result.params
    conf = result.conf_int()

    odds = np.exp(params)
    conf_odds = np.exp(conf)

    odds_table = pd.DataFrame({
        "OddsRatio": odds,
        "IC_2.5%": conf_odds[0],
        "IC_97.5%": conf_odds[1]
    })

    print('\nODDS RATIO + INTERVALO DE CONFIANÇA:\n')
    print(odds_table)

    #sklearn
    print('\nRESULTADOS DA REGRESSÃO LOGÍSTICA (SKLEARN):')
    print("\nIntercept:", model_scaled.intercept_)

    coef_table = pd.DataFrame({
        "Variavel": x_train_s.columns,
        "Coeficiente": model_scaled.coef_[0],
        "OddsRatio": np.exp(model_scaled.coef_[0])
    })

    print('\nTABELA COEFICIENTE: \n')
    print(coef_table.sort_values("OddsRatio", ascending=False))

    #teste sklearn
    y_pred = model_scaled.predict(x_test_s)

    print('\n==== TESTE ====\n')
    print('RESULTADOS DA REGRESSÃO LOGÍSTICA (SKLEARN):')
    print("\nParâmetros do modelo:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    #roc
    y_prob_train = model_scaled.predict_proba(x_train_s)[:, 1]
    y_prob_test = model_scaled.predict_proba(x_test_s)[:, 1]

    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_prob_train)
    fpr_te, tpr_te, _ = roc_curve(y_test, y_prob_test)

    auc_tr = roc_auc_score(y_train, y_prob_train)
    auc_te = roc_auc_score(y_test, y_prob_test)

    print("AUC Treino:", auc_tr)
    print("AUC Teste:", auc_te)


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr_tr, y=tpr_tr,
        name=f"Treino (AUC={auc_tr:.3f})"
    ))

    fig.add_trace(go.Scatter(
        x=fpr_te, y=tpr_te,
        name=f"Teste (AUC={auc_te:.3f})"
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash="dash"),
        name="Aleatório"
    ))

    fig.update_layout(
        title="ROC Curve - Treino vs Teste",
        xaxis_title="FPR",
        yaxis_title="TPR"
    )

    fig.show()

    #Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print('\nMATRIZ DE CONFUSÃO: \n')
    print(cm)

    labels = ["Sem Doença", "Com Doença"]

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues"
    )

    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predito",
        yaxis_title="Real"
    )

    fig.show()

    """
    ANÁLISE DOS RESULTADOS...
    O ODDS RATIO INDICA O AUMENTO NO RISCO.
    AGE FOI 1,56 SIGNIFICA QUE O AUMENTO DE UM DESVIO PADRÃO AUMENTA O RISCO EM 56%.
    CHOLESTEROL FOI DE 1,84 DEMONSTRANDO QUE O AUMENTO DO NÍVEL AUMENTA O RISCO EM 84%
    VALORES COM ODDS RATIO MENORES DO QUE UM INDICAM REDUÇAÕ.
    HEIGHT FOI 0,93 INDICANDO QUE QUANTO MAIOR A ALTURA O RISCO DIMINUI EM 7%.
    GLUC (GLICOSE) FOI 0,89. VALOR ESTRANHO, POSSIBILIDADE DE COLINEARIDADE COM OUTRA VARIÁVEL.
    
    PRECISION MEDE A CONFIABILIDADE DAS PREVISÕES POSITIVAS, DO CONJUNTO DE PREVISÕES POSITIVAS FEITA PELO MODELO QUANTAS SÃO VERDADEIRAS.
    RECALL MEDE A CAPACIDADE DE CLASSIFICAR OS POSITIVOS VERDADEIROS, DO CONJUNTO DE POSITIVOS REAIS QUANTO O MODELO CLASSIFICOU CORRETAMENTE.
    ALTA PRECISÃO E BAIXO RECALL INDICA CAUTELA EM PREVER POSITIVO (SOMENTE COM MUITA CERTEZA), PORÉM MUITOS CASOS REAIS SÃO PERDIDOS.
    BAIXA PRECISÃO E ALTO RECALL INDICA TER MENOS CAUTELA EM PREVER COMO POSITIVO, PORÉM MUITOS FALSOS POSITIVOS SÃO GERADOS.
    CONSIDERANDO QUE SÃO DADOS DE SAÚDE, É MELHOR FALAR QUE ALGUÉM SAUDÁVEL TEM UMA DOENÇA QUE FALAR QUE UM DOENTE É SAUDÁVEL.
    TER UM FALSO NEGATIVO É PIOR.
    RECALL ANTES DE OTIMIZAR DEIXAVA ESCAPAR 37% DOS DOENTES E APÓS A OTIMIZAÇÃO PASSOU A DEIXAR SOMENTE 10%.
    RECALL SAIU DE 63% PARA 90%, PORÉM O CUSTO FOI O AUMENTO DE FALSOS POSITIVOS, ANTES ERA 420 E DEPOIS 888.
    
    AUC NO TREINO E NO TESTE FORAM PARECIDOS, MODELO GENERALIZA BEM E NÃO TEM OVERFITTING. TEM UMA CAPACIDADE DE DISTINGUIR ENTRE AS CLASSES DE 69%.
    GRÁFICO AUC-ROC, LINHA ROC MOSTRA A TAXA DE VERDADEIROS POSITIVOS(TPR) E A TAXA DE FALSOS POSITIVOS(FPR)
    
    TARGET cardio_disease - tem doença cardio (1) não tem (0) 
    MATRIZ DE CONFUSÃO:
    TN FP
    FN TP
    
    REGRESSÃO LOGÍSTICA RETORNA PROBABILIDADE, POR PADRÃO O SKLEARN USA THRESHOLD = 0,5.
    P >= 0,5 (CLASSE 1)
    P <= 0,5 (CLASSE 0)
    THRESHOLD OTIMIZADO PARA 0,34, ISSO NÃO SIGNIFICA ALTERAÇÃO NO TESTE E SIM NA CLASSIFICAÇÃO.
    A REGRESSÃO LOGÍSTICA CONTINUA CALCULANDO A PROBABILIDADE, O QUE MUDA É O VALOR QUE DEFINE A CLASSIFICAÇÃO.
    """

    #ALTERANDO THRESHOLD PARA MELHORAR O MODELO
    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_prob_test
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=precision[:-1],
        name="Precision"
    ))

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=recall[:-1],
        name="Recall"
    ))

    fig.update_layout(
        title="Precision vs Recall por Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score"
    )

    fig.show()

    #threshold ótimo encontrado automaticamente
    #maximizar f1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print('\n==== MODELO OTIMIZADO ====:')
    print('\nTHRESHOLD AUTOMÁTICO: \n')
    print("Melhor threshold:", best_threshold)

    #modelo otimizado
    y_pred_opt = (y_prob_test >= best_threshold).astype(int)

    print("\nParâmetros do modelo otimizado:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred_opt))
    print("Precision:", precision_score(y_test, y_pred_opt))
    print("Recall:", recall_score(y_test, y_pred_opt))
    print("F1:", f1_score(y_test, y_pred_opt))

    cm_opt = confusion_matrix(y_test, y_pred_opt)
    print('\nMATRIZ DE CONFUSÃO MODELO OTIMIZADO PELO THRESHOLD ÓTIMO: \n')
    print(cm_opt)

    labels = ["Sem Doença", "Com Doença"]

    fig = ff.create_annotated_heatmap(
        z=cm_opt,
        x=labels,
        y=labels,
        colorscale="Blues"
    )

    fig.update_layout(
        title="Matriz de Confusão - Threshold ótimo",
        xaxis_title="Predito",
        yaxis_title="Real"
    )

    fig.show()

    """
    VIF COM VALORES: 1 - NENHUMA CORRELAÇÃO / 1 ATÉ 2 - MÍNIMO / 2 ATÉ 5 - MODERADO / >5 - PROBLEMA / >10 - GRAVE
    MODELO COM TODOS OS VIF COM VALORES ABAIXO DE DOIS, PORTANTO NÃO EXISTE MULTICOLINEARIDADE RELEVANTE.
    
    """

    #VIF
    # base de features do treino (sem target)
    X_vif = x_train_s.copy()

    # statsmodels exige constante para alguns diagnósticos
    X_vif_const = sm.add_constant(X_vif)

    vif_data = pd.DataFrame()
    vif_data["Variavel"] = X_vif_const.columns

    vif_data["VIF"] = [
        variance_inflation_factor(X_vif_const.values, i)
        for i in range(X_vif_const.shape[1])
    ]

    print("\n==== VIF (Multicolinearidade) ====\n")
    vif_data = vif_data.sort_values("VIF", ascending=False)
    print(vif_data)

    #FATORES DE RISCO
    ranking_risco = coef_table.copy()

    ranking_risco["Impacto"] = abs(ranking_risco["Coeficiente"])

    ranking_risco = ranking_risco.sort_values(
        "Impacto",
        ascending=False
    )

    print("\n==== RANKING DE FATORES DE RISCO ====\n")
    print(ranking_risco)

    fig = px.bar(
        ranking_risco,
        x="OddsRatio",
        y="Variavel",
        orientation="h",
        title="Ranking de Fatores de Risco Cardiovascular"
    )

    fig.show()

if __name__ == "__main__":
    main()