from __future__ import annotations
import warnings
from typing import Iterable
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

"""
Melhorias:
Como o K-means é sensível a valores atípicos (pois eles puxam a média dos centroides), uma prática recomendada é realizar 
uma análise de Boxplot antes de rodar o modelo. Identificar e tratar possíveis outliers nas medidas físicas dos pinguins 
pode resultar em clusters mais coesos e centroides mais representativos.
Dado que o K-means assume que os clusters são esféricos e de tamanhos similares, você poderia comparar os resultados 
com o DBSCAN ou Gaussian Mixture Models (GMM). Isso ajudaria a identificar se existem agrupamentos com formatos densos 
ou sobrepostos que o K-means pode ter dificuldade em separar perfeitamente (como a sobreposição citada entre Adelie e Chinstrap).

Conclusão:

2) É possível identificar dois agrupamentos no gráfico scatter matrix com dados padronizados.
Comparando com os dados originais, percebe-se que a espécie Adelie e Chinstrap são semelhantes.

O modelo com o número de clusters predefinido classificou 15% do Adelie como Chinstrap e 7% do Chinstrap como Adelie.
Nos gráficos podemos perceber similaridades entre as duas espécies.
Adelie e Chinstrap pertencem ao mesmo gênero (Pygoscelis), são parecidos em tamanho e estrutura e as diferenças são visuais.
Por isso, o modelo que utiliza o K ótimo identifica somente dois clusters.

6) Algoritmos de clusterização podem ser usados em segmentação de clientes para gerar perfis de clientes, podem ser usados para organizar documentos em temas,
podem ser usados para detectar anomalias em transações bancárias para evitar fraudes.

"""

def load_penguins_data() -> pd.DataFrame:
    """
    Carrega a base penguins do seaborn.
    """
    df = sns.load_dataset("penguins")
    return df


def describe_data(df: pd.DataFrame) -> None:
    """
    Exibe visão geral do dataset.
    """
    print("\n==== VISÃO GERAL DOS DADOS ====\n")
    print(df.head().to_string())

    print("\n==== SHAPE ====\n")
    print(df.shape)

    print("\n==== TIPOS DE DADOS ====\n")
    print(df.dtypes)

    print("\n==== VALORES NULOS ====\n")
    print(df.isnull().sum())

    print("\n==== ANÁLISE DOS DADOS ====\n")
    print(df.describe().to_string())


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove valores faltantes e mantém a coluna species para comparação,
    além das colunas numéricas usadas no K-means.
    """
    selected_cols = [
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    df_model = df[selected_cols].dropna().reset_index(drop=True)
    return df_model


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna apenas as variáveis numéricas usadas pelo modelo.
    """
    feature_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    return df[feature_cols].copy()


def plot_scatter_matrix(df: pd.DataFrame, title: str, color: str | None = None) -> None:
    """
    Equivalente ao pairplot usando Plotly.
    """
    feature_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    fig = px.scatter_matrix(
        df,
        dimensions=feature_cols,
        color=color,
        title=title,
        opacity=0.75,
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(width=1000, height=900)
    fig.show()


def scale_data(df_features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Padroniza os dados para média 0 e desvio padrão 1.
    Utiliza o dataframe sem a coluna species.
    Usado posteriormente para gerar o modelo.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    return X_scaled, scaler


def plot_scaled_scatter_matrix(X_scaled: np.ndarray, columns: list[str], title: str) -> None:
    """
    Mostra matriz de dispersão dos dados padronizados.
    """
    df_scaled = pd.DataFrame(X_scaled, columns=columns)

    fig = px.scatter_matrix(
        df_scaled,
        dimensions=columns,
        title=title,
        opacity=0.75,
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(width=1000, height=900)
    fig.show()


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    """
    Ajusta o modelo K-means.
    """
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=random_state,
    )
    model.fit(X_scaled)
    return model


def evaluate_kmeans_range(
    X_scaled: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Avalia vários valores de K usando:
    - Inércia (cotovelo) - Compactação do cluster - menor valor - Problemático, pois sempre vai diminuir ao aumentar a quantidade de clusters.
    - Silhouette Score - Separação entre os clusters - maior valor - Validação principal
    - Calinski-Harabasz - Separação vs compactação - maior valor - Confirmação
    - Davies-Bouldin - Similaridade entre clusters - menor valor - Confirmação adicional
    """
    results = []

    for k in k_values:
        model = fit_kmeans(X_scaled, n_clusters=k, random_state=random_state)
        labels = model.labels_

        results.append(
            {
                "k": k,
                "Inércia": model.inertia_,
                "Silhouette": silhouette_score(X_scaled, labels),
                "Calinski_Harabasz": calinski_harabasz_score(X_scaled, labels),
                "Davies_Bouldin": davies_bouldin_score(X_scaled, labels),
            }
        )

    return pd.DataFrame(results)


def plot_elbow_and_validation(results: pd.DataFrame) -> None:
    """
    Plota o metodo do cotovelo e métricas de validação interna.

    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["k"],
            y=results["Inércia"],
            mode="lines+markers",
            name="Inércia (Elbow)",
        )
    )

    fig.update_layout(
        title="Método do Cotovelo",
        xaxis_title="Número de clusters (k)",
        yaxis_title="Inércia",
        width=900,
        height=500,
    )
    fig.show()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["k"],
            y=results["Silhouette"],
            mode="lines+markers",
            name="Silhouette",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["k"],
            y=results["Calinski_Harabasz"],
            mode="lines+markers",
            name="Calinski-Harabasz",
            yaxis="y2",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["k"],
            y=results["Davies_Bouldin"],
            mode="lines+markers",
            name="Davies-Bouldin",
            yaxis="y3",
        )
    )

    fig.update_layout(
        title="Validação Interna dos Clusters",
        xaxis=dict(title="Número de clusters (k)"),
        yaxis=dict(title="Silhouette"),
        yaxis2=dict(
            title="Calinski-Harabasz",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        yaxis3=dict(
            title="Davies-Bouldin",
            anchor="free",
            overlaying="y",
            side="right",
            position=0.95,
            showgrid=False,
        ),
        legend=dict(x=0.01, y=0.99),
        width=950,
        height=550,
    )
    fig.show()


def choose_best_k(results: pd.DataFrame) -> int:
    """
    Escolhe o melhor K com base no maior silhouette score.
    """
    best_k = int(results.loc[results["Silhouette"].idxmax(), "k"])
    return best_k


def add_cluster_labels(df: pd.DataFrame, labels: np.ndarray, cluster_col: str) -> pd.DataFrame:
    """
    Retorna novo DataFrame com os rótulos de cluster.
    """
    df_clustered = df.copy()
    df_clustered[cluster_col] = labels.astype(str)
    return df_clustered


def get_centroids_original_scale(
    model: KMeans,
    scaler: StandardScaler,
    columns: list[str],
) -> pd.DataFrame:
    """
    Converte centróides do espaço padronizado para a escala original.
    """
    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    return pd.DataFrame(centroids_original, columns=columns)


def plot_clusters_with_centroids(
    df_clustered: pd.DataFrame,
    centroids_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    title: str,
) -> None:
    """
    Plota dispersão dos clusters com centróides em destaque.
    """
    fig = px.scatter(
        df_clustered,
        x=x_col,
        y=y_col,
        color=cluster_col,
        title=title,
        opacity=0.75,
    )

    fig.add_trace(
        go.Scatter(
            x=centroids_df[x_col],
            y=centroids_df[y_col],
            mode="markers+text",
            text=[f"C{i}" for i in range(len(centroids_df))],
            textposition="top center",
            marker=dict(size=18, symbol="x"),
            name="Centróides",
        )
    )

    fig.update_layout(width=900, height=600)
    fig.show()


def summarize_clusters(df_clustered: pd.DataFrame, cluster_col: str) -> None:
    """
    Exibe médias numéricas e contagem por cluster.
    """
    print(f"\n==== MÉDIAS POR {cluster_col.upper()} ====\n")
    print(
        df_clustered.groupby(cluster_col)[
            ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        ]
        .mean()
        .round(2)
        .to_string()
    )

    print(f"\n==== CONTAGEM POR {cluster_col.upper()} ====\n")
    print(df_clustered[cluster_col].value_counts().sort_index().to_string())


def compare_clusters_with_species(df_clustered: pd.DataFrame, cluster_col: str) -> None:
    """
    Exibe tabela cruzada entre espécie real e cluster encontrado.
    """
    print(f"\n==== TABELA CRUZADA: SPECIES x {cluster_col.upper()} ====\n")
    print("Comparação da quantidade real com a quantidade gerada pelo modelo\n")
    crosstab_abs = pd.crosstab(df_clustered["species"], df_clustered[cluster_col])
    print(crosstab_abs.to_string())

    print(f"\n==== TABELA CRUZADA (% POR ESPÉCIE): SPECIES x {cluster_col.upper()} ====\n")
    crosstab_pct = pd.crosstab(
        df_clustered["species"],
        df_clustered[cluster_col],
        normalize="index",
    ) * 100
    print(crosstab_pct.round(2).to_string())


def plot_species_vs_clusters(
    df_clustered: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    title: str,
) -> None:
    """
    Plota gráfico comparando espécie real e cluster encontrado.
    """
    fig = px.scatter(
        df_clustered,
        x=x_col,
        y=y_col,
        color=cluster_col,
        symbol="species",
        title=title,
        opacity=0.80,
    )
    fig.update_layout(width=950, height=650)
    fig.show()


def main() -> None:
    """
    Pipeline principal do projeto.
    """
    df_raw = load_penguins_data()
    describe_data(df_raw)

    print("\n==== PREPARAÇÃO DOS DADOS ====\n")
    df_model = prepare_data(df_raw)
    print(df_model.head().to_string())
    print("\nDataFrame após remoção de nulos:", df_model.shape)

    #print("\n==== MATRIZ DE DISPERSÃO - DADOS ORIGINAIS ====\n")
    plot_scatter_matrix(
        df_model,
        title="Scatter Matrix - Penguins (Dados Originais)",
        color="species",
    )

    df_features = get_feature_matrix(df_model)

    #print("\n==== PADRONIZAÇÃO DOS DADOS ====\n")
    X_scaled, scaler = scale_data(df_features)
    plot_scaled_scatter_matrix(
        X_scaled,
        columns=df_features.columns.tolist(),
        title="Scatter Matrix - Penguins (Dados Padronizados)",
    )

    #print("\n==== K-MEANS COM 3 CLUSTERS ====\n")
    model_k3 = fit_kmeans(X_scaled, n_clusters=3)
    df_k3 = add_cluster_labels(df_model, model_k3.labels_, cluster_col="cluster_k3")
    centroids_k3 = get_centroids_original_scale(
        model_k3,
        scaler,
        df_features.columns.tolist(),
    )

    summarize_clusters(df_k3, cluster_col="cluster_k3")
    compare_clusters_with_species(df_k3, cluster_col="cluster_k3")

    plot_clusters_with_centroids(
        df_clustered=df_k3,
        centroids_df=centroids_k3,
        x_col="bill_length_mm",
        y_col="bill_depth_mm",
        cluster_col="cluster_k3",
        title="K-means (k=3) - bill_length_mm x bill_depth_mm",
    )

    plot_clusters_with_centroids(
        df_clustered=df_k3,
        centroids_df=centroids_k3,
        x_col="flipper_length_mm",
        y_col="body_mass_g",
        cluster_col="cluster_k3",
        title="K-means (k=3) - flipper_length_mm x body_mass_g",
    )

    plot_species_vs_clusters(
        df_clustered=df_k3,
        x_col="bill_length_mm",
        y_col="bill_depth_mm",
        cluster_col="cluster_k3",
        title="Espécies reais x Clusters (k=3) - bill_length_mm x bill_depth_mm",
    )

    print("\n==== MÉTODO DO COTOVELO + VALIDAÇÃO INTERNA ====\n")
    k_values = range(2, 8)
    results = evaluate_kmeans_range(X_scaled, k_values=k_values)
    print(results.round(4).to_string(index=False))
    plot_elbow_and_validation(results)

    best_k = choose_best_k(results)
    print(f"\nMelhor k pelo silhouette score: {best_k}")

    print("\n==== K-MEANS COM K ÓTIMO ====\n")
    model_best = fit_kmeans(X_scaled, n_clusters=best_k)
    df_best = add_cluster_labels(df_model, model_best.labels_, cluster_col="cluster_best")
    centroids_best = get_centroids_original_scale(
        model_best,
        scaler,
        df_features.columns.tolist(),
    )

    summarize_clusters(df_best, cluster_col="cluster_best")
    compare_clusters_with_species(df_best, cluster_col="cluster_best")

    plot_clusters_with_centroids(
        df_clustered=df_best,
        centroids_df=centroids_best,
        x_col="bill_length_mm",
        y_col="bill_depth_mm",
        cluster_col="cluster_best",
        title=f"K-means (k={best_k}) - bill_length_mm x bill_depth_mm",
    )

    plot_clusters_with_centroids(
        df_clustered=df_best,
        centroids_df=centroids_best,
        x_col="flipper_length_mm",
        y_col="body_mass_g",
        cluster_col="cluster_best",
        title=f"K-means (k={best_k}) - flipper_length_mm x body_mass_g",
    )

    plot_species_vs_clusters(
        df_clustered=df_best,
        x_col="flipper_length_mm",
        y_col="body_mass_g",
        cluster_col="cluster_best",
        title=f"Espécies reais x Clusters (k={best_k}) - flipper_length_mm x body_mass_g",
    )


if __name__ == "__main__":
    main()