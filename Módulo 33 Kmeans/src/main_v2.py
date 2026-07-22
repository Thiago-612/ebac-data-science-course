from __future__ import annotations
import warnings
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def get_project_paths() -> tuple[Path, Path, Path]:
    """
    EN: Return project root, data dir, and outputs dir.
    PT: Retorna a raiz do projeto, pasta de dados e pasta de saída.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return project_root, data_dir, outputs_dir


def load_data() -> pd.DataFrame:
    """
    EN: Load the dataset.
    PT: Carrega o dataset.
    """
    _, data_dir, _ = get_project_paths()
    df = pd.read_csv(data_dir / "Mall_Customers.csv")
    return df


def describe_data(df: pd.DataFrame) -> None:
    """
    EN: Show general dataset overview.
    PT: Exibe visão geral do dataset.
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


def prepare_data(df: pd.DataFrame):
    """
    Prepara dados contínuos para KMeans.

    Etapas:
    - remove CustomerID
    - aplica one-hot encoding em Gender
    - mantém Age, Annual Income (k$) e Spending Score (1-100) contínuos
    - padroniza todas as colunas finais

    Returns
    -------
    df_scaled : pd.DataFrame
        DataFrame final padronizado.
    legends : dict
        Dicionário de legenda das variáveis.
    scaler : StandardScaler
        Objeto scaler ajustado.
    """

    df_prep = df.copy()

    # =========================================================
    # 1. Remover CustomerID
    # =========================================================
    if "CustomerID" in df_prep.columns:
        df_prep = df_prep.drop(columns=["CustomerID"])

    # =========================================================
    # 2. One-hot encoding em Gender
    # =========================================================
    gender_dummies = pd.get_dummies(
        df_prep["Gender"],
        prefix="Gender",
        dtype=int,
        drop_first=True
    )

    # =========================================================
    # 3. Montar dataframe final contínuo
    # =========================================================
    df_final = pd.DataFrame({
        "Age": df_prep["Age"],
        "Income": df_prep["Annual Income (k$)"],
        "Score": df_prep["Spending Score (1-100)"]
    })

    df_final = pd.concat([df_final, gender_dummies], axis=1)

    # =========================================================
    # 4. Legendas
    # =========================================================
    legends = {
        "Age": "Idade do cliente",
        "Income": "Renda anual (k$)",
        "Score": "Spending Score (1-100)",
        "Gender_Male": {
            0: "Female",
            1: "Male"
        }
    }

    # =========================================================
    # 5. Padronizar
    # =========================================================
    scaler = StandardScaler()
    df_scaled_array = scaler.fit_transform(df_final)

    df_prepare = pd.DataFrame(
        df_scaled_array,
        columns=df_final.columns,
        index=df_final.index
    )

    return df_prepare, legends, scaler


def plot_scatter_matrix(df: pd.DataFrame, title: str, color: str | None = None):

    fig = px.scatter_matrix(
        df,
        dimensions=df.columns,
        color=color,
        title=title,
        opacity=0.75,
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(width=1000, height=900)
    fig.show()


def decode_scaled_data(df_scaled, scaler, legends):

    df_original = scaler.inverse_transform(df_scaled)
    df_original = pd.DataFrame(df_original, columns=df_scaled.columns)

    df_original = df_original.round(0)

    df_decoded = df_original.copy()

    for col, mapping in legends.items():
        if col in df_decoded.columns and isinstance(mapping, dict):
            df_decoded[col] = df_decoded[col].round(0).astype(int).map(mapping)

    return df_decoded


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


def plot_elbow_and_validation(
    results: pd.DataFrame,
    title_elbow: str = "Método do Cotovelo",
    title_validation: str = "Validação Interna dos Clusters"
) -> None:
    """
    Plota:
    - Método do cotovelo
    - Métricas de validação interna

    """

    # =========================
    # ELBOW
    # =========================
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
        title=title_elbow,
        xaxis_title="Número de clusters (k)",
        yaxis_title="Inércia",
        width=900,
        height=500,
    )

    fig.show()

    # =========================
    # VALIDAÇÃO
    # =========================
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
        title=title_validation,
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


def choose_best_k(
    results: pd.DataFrame,
    max_reasonable_k: int | None = 5,
    silhouette_weight: float = 0.35,
    calinski_weight: float = 0.15,
    davies_weight: float = 0.15,
    simplicity_weight: float = 0.35,
) -> tuple[int, pd.DataFrame]:
    """
    Escolhe K combinando métricas internas com penalização mais forte
    para muitos clusters.

    Escolhe K de forma mais inteligente, combinando:
    - Silhouette (maior é melhor)
    - Calinski-Harabasz (maior é melhor)
    - Davies-Bouldin (menor é melhor)
    - Simplicidade / penalização para muitos clusters
    """

    df = results.copy()

    required_cols = ["k", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em results: {missing}")

    def minmax(series: pd.Series) -> pd.Series:
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return pd.Series([1.0] * len(series), index=series.index)
        return (series - s_min) / (s_max - s_min)

    df["silhouette_norm"] = minmax(df["Silhouette"])
    df["calinski_norm"] = minmax(df["Calinski_Harabasz"])
    df["davies_norm"] = 1 - minmax(df["Davies_Bouldin"])

    k_min = df["k"].min()
    k_max = df["k"].max()

    if k_max == k_min:
        df["simplicity_norm"] = 1.0
    else:
        df["simplicity_norm"] = 1 - ((df["k"] - k_min) / (k_max - k_min))

    if max_reasonable_k is not None:
        penalty = (df["k"] - max_reasonable_k).clip(lower=0)

        if penalty.max() > 0:
            penalty = penalty / penalty.max()
            df["simplicity_norm"] = (df["simplicity_norm"] - 0.8 * penalty).clip(lower=0)

    total_weight = (
        silhouette_weight
        + calinski_weight
        + davies_weight
        + simplicity_weight
    )

    df["smart_score"] = (
        silhouette_weight * df["silhouette_norm"]
        + calinski_weight * df["calinski_norm"]
        + davies_weight * df["davies_norm"]
        + simplicity_weight * df["simplicity_norm"]
    ) / total_weight

    ranked_results = df.sort_values(
        by=["smart_score", "Silhouette"],
        ascending=[False, False]
    ).reset_index(drop=True)

    best_k = int(ranked_results.loc[0, "k"])
    return best_k, ranked_results


def add_cluster_labels(df_scaled: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df_clustered = df_scaled.copy()
    df_clustered["cluster"] = labels.astype(str)
    return df_clustered


def get_centroids_original_scale(model, scaler, columns):
    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    df_centroids = pd.DataFrame(centroids_original, columns=columns)
    df_centroids = df_centroids.round(0).astype(int)

    return df_centroids


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


def summarize_clusters(df_original, labels):
    df_analysis = df_original.copy()
    df_analysis["cluster"] = labels

    if "Gender_Male" in df_analysis.columns:
        df_analysis["Gender_Male"] = df_analysis["Gender_Male"].round().astype(int)

    print("\n==== PERFIL AVANÇADO DOS CLUSTERS ====\n")

    for cluster in sorted(df_analysis["cluster"].unique()):
        subset = df_analysis[df_analysis["cluster"] == cluster]

        print(f"\nCluster {cluster}:")
        print(f"Total: {len(subset)}")

        for col in ["Age", "Income", "Score"]:
            print(f"\n{col}:")
            print(f"  min: {subset[col].min():.1f}")
            print(f"  mean: {subset[col].mean():.1f}")
            print(f"  max: {subset[col].max():.1f}")

        if "Gender_Male" in subset.columns:
            male_pct = subset["Gender_Male"].mean() * 100
            female_pct = 100 - male_pct

            print("\nSexo:")
            print(f"  % Male: {male_pct:.1f}")
            print(f"  % Female: {female_pct:.1f}")

            if male_pct >= 60:
                dominant_gender = "Predominância Masculina"
            elif female_pct >= 60:
                dominant_gender = "Predominância Feminina"
            else:
                dominant_gender = "Misto"

            print(f"  Perfil: {dominant_gender}")


def name_clusters(df_original, labels):
    df_analysis = df_original.copy()
    df_analysis["cluster"] = labels

    cluster_names = {}

    for cluster in sorted(df_analysis["cluster"].unique()):
        subset = df_analysis[df_analysis["cluster"] == cluster]

        age = subset["Age"].mean()
        income = subset["Income"].mean()
        score = subset["Score"].mean()

        if "Gender_Male" in subset.columns:
            male_pct = subset["Gender_Male"].mean() * 100
            female_pct = 100 - male_pct

            if male_pct >= 60:
                gender_label = "Masculino"
            elif female_pct >= 60:
                gender_label = "Feminino"
            else:
                gender_label = "Misto"
        else:
            gender_label = "Sem Gênero"

        # Score
        if score > 65:
            score_label = "Alto Consumo"
        elif score > 40:
            score_label = "Consumo Médio"
        else:
            score_label = "Baixo Consumo"

        # Income
        if income > 70:
            income_label = "Alta Renda"
        elif income > 50:
            income_label = "Renda Média"
        else:
            income_label = "Baixa Renda"

        # Age
        if age < 30:
            age_label = "Jovens"
        elif age < 50:
            age_label = "Adultos"
        else:
            age_label = "Idosos"

        name = f"{age_label} | {income_label} | {score_label} | {gender_label}"
        cluster_names[cluster] = name

    return cluster_names


def plot_pca_clusters(df_scaled, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)

    df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_plot["cluster"] = labels.astype(str)

    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="cluster",
        title="Clusters (PCA 2D)",
        opacity=0.8
    )

    fig.update_layout(width=800, height=600)
    fig.show()


def plot_scatter_matrix_clusters(df_original, labels, title):
    """
    Scatter matrix em escala real com cluster como cor.
    """

    df_plot = df_original.copy()
    df_plot["cluster"] = labels.astype(str)
    df_plot["Gender"] = df_plot["Gender_Male"].round().astype(int).map({
        0: "Female",
        1: "Male"
    })

    df_plot = df_plot.drop(columns=["Gender_Male"])

    fig = px.scatter_matrix(
        df_plot,
        dimensions=["Age", "Income", "Score"],
        color="cluster",
        symbol="Gender",
        title=title,
        opacity=0.75
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(width=1000, height=900)
    fig.show()


def main() -> None:
    """
    EN: Main advanced pipeline.
    PT: Pipeline avançado principal.
    """
    _, _, outputs_dir = get_project_paths()

    df = load_data()
    describe_data(df)

    # =========================================================
    # 1. PREPARAR DADOS
    # =========================================================
    print('\n==== VERIFICAÇÃO DO DATAFRAME APÓS PREPARAÇÃO ====\n')
    df_prepare, legends, scaler = prepare_data(df)
    print(df_prepare.info())
    print('\n', df_prepare.head(20).to_string(), '\n')

    # =========================================================
    # 2. VISUALIZAÇÃO
    # =========================================================
    plot_scatter_matrix(df_prepare, title="Scatter Matrix")

    # =========================================================
    # 3. ESCOLHER K
    # =========================================================
    """
    range(2, 7) gera 6 clusters e dois são iguais.
    Função avançada está mais rigorosa, range e valor maximo de k foram reduzidos.
    O modelo com dados contínuos perfoma melhor, porém o numero de clusters continua alto.
    range(2, 6) gera 5 clusters e dois são iguais.
    range(2, 5) gera 4 clusters 
    """
    results = evaluate_kmeans_range(df_prepare.values, range(2, 5))
    print("\n==== MÉTODOS DE ESCOLHA DOS CLUSTERS ====\n")
    print("Avalia vários valores de K usando:")
    print("- Inércia (cotovelo) - Compactação do cluster - menor valor - Problemático, pois sempre vai diminuir ao aumentar a quantidade de clusters.")
    print("- Silhouette Score - Separação entre os clusters - maior valor - Validação principal")
    print("- Calinski-Harabasz - Separação vs compactação - maior valor - Confirmação")
    print("- Davies-Bouldin - Similaridade entre clusters - menor valor - Confirmação adicional")

    print("\n", results.sort_values("Silhouette", ascending=False))

    plot_elbow_and_validation(
        results,
        title_elbow="Elbow",
        title_validation="Validação"
    )

    best_k, ranked_results = choose_best_k(
        results,
        max_reasonable_k=4
    )

    print("\nRANKING INTELIGENTE DE K\n")
    print(
        ranked_results[
            [
                "k",
                "Silhouette",
                "Calinski_Harabasz",
                "Davies_Bouldin",
                "simplicity_norm",
                "smart_score",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nK sugerido: {best_k}")

    # =========================================================
    # 4. TREINAR KMEANS
    # =========================================================
    model = fit_kmeans(df_prepare.values, n_clusters=best_k)

    # =========================================================
    # 5. ADICIONAR CLUSTERS
    # =========================================================
    df_clustered = add_cluster_labels(df_prepare, model.labels_)

    # =========================================================
    # 6. DECODIFICAR PARA INTERPRETAÇÃO
    # =========================================================
    df_decoded = decode_scaled_data(
        df_prepare,
        scaler,
        legends
    )

    df_decoded["cluster"] = model.labels_

    # =========================================================
    # 7. CENTRÓIDES
    # =========================================================
    centroids = get_centroids_original_scale(
        model,
        scaler,
        df_prepare.columns
    )

    print("\n==== CENTRÓIDES ====\n")
    print(centroids)

    # =========================================================
    # 8. PERFIL DOS CLUSTERS
    # =========================================================
    df_original = pd.DataFrame(
        scaler.inverse_transform(df_prepare),
        columns=df_prepare.columns
    )
    df_original["Gender_Male"] = df_original["Gender_Male"].round().astype(int)

    summarize_clusters(df_original, model.labels_)

    # CONVERTER PARA ESCALA ORIGINAL
    df_plot = pd.DataFrame(
        scaler.inverse_transform(df_prepare),
        columns=df_prepare.columns
    )

    df_plot["cluster"] = model.labels_

    # CENTRÓIDES NA MESMA ESCALA
    centroids_plot = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=df_prepare.columns
    )

    plot_clusters_with_centroids(
        df_plot,
        centroids_plot,
        x_col="Income",
        y_col="Score",
        cluster_col="cluster",
        title="Clusters (dados reais - Income vs Score)"
    )

    cluster_names = name_clusters(df_original, model.labels_)

    print("\n==== NOMES DOS CLUSTERS ====\n")
    for k, v in cluster_names.items():
        print(f"Cluster {k}: {v}")

    plot_pca_clusters(df_prepare, model.labels_)

    plot_scatter_matrix_clusters(
        df_original,
        model.labels_,
        title="Segmentação de Clientes - Scatter Matrix com Clusters"
    )

if __name__ == "__main__":
    main()