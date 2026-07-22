from __future__ import annotations

from pathlib import Path
import time
import warnings
import re

import pandas as pd
import numpy as np

import itertools
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.float_format", "{:.4f}".format)


# =========================================================
# PATHS
# =========================================================

def get_project_paths() -> tuple[Path, Path, Path]:
    """
    Retorna os caminhos principais do projeto.

    Returns
    -------
    tuple[Path, Path, Path]
        project_root : pasta raiz do projeto
        data_dir : pasta onde ficam os datasets
        outputs_dir : pasta onde serão salvos os resultados
    """

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    return project_root, data_dir, outputs_dir


# =========================================================
# LOAD
# =========================================================

def load_data() -> pd.DataFrame:
    """
    Carrega o dataset marketing_campaign.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame original carregado.
    """

    _, data_dir, _ = get_project_paths()

    df = pd.read_csv(
        data_dir / "marketing_campaign.csv",
        sep=";",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"],
        #index_col=0
    )

    return df


# =========================================================
# CLEAN COLUMNS
# =========================================================

def clean_columns(
    df: pd.DataFrame,
    drop_cols: list | None = None,
    to_lower: bool = False,
    remove_special: bool = False,
    remove_unnamed: bool = True
) -> pd.DataFrame:
    """
    Padroniza nomes das colunas e remove colunas indesejadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    drop_cols : list | None
        Lista de colunas a remover.

    to_lower : bool
        Se True, converte nomes das colunas para minúsculo.

    remove_special : bool
        Se True, remove caracteres especiais.

    remove_unnamed : bool
        Se True, remove colunas com nome começando por 'Unnamed'.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas tratadas.
    """

    df = df.copy()

    new_cols = []

    for col in df.columns:
        col = col.strip()

        if to_lower:
            col = col.lower()

        col = col.replace(" ", "_")

        if remove_special:
            col = re.sub(r"[^\w]", "", col)

        new_cols.append(col)

    df.columns = new_cols

    if remove_unnamed:
        df = df.loc[:, ~df.columns.str.contains("^Unnamed|^unnamed", regex=True)]

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    return df


# =========================================================
# EDA
# =========================================================

def describe_data(df: pd.DataFrame, target: str) -> None:
    """
    Exibe uma visão geral do dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame analisado.

    target : str
        Nome da variável alvo.
    """

    print("\n==== VISÃO GERAL DOS DADOS ====\n")
    print(df.head().to_string())

    print("\n==== SHAPE ====\n")
    print(df.shape)

    print("\n==== TIPOS DE DADOS ====\n")
    print(df.dtypes)

    print("\n==== VALORES NULOS ====\n")
    print(df.isnull().sum())

    print("\n==== ANÁLISE DESCRITIVA ====\n")
    print(df.describe().to_string())

    if target in df.columns:
        print("\n==== DISTRIBUIÇÃO DO TARGET ====\n")
        print(df[target].value_counts())
        print("\nProporção:")
        print(df[target].value_counts(normalize=True))


def analyze_by_class(df: pd.DataFrame, target: str) -> None:
    """
    Analisa a separação das variáveis em relação ao target.

    Para variáveis numéricas contínuas, cria faixas com pd.cut.
    Para variáveis com poucas categorias, usa crosstab diretamente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame analisado.

    target : str
        Nome da variável alvo.
    """

    print("\n==== SEPARAÇÃO DAS CLASSES ====\n")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if target in numeric_cols:
        numeric_features = [col for col in numeric_cols if col != target]

        print("\nMédia por classe:\n")
        print(df.groupby(target)[numeric_features].mean())

    print("\nDistribuição por classe:\n")

    for col in df.columns:
        if col == target:
            continue

        print(f"\n==== {col} por classe ====\n")

        if df[col].nunique() <= 10:
            print(pd.crosstab(df[col], df[target], normalize="columns"))
        elif pd.api.types.is_numeric_dtype(df[col]):
            print(pd.crosstab(
                pd.cut(df[col], bins=5, duplicates="drop"),
                df[target],
                normalize="columns"
            ))
        else:
            print(pd.crosstab(df[col], df[target], normalize="columns"))


# =========================================================
# PREPROCESSING
# =========================================================

def get_feature_types(X: pd.DataFrame) -> tuple[list, list]:
    """
    Separa colunas numéricas e categóricas.

    Parameters
    ----------
    X : pd.DataFrame
        Features do modelo.

    Returns
    -------
    tuple[list, list]
        Lista de colunas numéricas e lista de colunas categóricas.
    """

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Cria o pré-processador do modelo.

    Etapas:
    - numéricas: imputação pela mediana + padronização
    - categóricas: imputação pela moda + One-Hot Encoding

    Parameters
    ----------
    X : pd.DataFrame
        Features do modelo.

    Returns
    -------
    ColumnTransformer
        Transformador pronto para ser usado no Pipeline.
    """

    numeric_features, categorical_features = get_feature_types(X)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# =========================================================
# MODELS
# =========================================================

def build_model_pipeline(
    X: pd.DataFrame,
    model_type: str,
    use_pca: bool = False,
    pca_components: float | int = 0.95
) -> Pipeline:
    """
    Cria um Pipeline completo de modelagem.

    Parameters
    ----------
    X : pd.DataFrame
        Features usadas para identificar tipos de colunas.

    model_type : str
        Tipo de modelo:
        - "logistic"
        - "random_forest"

    use_pca : bool
        Se True, adiciona PCA após o pré-processamento.

    pca_components : float | int
        Número de componentes do PCA.
        Se float entre 0 e 1, representa a variância explicada desejada.
        Ex: 0.95 mantém 95% da variância.

    Returns
    -------
    Pipeline
        Pipeline completo.
    """

    preprocessor = build_preprocessor(X)

    steps = [
        ("preprocessor", preprocessor)
    ]

    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components)))

    if model_type == "logistic":
        model = LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("model_type deve ser 'logistic' ou 'random_forest'.")

    steps.append(("classifier", model))

    return Pipeline(steps=steps)


# =========================================================
# EVALUATION
# =========================================================

def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
    """
    Avalia um modelo de classificação binária.

    Parameters
    ----------
    model
        Modelo treinado.

    X_test
        Features de teste.

    y_test
        Target de teste.

    model_name : str
        Nome do modelo para exibição.

    Returns
    -------
    dict
        Métricas do modelo.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def run_cross_validation(model, X, y, cv: int = 5) -> pd.DataFrame:
    """
    Executa validação cruzada estratificada com múltiplas métricas.

    Parameters
    ----------
    model
        Modelo ou Pipeline.

    X
        Features.

    y
        Target.

    cv : int
        Número de folds.

    Returns
    -------
    pd.DataFrame
        Tabela com média e desvio padrão das métricas.
    """

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1_score": "f1",
        "roc_auc": "roc_auc"
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1
    )

    results = {}

    for metric in scoring.keys():
        results[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
        results[f"{metric}_std"] = scores[f"test_{metric}"].std()

    return pd.DataFrame([results])


# =========================================================
# GRID SEARCH
# =========================================================

def get_param_grid(model_name: str) -> dict:
    """
    Retorna o grid de hiperparâmetros conforme o modelo vencedor.

    Parameters
    ----------
    model_name : str
        Nome do modelo vencedor.

    Returns
    -------
    dict
        Grade de hiperparâmetros para o GridSearchCV.
    """

    if "Logistic" in model_name:
        param_grid = {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"]
        }

    else:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features": ["sqrt", "log2"]
        }

    if "PCA" in model_name:
        param_grid["pca__n_components"] = [0.90, 0.95, 0.99]

    return param_grid


def run_grid_search(
    model,
    model_name: str,
    X_train,
    y_train,
    cv: int = 5
) -> GridSearchCV:
    """
    Executa GridSearchCV no melhor modelo inicial.

    Parameters
    ----------
    model
        Pipeline do modelo vencedor.

    model_name : str
        Nome do modelo vencedor.

    X_train
        Features de treino.

    y_train
        Target de treino.

    cv : int
        Número de folds do GridSearch.

    Returns
    -------
    GridSearchCV
        Objeto GridSearch treinado.
    """

    param_grid = get_param_grid(model_name)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    return grid


# =========================================================
# PCA REPORT
# =========================================================

def get_pca_report(pca_pipeline: Pipeline) -> pd.DataFrame:
    """
    Extrai a variância explicada de um Pipeline com PCA.

    Parameters
    ----------
    pca_pipeline : Pipeline
        Pipeline treinado contendo a etapa 'pca'.

    Returns
    -------
    pd.DataFrame
        Tabela com variância explicada por componente.
    """

    pca = pca_pipeline.named_steps["pca"]

    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()

    pca_df = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(explained))],
        "explained_variance": explained,
        "cumulative_variance": cumulative
    })

    return pca_df


def plot_eda_plotly(
    df: pd.DataFrame,
    target: str | None = None,
    max_scatter_cols: int = 6,
    nbins: int = 30,
    sample_size: int = 2000,
    show_hist: bool = True,
    show_box: bool = True,
    show_scatter_matrix: bool = True,
    show_corr: bool = True
) -> None:
    """
    Gera gráficos de EDA com Plotly para variáveis numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    target : str | None
        Nome da variável alvo. Se informado, será removido dos gráficos numéricos
        e usado como cor no scatter matrix.

    max_scatter_cols : int
        Número máximo de variáveis numéricas no scatter matrix.

    nbins : int
        Número de bins dos histogramas.

    sample_size : int
        Tamanho da amostra usada no scatter matrix para evitar lentidão.

    show_hist, show_box, show_scatter_matrix, show_corr : bool
        Controlam quais gráficos serão exibidos.
    """

    num_df = df.select_dtypes(include="number").copy()

    if num_df.empty:
        raise ValueError("O DataFrame não possui colunas numéricas.")

    numeric_cols = num_df.columns.tolist()

    if target in numeric_cols:
        feature_cols = [col for col in numeric_cols if col != target]
    else:
        feature_cols = numeric_cols

    print("\n==== COLUNAS NUMÉRICAS IDENTIFICADAS ====\n")
    print(numeric_cols)

    if show_hist:
        print("\nGerando histogramas...")
        for col in feature_cols:
            fig = px.histogram(
                df,
                x=col,
                color=target if target else None,
                nbins=nbins,
                marginal="box",
                title=f"Distribuição - {col}"
            )
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Frequência",
                bargap=0.05
            )
            fig.show()

    if show_box:
        print("\nGerando boxplots...")
        for col in feature_cols:
            fig = px.box(
                df,
                x=target if target else None,
                y=col,
                title=f"Boxplot - {col}"
            )
            fig.update_layout(yaxis_title=col)
            fig.show()

    if show_scatter_matrix:
        scatter_cols = feature_cols[:max_scatter_cols]

        if len(scatter_cols) >= 2:
            print("\nGerando scatter matrix...")

            df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

            fig = px.scatter_matrix(
                df_sample,
                dimensions=scatter_cols,
                color=target if target else None,
                title="Scatter Matrix"
            )

            fig.update_traces(
                diagonal_visible=True,
                showupperhalf=False
            )

            fig.update_layout(
                height=900,
                width=900
            )

            fig.show()
        else:
            print("\nScatter matrix ignorado: menos de 2 colunas numéricas.")

    if show_corr:
        print("\nGerando heatmap de correlação...")

        corr = num_df.corr(numeric_only=True)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                text=corr.round(2).values,
                texttemplate="%{text}",
                hovertemplate="X: %{x}<br>Y: %{y}<br>Correlação: %{z:.2f}<extra></extra>"
            )
        )

        fig.update_layout(
            title="Matriz de Correlação",
            xaxis_title="Variáveis",
            yaxis_title="Variáveis",
            height=700,
            width=900
        )

        fig.show()


def scatter_pairs(
    df: pd.DataFrame,
    cols: list | None = None,
    target: str | None = None,
    sample_size: int = 2000,
    max_cols: int | None = None,
    save_html: bool = False
) -> None:
    """
    Gera gráficos scatter separados para pares de variáveis numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    cols : list | None
        Lista de colunas numéricas que serão combinadas duas a duas.
        Se None, usa todas as colunas numéricas, exceto o target.

    target : str | None
        Nome da variável alvo usada como cor.

    sample_size : int
        Tamanho da amostra usada nos gráficos.

    max_cols : int | None
        Limita a quantidade de colunas usadas.

    save_html : bool
        Se True, salva cada gráfico como HTML na pasta atual.
    """

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if cols is None:
        cols = numeric_cols

    if target in cols:
        cols = [col for col in cols if col != target]

    if max_cols:
        cols = cols[:max_cols]

    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    for x, y in itertools.combinations(cols, 2):
        fig = px.scatter(
            df_sample,
            x=x,
            y=y,
            color=target if target else None,
            title=f"{x} vs {y}",
            opacity=0.6
        )

        fig.show()

        if save_html:
            safe_x = str(x).replace("/", "_").replace(" ", "_")
            safe_y = str(y).replace("/", "_").replace(" ", "_")
            fig.write_html(f"scatter_{safe_x}_vs_{safe_y}.html")


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
    normalize: bool = False
) -> None:
    """
    Gera matriz de confusão com Plotly.

    Parameters
    ----------
    model
        Modelo treinado (Pipeline ou sklearn).

    X_test
        Features de teste.

    y_test
        Target real.

    model_name : str
        Nome do modelo (para título do gráfico).

    normalize : bool
        Se True, mostra proporções ao invés de valores absolutos.
    """

    # Previsões
    y_pred = model.predict(X_test)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = f"Matriz de Confusão (Normalizada) - {model_name}"
    else:
        title = f"Matriz de Confusão - {model_name}"

    # Labels
    labels = ["0", "1"]

    # Plot
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {l}" for l in labels],
            y=[f"Real {l}" for l in labels],
            text=np.round(cm, 3),
            texttemplate="%{text}",
            colorscale="Blues"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predição",
        yaxis_title="Real",
        width=600,
        height=600
    )

    fig.show()


# =========================================================
# FULL WORKFLOW
# =========================================================

def full_model_comparison_workflow(
    df: pd.DataFrame,
    target: str = "WebPurchases",
    test_size: float = 0.2,
    pca_components: float | int = 0.95,
    grid_cv: int = 5,
    final_cv: int = 5
) -> dict:
    """
    Executa o fluxo completo do projeto.

    Etapas:
    1. separa X e y
    2. divide treino e teste
    3. treina Logistic Regression sem PCA
    4. treina Random Forest sem PCA
    5. treina Logistic Regression com PCA
    6. treina Random Forest com PCA
    7. compara os quatro modelos
    8. executa GridSearch no melhor modelo
    9. compara baseline vs GridSearch
    10. executa validação cruzada no melhor modelo final
    11. salva os resultados em CSV

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.

    target : str
        Nome da variável alvo.

    test_size : float
        Percentual da base usado para teste.

    pca_components : float | int
        Número de componentes do PCA ou percentual de variância.

    grid_cv : int
        Número de folds do GridSearchCV.

    final_cv : int
        Número de folds da validação cruzada final.

    Returns
    -------
    dict
        Resultados principais do fluxo.
    """

    start_time = time.time()

    _, _, outputs_dir = get_project_paths()

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression - Original": build_model_pipeline(
            X_train,
            model_type="logistic",
            use_pca=False
        ),
        "Random Forest - Original": build_model_pipeline(
            X_train,
            model_type="random_forest",
            use_pca=False
        ),
        "Logistic Regression - PCA": build_model_pipeline(
            X_train,
            model_type="logistic",
            use_pca=True,
            pca_components=pca_components
        ),
        "Random Forest - PCA": build_model_pipeline(
            X_train,
            model_type="random_forest",
            use_pca=True,
            pca_components=pca_components
        )
    }

    initial_results = []
    trained_models = {}

    print("\n==== TREINANDO MODELOS INICIAIS ====\n")

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        result = evaluate_classifier(
            model,
            X_test,
            y_test,
            model_name=model_name
        )

        initial_results.append(result)
        trained_models[model_name] = model

    initial_comparison_df = pd.DataFrame(initial_results)

    print("\n==== COMPARAÇÃO INICIAL ====\n")
    print(initial_comparison_df.to_string(index=False))

    best_initial = initial_comparison_df.sort_values(
        by="f1_score",
        ascending=False
    ).iloc[0]

    best_initial_name = best_initial["model"]
    best_initial_model = trained_models[best_initial_name]

    print("\n==== MELHOR MODELO INICIAL ====\n")
    print(best_initial_name)

    if "PCA" in best_initial_name:
        pca_report = get_pca_report(best_initial_model)

        print("\n==== RELATÓRIO PCA DO MELHOR MODELO ====\n")
        print(pca_report.to_string(index=False))

        pca_report.to_csv(
            outputs_dir / "pca_report_best_model.csv",
            index=False
        )
    else:
        pca_report = pd.DataFrame()

    print("\n==== GRIDSEARCH NO MELHOR MODELO ====\n")

    grid = run_grid_search(
        model=best_initial_model,
        model_name=best_initial_name,
        X_train=X_train,
        y_train=y_train,
        cv=grid_cv
    )

    grid_model = grid.best_estimator_

    print("\n==== MELHORES PARÂMETROS ====\n")
    print(grid.best_params_)

    baseline_result = evaluate_classifier(
        best_initial_model,
        X_test,
        y_test,
        model_name=f"Baseline - {best_initial_name}"
    )

    grid_result = evaluate_classifier(
        grid_model,
        X_test,
        y_test,
        model_name=f"GridSearch - {best_initial_name}"
    )

    final_comparison_df = pd.DataFrame([
        baseline_result,
        grid_result
    ])

    print("\n==== COMPARAÇÃO FINAL: BASELINE VS GRIDSEARCH ====\n")
    print(final_comparison_df.to_string(index=False))

    best_final = final_comparison_df.sort_values(
        by="f1_score",
        ascending=False
    ).iloc[0]

    if "GridSearch" in best_final["model"]:
        final_model = grid_model
        final_model_name = best_final["model"]
    else:
        final_model = best_initial_model
        final_model_name = best_final["model"]

    print("\n==== MELHOR MODELO FINAL ====\n")
    print(final_model_name)

    cv_df = run_cross_validation(
        final_model,
        X,
        y,
        cv=final_cv
    )

    cv_df.insert(0, "model", final_model_name)

    print("\n==== CROSS VALIDATION FINAL ====\n")
    print(cv_df.round(4).to_string(index=False))

    initial_comparison_df.to_csv(
        outputs_dir / "initial_model_comparison.csv",
        index=False
    )

    final_comparison_df.to_csv(
        outputs_dir / "final_model_comparison.csv",
        index=False
    )

    cv_df.to_csv(
        outputs_dir / "cross_validation_results.csv",
        index=False
    )

    elapsed_time = time.time() - start_time

    print("\n==== ARQUIVOS SALVOS ====")
    print(outputs_dir / "initial_model_comparison.csv")
    print(outputs_dir / "final_model_comparison.csv")
    print(outputs_dir / "cross_validation_results.csv")

    if not pca_report.empty:
        print(outputs_dir / "pca_report_best_model.csv")

    print(f"\nTempo total: {elapsed_time / 60:.2f} minutos")

    return {
        "initial_comparison": initial_comparison_df,
        "final_comparison": final_comparison_df,
        "cross_validation": cv_df,
        "pca_report": pca_report,
        "best_model": final_model,
        "best_model_name": final_model_name,
        "X_test": X_test,
        "y_test": y_test
    }


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """
    Função principal do projeto.

    Fluxo:
    1. carrega dados
    2. limpa nomes das colunas
    3. executa EDA inicial
    4. executa comparação de modelos com e sem PCA
    """

    target = "WebPurchases"

    df = load_data()

    df = clean_columns(
        df,
        to_lower=False,
        remove_special=False,
        remove_unnamed=True
    )

    print("\n==== DATASET INICIAL ====\n")
    describe_data(df, target=target)
    analyze_by_class(df, target=target)

    plot_eda_plotly(
        df=df,
        target=target,
        max_scatter_cols=6,
        nbins=30,
        sample_size=2000,
        show_hist=True,
        show_box=True,
        show_scatter_matrix=True,
        show_corr=True
    )

    scatter_pairs(
        df=df,
        target=target,
        sample_size=2000,
        max_cols=6,
        save_html=False
    )

    results = full_model_comparison_workflow(
        df=df,
        target=target,
        test_size=0.2,
        pca_components=0.95,
        grid_cv=5,
        final_cv=5
    )

    plot_confusion_matrix(
        model=results["best_model"],
        X_test=results["X_test"],
        y_test=results["y_test"],
        model_name=results["best_model_name"]
    )


if __name__ == "__main__":
    main()