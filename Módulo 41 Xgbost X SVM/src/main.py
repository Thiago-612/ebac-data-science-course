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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from xgboost import XGBClassifier

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
    Carrega o dataset em csv.

    Returns
    -------
    pd.DataFrame
        DataFrame original carregado.
    """

    _, data_dir, _ = get_project_paths()

    df = pd.read_csv(
        data_dir / "train.csv",
        sep=",",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"],
        #index_col=0
    )

    df_test = pd.read_csv(
        data_dir / "test.csv",
        sep=",",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"],
        #index_col=0
    )

    return df, df_test


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



def encode_categorical_columns(
    df: pd.DataFrame,
    cols: list
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica Label Encoding em colunas categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    cols : list
        Lista de colunas categóricas para encoding.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame transformado e dicionário com encoders.
    """

    df = df.copy()

    encoders = {}

    for col in cols:
        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

        print(f"\nEncoding aplicado na coluna: {col}")
        print(dict(zip(le.classes_, le.transform(le.classes_))))

    return df, encoders


def train_xgboost_baseline(X_train, y_train):
    """
    Treina um modelo XGBoost baseline com preprocessing.
    """

    preprocessor = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),

        ("classifier", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    return model


def evaluate_classifier(model, X_test, y_test, model_name):
    """
    Avalia um modelo de classificação binária.
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


def get_xgboost_feature_importance(model):
    """
    Retorna a importância das features após preprocessing.
    """

    # Features transformadas pelo preprocessor
    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    # Importâncias do XGBoost
    importances = (
        model.named_steps["classifier"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df


def run_xgboost_grid_search(X_train, y_train, cv=5):
    """
    Executa GridSearchCV para XGBoost.
    """

    preprocessor = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),

        ("classifier", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])

    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [2, 3],
        "classifier__learning_rate": [0.05, 0.1]
    }

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


def run_cross_validation(model, X, y, cv=5):
    """
    Executa validação cruzada estratificada.
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


def run_xgboost_experiment(
    X,
    y,
    feature_set_name,
    test_size=0.2,
    grid_cv=5,
    final_cv=5
):
    """
    Executa experimento completo com XGBoost.

    Train Test Split:
     - Baseline (All Features/Top Features)
     - GridSearch (All Features/Top Features)
    Cross Validation:
     - Baseline (All Features/Top Features)
     - GridSearch (All Features/Top Features)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Baseline
    baseline_model = train_xgboost_baseline(X_train, y_train)

    baseline_result = evaluate_classifier(
        baseline_model,
        X_test,
        y_test,
        model_name=f"XGBoost Baseline - Train Test Split - {feature_set_name}"
    )

    # GridSearch
    grid = run_xgboost_grid_search(
        X_train,
        y_train,
        cv=grid_cv
    )

    grid_model = grid.best_estimator_

    grid_result = evaluate_classifier(
        grid_model,
        X_test,
        y_test,
        model_name=f"XGBoost GridSearch - Train Test Split - {feature_set_name}"
    )

    print(f"\n==== MELHORES PARÂMETROS - {feature_set_name} ====\n")
    print(grid.best_params_)

    # Cross Validation baseline
    cv_baseline = run_cross_validation(
        baseline_model,
        X,
        y,
        cv=final_cv
    )

    cv_baseline.insert(0, "model", f"XGBoost Baseline - Cross Validation - {feature_set_name}")

    # Cross Validation grid
    cv_grid = run_cross_validation(
        grid_model,
        X,
        y,
        cv=final_cv
    )

    cv_grid.insert(0, "model", f"XGBoost GridSearch - Cross Validation - {feature_set_name}")

    test_results = pd.DataFrame([
        baseline_result,
        grid_result
    ])

    cv_results = pd.concat(
        [cv_baseline, cv_grid],
        ignore_index=True
    )

    return {
        "baseline_model": baseline_model,
        "grid_model": grid_model,
        "grid_best_params": grid.best_params_,
        "test_results": test_results,
        "cv_results": cv_results,
        "X_test": X_test,
        "y_test": y_test
    }


def create_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features para o dataset Titanic.
    """

    df = df.copy()

    # =====================================================
    # FAMILY SIZE
    # =====================================================

    df["FamilySize"] = (
        df["SibSp"] +
        df["Parch"] +
        1
    )

    # =====================================================
    # AGE BINS
    # =====================================================

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[
            "Child",
            "Teen",
            "YoungAdult",
            "Adult",
            "Senior"
        ]
    )

    # =====================================================
    # FARE PER PERSON
    # =====================================================

    df["FarePerPerson"] = (
            df["Fare"] / df["FamilySize"]
    )

    return df


def full_xgboost_feature_workflow(
    df,
    target="Survived",
    n_top_features=2,
    test_size=0.2,
    grid_cv=5,
    final_cv=5
):
    """
    Executa dois blocos de comparação:

     1. XGBoost com todas as features
     2. XGBoost com top features por importância
    """

    start_time = time.time()

    _, _, outputs_dir = get_project_paths()

    X = df.drop(columns=[target])
    y = df[target]

    # =====================================================
    # BLOCO 1 - TODAS AS FEATURES
    # =====================================================

    print("\n==== BLOCO 1: XGBOOST COM TODAS AS FEATURES ====\n")

    all_features_results = run_xgboost_experiment(
        X=X,
        y=y,
        feature_set_name="All Features",
        test_size=test_size,
        grid_cv=grid_cv,
        final_cv=final_cv
    )

    # Feature importance usando baseline com todas as features
    importance_df = get_xgboost_feature_importance(
        all_features_results["baseline_model"]
    )

    print("\n==== FEATURE IMPORTANCE - XGBOOST ====\n")
    print(importance_df)

    # =====================================================
    # COMPARAÇÃO FINAL
    # =====================================================

    test_comparison_df = all_features_results["test_results"]

    cv_comparison_df = all_features_results["cv_results"]

    test_comparison_display = test_comparison_df.copy()
    test_comparison_display["evaluation_type"] = "Train Test Split"

    cv_comparison_display = cv_comparison_df.copy()
    cv_comparison_display["evaluation_type"] = "Cross Validation"

    cv_comparison_display = cv_comparison_display.rename(columns={
        "accuracy_mean": "accuracy",
        "precision_mean": "precision",
        "recall_mean": "recall",
        "f1_score_mean": "f1_score",
        "roc_auc_mean": "roc_auc"
    })

    cols = [
        "model",
        "evaluation_type",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    full_comparison_df = pd.concat(
        [
            test_comparison_display[cols],
            cv_comparison_display[cols]
        ],
        ignore_index=True
    )

    print("\n==== COMPARAÇÃO GERAL: TRAIN/TEST VS CROSS VALIDATION ====\n")
    print(full_comparison_df.round(4).to_string(index=False))

    # Melhor modelo pelo F1 médio da validação cruzada
    best_cv = cv_comparison_df.sort_values(
        by="f1_score_mean",
        ascending=False
    ).iloc[0]

    best_model_name = best_cv["model"]

    if "Baseline" in best_model_name:
        best_model = all_features_results["baseline_model"]
    else:
        best_model = all_features_results["grid_model"]

    best_X_test = all_features_results["X_test"]
    best_y_test = all_features_results["y_test"]

    print("\n==== MELHOR RESULTADO FINAL ====")
    print("UTILIZANDO SOMENTE CROSS VALIDATION\n")
    print(best_cv)

    # Salvar arquivos
    test_comparison_df.to_csv(
        outputs_dir / "xgboost_test_comparison.csv",
        index=False
    )

    cv_comparison_df.to_csv(
        outputs_dir / "xgboost_cv_comparison.csv",
        index=False
    )

    full_comparison_df.to_csv(
        outputs_dir / "xgboost_full_comparison.csv",
        index=False
    )

    importance_df.to_csv(
        outputs_dir / "xgboost_feature_importance.csv",
        index=False
    )

    elapsed_time = time.time() - start_time

    print("\n==== ARQUIVOS SALVOS ====")
    print(outputs_dir / "xgboost_test_comparison.csv")
    print(outputs_dir / "xgboost_cv_comparison.csv")
    print(outputs_dir / "xgboost_full_comparison.csv")
    print(outputs_dir / "xgboost_feature_importance.csv")
    print(f"\nTempo total: {elapsed_time / 60:.2f} minutos")

    return {
        "all_features_results": all_features_results,
        "feature_importance": importance_df,
        "test_comparison": test_comparison_df,
        "cv_comparison": cv_comparison_df,
        "best_result": best_cv,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_X_test": best_X_test,
        "best_y_test": best_y_test
    }


def show_predictions_table(
    model,
    X_test,
    y_test,
    threshold: float = 0.5,
    n_rows: int = 20
) -> pd.DataFrame:
    """
    Mostra tabela com:
    - classe real
    - probabilidade prevista
    - previsão binária

    Parameters
    ----------
    model
        Modelo treinado.

    X_test
        Features de teste.

    y_test
        Target real.

    threshold : float
        Limite para converter probabilidade em classe binária.

    n_rows : int
        Quantidade de linhas exibidas.
    """

    # Probabilidade da classe 1
    y_proba = model.predict_proba(X_test)[:, 1]

    # Conversão para binário
    y_pred = (y_proba >= threshold).astype(int)

    results_df = pd.DataFrame({
        "real_class": y_test.values,
        "predicted_probability": y_proba,
        "binary_prediction": y_pred
    })

    results_df["predicted_probability"] = (
        results_df["predicted_probability"]
        .round(4)
    )

    print("\n==== PREVISÕES DO MODELO ====\n")
    print(results_df.head(n_rows).to_string(index=False))

    return results_df


def main() -> None:
    """
    Função principal do projeto.

    Fluxo:
    1.
    """

    target = "Survived"

    df, df_test = load_data()

    test_passenger_ids = df_test["PassengerId"]

    df["HasCabin"] = df["Cabin"].notnull().astype(int)

    df_test["HasCabin"] = df_test["Cabin"].notnull().astype(int)

    df["Title"] = (
        df["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
    )

    rare_titles = [
        "Lady", "Countess", "Capt", "Col",
        "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer", "Dona"
    ]

    df["Title"] = df["Title"].replace(
        rare_titles,
        "Rare"
    )

    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    df = clean_columns(
        df,
        to_lower=False,
        remove_special=False,
        remove_unnamed=True,
        drop_cols= ["PassengerId", "Ticket", "Cabin", "Name"]
    )

    df_test = clean_columns(
        df_test,
        to_lower=False,
        remove_special=False,
        remove_unnamed=True,
        drop_cols= ["PassengerId", "Ticket", "Cabin", "Name"]
    )

    df = create_titanic_features(df)

    df_test = create_titanic_features(df_test)

    print("\n==== DATASET INICIAL ====\n")
    describe_data(df, target=target)
    analyze_by_class(df, target=target)

    # plot_eda_plotly(
    #     df=df,
    #     target=target,
    #     max_scatter_cols=6,
    #     nbins=30,
    #     sample_size=2000,
    #     show_hist=True,
    #     show_box=True,
    #     show_scatter_matrix=True,
    #     show_corr=True
    # )
    #
    # scatter_pairs(
    #     df=df,
    #     target=target,
    #     sample_size=2000,
    #     max_cols=6,
    #     save_html=False
    # )

    results = full_xgboost_feature_workflow(
        df=df,
        target="Survived",
        n_top_features=7,
        test_size=0.2,
        grid_cv=10,
        final_cv=10
    )

    prediction_table = show_predictions_table(
        model=results["best_model"],
        X_test=results["best_X_test"],
        y_test=results["best_y_test"],
        threshold=0.5,
        n_rows=20
    )

    #KAGGLE
    best_model = results["best_model"]

    X_full = df.drop(columns=[target])
    y_full = df[target]

    X_kaggle_test = df_test.copy()

    best_model.fit(X_full, y_full)

    test_predictions = best_model.predict(X_kaggle_test)

    submission = pd.DataFrame({
        "PassengerId": test_passenger_ids,
        "Survived": test_predictions
    })

    _, _, outputs_dir = get_project_paths()

    submission.to_csv(
        outputs_dir / "submission.csv",
        index=False
    )

if __name__ == "__main__":
    main()
