from __future__ import annotations
from pathlib import Path
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import itertools
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


"""
As variáveis CNT e UTC foram excluídas.
CNT é um contador e pode prejudicar o modelo, não é uma característica física do ambiente.
UTC mostra o tempo em segundos, pode prejudicar o modelo.

A escolha do modelo levou em consideração o padrão dos dados, a correlação, a presença de outliers, a escala dos 
dados e o tamanho do dataset.
Os dados não possuem um padrão claro, as correlações são altas, os outliers são numerosos, cada variável possui uma escala diferente e o dataset é grande.
Desta forma, optou-se por utilizar Random Forest.
Alternativamente foi utilizado Logistic Regression para verificar se o Random Forest é o melhor modelo, pela análise dos gráficos os dados não aparentam ter
uma relação linear com a target.

Interpretação:

relação linear → LinearRegression, Ridge, Lasso
relação curva → Polynomial, árvores
sem padrão claro → modelos mais robustos (árvore, ensemble)

Correlação entre variáveis (multicolinearidade)
correlação alta entre features → modelos lineares sofrem
use:
Ridge / Lasso
ou árvores (Decision Tree, Random Forest)

Outliers
muitos outliers → regressão linear sofre
melhores opções:
árvores
robustos (RANSAC, Huber)

Escala dos dados
dados em escalas diferentes → precisa normalizar
modelos sensíveis:
Linear, Logistic, SVM, KNN
modelos que não ligam:
árvores (Decision Tree, Random Forest)

Tamanho do dataset
pequeno → modelos simples (linear, árvore rasa)
grande → pode usar modelos mais complexos
"""


warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.float_format", "{:.4f}".format)


# =========================================================
# PATHS
# =========================================================
def get_project_paths() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return project_root, data_dir, outputs_dir


# =========================================================
# LOAD
# =========================================================
def load_data() -> pd.DataFrame:
    _, data_dir, _ = get_project_paths()

    df = pd.read_csv(
        data_dir / "smoke_detection_iot.csv",
        sep=",",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"],
        index_col=0
    )
    return df


def clean_columns(
    df: pd.DataFrame,
    drop_cols: list = None,
    to_lower: bool = True,
    remove_special: bool = True
) -> pd.DataFrame:
    """
    Limpa nomes das colunas e remove colunas específicas.
    """

    df = df.copy()

    # ---------------------------------------------------
    # 1) Padronizar nomes primeiro
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # 2) Ajustar nomes das colunas a remover
    # ---------------------------------------------------
    if drop_cols:
        clean_drop_cols = []

        for col in drop_cols:
            col = col.strip()

            if to_lower:
                col = col.lower()

            col = col.replace(" ", "_")

            if remove_special:
                col = re.sub(r"[^\w]", "", col)

            clean_drop_cols.append(col)

        df = df.drop(columns=clean_drop_cols, errors="ignore")

    return df


def describe_data(df: pd.DataFrame) -> None:
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

    print("\n==== SEPARAÇÃO DAS CLASSES ====\n")
    print("Média:\n")
    print(df.groupby("Fire_Alarm").mean())
    print("\nGeral:\n")
    for col in df.columns:
        if col != "Fire_Alarm":
            print(f"\n==== {col} por classe ====\n")

            if df[col].nunique() <= 10:
                # categórica
                print(pd.crosstab(df[col], df["Fire_Alarm"], normalize="columns"))
            else:
                # contínua
                print(pd.crosstab(
                    pd.cut(df[col], bins=5),
                    df["Fire_Alarm"],
                    normalize="columns"
                ))


def plot_eda_plotly(
    df: pd.DataFrame,
    max_scatter_cols: int = 6,
    nbins: int = 30,
    show_hist: bool = True,
    show_box: bool = True,
    show_scatter_matrix: bool = True,
    show_corr: bool = True
) -> None:
    """
    Gera gráficos de análise exploratória (EDA) com Plotly para variáveis numéricas.
    """

    # Seleciona apenas colunas numéricas
    num_df = df.select_dtypes(include="number").copy()

    if num_df.empty:
        raise ValueError("O DataFrame não possui colunas numéricas para análise.")

    numeric_cols = num_df.columns.tolist()

    print("=" * 60)
    print("COLUNAS NUMÉRICAS IDENTIFICADAS:")
    print(numeric_cols)
    print("=" * 60)

    # ---------------------------------------------------
    # 1) Histogramas
    # ---------------------------------------------------
    if show_hist:
        print("\nGerando histogramas...")
        for col in numeric_cols:
            fig = px.histogram(
                num_df,
                x=col,
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

    # ---------------------------------------------------
    # 2) Boxplots
    # ---------------------------------------------------
    if show_box:
        print("\nGerando boxplots...")
        for col in numeric_cols:
            fig = px.box(
                num_df,
                y=col,
                title=f"Boxplot - {col}"
            )
            fig.update_layout(
                yaxis_title=col
            )
            fig.show()

    # ---------------------------------------------------
    # 3) Scatter Matrix
    # ---------------------------------------------------
    if show_scatter_matrix:
        scatter_cols = numeric_cols[:max_scatter_cols]

        if len(scatter_cols) >= 2:
            print("\nGerando scatter matrix...")
            fig = px.scatter_matrix(
                num_df,
                dimensions=scatter_cols,
                title="Scatter Matrix"
            )
            fig.update_traces(diagonal_visible=True, showupperhalf=False)
            fig.update_layout(
                height=900,
                width=900
            )
            fig.show()
        else:
            print("\nScatter matrix ignorado: menos de 2 colunas numéricas.")

    # ---------------------------------------------------
    # 4) Heatmap de Correlação
    # ---------------------------------------------------
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
    df,
    cols=None,
    target=None,
    sample_size=2000,
    max_cols=None,
    save_html=False
):
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

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
            fig.write_html(f"scatter_{x}_vs_{y}.html")


def get_top_corr_features(
    df: pd.DataFrame,
    target: str,
    n_features: int = 5,
    method: str = "pearson",
    absolute: bool = True
) -> pd.DataFrame:
    """
    Seleciona as variáveis numéricas mais correlacionadas com o target.
    """

    df = df.copy()

    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrado no DataFrame.")

    numeric_df = df.select_dtypes(include=np.number)

    if target not in numeric_df.columns:
        raise ValueError(f"Target '{target}' precisa ser numérico para calcular correlação.")

    corr = numeric_df.corr(method=method)[target].drop(target)

    result = corr.reset_index()
    result.columns = ["feature", "correlation"]

    if absolute:
        result["abs_correlation"] = result["correlation"].abs()
        result = result.sort_values(by="abs_correlation", ascending=False)
    else:
        result = result.sort_values(by="correlation", ascending=False)

    return result.head(n_features)


def evaluate_classifier(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def get_feature_importance(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)


def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000, random_state=42))
    ])

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def run_grid_search(best_model_name, X_train, y_train, cv=5):
    if "Logistic" in best_model_name:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=2000, random_state=42))
        ])

        param_grid = {
            "logistic__C": [0.01, 0.1, 1, 10],
            "logistic__penalty": ["l2"],
            "logistic__solver": ["lbfgs"]
        }

    else:
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"]
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


def full_model_comparison_workflow(
    df,
    target="Fire_Alarm",
    n_top_features=5,
    test_size=0.2,
    grid_cv=5,
    final_cv=6
):
    start_time = time.time()

    _, _, outputs_dir = get_project_paths()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # =====================================================
    # 1) Feature Importance usando Random Forest
    # =====================================================
    importance_df = get_feature_importance(X_train, y_train)

    top_features = importance_df["feature"].head(n_top_features).tolist()

    print("\n==== FEATURE IMPORTANCE ====\n")
    print(importance_df)

    print("\n==== TOP FEATURES ====\n")
    print(top_features)

    # =====================================================
    # 2) Logistic Regression - todas features
    # =====================================================
    log_all = train_logistic_regression(X_train, y_train)

    log_all_result = evaluate_classifier(
        log_all,
        X_test,
        y_test,
        "Logistic Regression - All Features"
    )

    # =====================================================
    # 3) Logistic Regression - top features
    # =====================================================
    log_top = train_logistic_regression(X_train[top_features], y_train)

    log_top_result = evaluate_classifier(
        log_top,
        X_test[top_features],
        y_test,
        "Logistic Regression - Top Features"
    )

    # =====================================================
    # 4) Random Forest - todas features
    # =====================================================
    rf_all = train_random_forest(X_train, y_train)

    rf_all_result = evaluate_classifier(
        rf_all,
        X_test,
        y_test,
        "Random Forest - All Features"
    )

    # =====================================================
    # 5) Random Forest - top features
    # =====================================================
    rf_top = train_random_forest(X_train[top_features], y_train)

    rf_top_result = evaluate_classifier(
        rf_top,
        X_test[top_features],
        y_test,
        "Random Forest - Top Features"
    )

    initial_comparison_df = pd.DataFrame([
        log_all_result,
        log_top_result,
        rf_all_result,
        rf_top_result
    ])

    print("\n==== COMPARAÇÃO INICIAL ====\n")
    print(initial_comparison_df)

    # =====================================================
    # 6) Selecionar melhor modelo inicial
    # =====================================================
    best_initial = initial_comparison_df.sort_values(
        by="f1_score",
        ascending=False
    ).iloc[0]

    best_initial_name = best_initial["model"]

    if "Top Features" in best_initial_name:
        selected_features = top_features
    else:
        selected_features = X.columns.tolist()

    if best_initial_name == "Logistic Regression - All Features":
        baseline_model = log_all
    elif best_initial_name == "Logistic Regression - Top Features":
        baseline_model = log_top
    elif best_initial_name == "Random Forest - All Features":
        baseline_model = rf_all
    else:
        baseline_model = rf_top

    print("\n==== MELHOR MODELO INICIAL ====\n")
    print(best_initial_name)
    print("\nFeatures usadas:")
    print(selected_features)

    # =====================================================
    # 7) GridSearch apenas no melhor modelo
    # =====================================================
    grid = run_grid_search(
        best_model_name=best_initial_name,
        X_train=X_train[selected_features],
        y_train=y_train,
        cv=grid_cv
    )

    grid_model = grid.best_estimator_

    print("\n==== MELHORES PARÂMETROS DO GRIDSEARCH ====\n")
    print(grid.best_params_)

    # =====================================================
    # 8) Comparar melhor baseline vs GridSearch
    # =====================================================
    baseline_result = evaluate_classifier(
        baseline_model,
        X_test[selected_features],
        y_test,
        f"Baseline - {best_initial_name}"
    )

    grid_result = evaluate_classifier(
        grid_model,
        X_test[selected_features],
        y_test,
        f"GridSearch - {best_initial_name}"
    )

    final_comparison_df = pd.DataFrame([
        baseline_result,
        grid_result
    ])

    print("\n==== COMPARAÇÃO FINAL: BASELINE VS GRIDSEARCH ====\n")
    print(final_comparison_df)

    # =====================================================
    # 9) Melhor modelo final
    # =====================================================
    best_final = final_comparison_df.sort_values(
        by="f1_score",
        ascending=False
    ).iloc[0]

    if "GridSearch" in best_final["model"]:
        final_model = grid_model
        final_model_name = best_final["model"]
    else:
        final_model = baseline_model
        final_model_name = best_final["model"]

    print("\n==== MELHOR MODELO FINAL ====\n")
    print(final_model_name)

    # =====================================================
    # 10) Cross Validation no melhor modelo final
    # =====================================================
    cv_df = run_cross_validation(
        final_model,
        X[selected_features],
        y,
        cv=final_cv
    )

    cv_df.insert(0, "model", final_model_name)
    cv_df.insert(1, "features_used", ", ".join(selected_features))

    print("\n==== CROSS VALIDATION FINAL ====\n")

    cv_display = cv_df.copy()

    # arredondar métricas
    cols_to_round = [col for col in cv_display.columns if "mean" in col or "std" in col]
    cv_display[cols_to_round] = cv_display[cols_to_round].round(4)

    # encurtar features
    cv_display["features_used"] = cv_display["features_used"].apply(
        lambda x: f"{len(x.split(', '))} features"
    )

    print(cv_display.drop(columns=["features_used"]).to_string(index=False))

    print("\nFeatures utilizadas:")
    for f in selected_features:
        print(f"- {f}")

    # =====================================================
    # 11) Salvar resultados
    # =====================================================
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

    importance_df.to_csv(
        outputs_dir / "feature_importance.csv",
        index=False
    )

    elapsed_time = time.time() - start_time

    print("\n==== ARQUIVOS SALVOS ====")
    print(outputs_dir / "initial_model_comparison.csv")
    print(outputs_dir / "final_model_comparison.csv")
    print(outputs_dir / "cross_validation_results.csv")
    print(outputs_dir / "feature_importance.csv")

    print(f"\nTempo total: {elapsed_time / 60:.2f} minutos")

    return {
        "feature_importance": importance_df,
        "top_features": top_features,
        "initial_comparison": initial_comparison_df,
        "final_comparison": final_comparison_df,
        "cross_validation": cv_df,
        "best_model": final_model,
        "best_features": selected_features
    }


def suggest_correlated_features_to_drop(
    df: pd.DataFrame,
    target: str = "Fire_Alarm",
    threshold: float = 0.90
) -> tuple[pd.DataFrame, list]:
    """
    Identifica features altamente correlacionadas entre si e sugere quais remover.
    """

    df = df.copy()

    numeric_df = df.select_dtypes(include=np.number)

    if target not in numeric_df.columns:
        raise ValueError(f"Target '{target}' não encontrado ou não é numérico.")

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    corr_matrix = X.corr().abs()
    target_corr = X.corrwith(y).abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    results = []
    drop_features = set()

    for col in upper_triangle.columns:
        high_corr_rows = upper_triangle.index[upper_triangle[col] >= threshold].tolist()

        for row in high_corr_rows:
            corr_between_features = upper_triangle.loc[row, col]

            row_target_corr = target_corr[row]
            col_target_corr = target_corr[col]

            if row_target_corr >= col_target_corr:
                suggested_drop = col
                suggested_keep = row
            else:
                suggested_drop = row
                suggested_keep = col

            drop_features.add(suggested_drop)

            results.append({
                "feature_1": row,
                "feature_2": col,
                "corr_between_features": corr_between_features,
                "feature_1_corr_with_target": row_target_corr,
                "feature_2_corr_with_target": col_target_corr,
                "suggested_keep": suggested_keep,
                "suggested_drop": suggested_drop
            })

    correlated_pairs_df = pd.DataFrame(results)

    if not correlated_pairs_df.empty:
        correlated_pairs_df = correlated_pairs_df.sort_values(
            by="corr_between_features",
            ascending=False
        )

    drop_features = sorted(list(drop_features))

    return correlated_pairs_df, drop_features


def main() -> None:
    df = load_data()

    # Limpeza
    df = clean_columns(df, to_lower=False, remove_special=False, drop_cols=["CNT", "UTC"])

    # Análise
    describe_data(df)
    # plot_eda_plotly(df)
    #
    # top_corr = get_top_corr_features(
    #     df,
    #     target="Fire_Alarm",
    #     n_features=5,
    #     method="pearson"
    # )
    #
    # top_corr_cols = top_corr["feature"].tolist()
    #
    # scatter_pairs(
    #     df,
    #     cols=top_corr_cols,
    #     target="Fire_Alarm",
    #     sample_size=2000,
    #     save_html=True
    # )

    # Modelagem completa
    results = full_model_comparison_workflow(
        df=df,
        target="Fire_Alarm",
        n_top_features=10,
        grid_cv=10,
        final_cv=10
    )

    #Modelagem reduzida
    correlated_pairs_df, drop_features = suggest_correlated_features_to_drop(
        df,
        target="Fire_Alarm",
        threshold=0.90
    )

    print("\n==== MODELO REDUZIDO ====\n")
    print("\nPARES ALTAMENTE CORRELACIONADOS\n")
    print(correlated_pairs_df)

    print("\nFEATURES SUGERIDAS PARA REMOVER\n")
    print(drop_features)

    df_reduced = df.drop(columns=drop_features)

    results2 = full_model_comparison_workflow(
        df=df_reduced,
        target="Fire_Alarm",
        n_top_features=10,
        grid_cv=10,
        final_cv=10
    )

if __name__ == "__main__":
    main()