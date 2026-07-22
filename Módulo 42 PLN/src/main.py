from __future__ import annotations

from pathlib import Path
import time
import warnings
import re
import html

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 80)
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

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, data_dir, _ = get_project_paths()

    df = pd.read_csv(
        data_dir / "Training_data_5.csv",
        sep=",",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"]
    )

    df_test = pd.read_csv(
        data_dir / "Test_data.csv",
        sep=",",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"]
    )

    return df, df_test


# =========================================================
# EDA
# =========================================================

def describe_data(df: pd.DataFrame, target: str) -> None:
    print("\n==== VISÃO GERAL DOS DADOS ====\n")
    print(df.head().to_string())

    print("\n==== SHAPE ====\n")
    print(df.shape)

    print("\n==== TIPOS DE DADOS ====\n")
    print(df.dtypes)

    print("\n==== VALORES NULOS ====\n")
    print(df.isnull().sum())

    print("\n==== ANÁLISE DESCRITIVA ====\n")
    print(df.describe(include="all").to_string())

    if target in df.columns:
        print("\n==== DISTRIBUIÇÃO DO TARGET ====\n")
        print(df[target].value_counts())

        print("\nProporção:")
        print(df[target].value_counts(normalize=True))


def analyze_text_length(
    df: pd.DataFrame,
    text_col: str,
    target: str
) -> None:
    df = df.copy()

    df["text_length"] = df[text_col].astype(str).str.len()
    df["word_count"] = df[text_col].astype(str).str.split().str.len()

    print("\n==== TAMANHO DOS TEXTOS POR CLASSE ====\n")
    print(df.groupby(target)[["text_length", "word_count"]].mean())


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text: str) -> str:
    """
    Limpa texto removendo HTML, símbolos e espaços extras.
    """

    text = str(text)

    text = html.unescape(text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"News Headlines:", " ", text, flags=re.IGNORECASE)

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_text_data(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    text_col: str,
    target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica limpeza de texto no treino e teste.
    """

    df = df.copy()
    df_test = df_test.copy()

    df[text_col] = df[text_col].apply(clean_text)
    df_test[text_col] = df_test[text_col].apply(clean_text)

    df = df.dropna(subset=[text_col, target])

    return df, df_test


# =========================================================
# MODELS
# =========================================================

def build_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english"
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])


def build_svm_pipeline() -> Pipeline:
    svm = LinearSVC(
        random_state=42
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=svm,
        cv=3
    )

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english"
        )),
        ("classifier", calibrated_svm)
    ])


def get_param_grid(model_type: str) -> dict:
    if model_type == "logistic":
        return {
            "tfidf__max_features": [10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [2, 5],
            "classifier__C": [1, 3, 5]
        }

    if model_type == "svm":
        return {
            "tfidf__max_features": [10000, 20000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [2, 5],
            "classifier__estimator__C": [0.5, 1, 2]
        }

    raise ValueError("model_type inválido.")


# =========================================================
# EVALUATION
# =========================================================

def evaluate_classifier(
    model,
    X_test,
    y_test,
    model_name: str
) -> dict:
    y_pred = model.predict(X_test)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }


def run_cross_validation(
    model,
    X,
    y,
    cv: int = 5
) -> pd.DataFrame:
    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted"
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
# EXPERIMENT
# =========================================================

def run_model_experiment(
    X,
    y,
    model_type: str,
    model_display_name: str,
    test_size: float = 0.2,
    grid_cv: int = 3,
    final_cv: int = 5
) -> dict:
    if model_type == "logistic":
        baseline_model = build_logistic_pipeline()

    elif model_type == "svm":
        baseline_model = build_svm_pipeline()

    else:
        raise ValueError("model_type inválido.")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Baseline
    baseline_model.fit(X_train, y_train)

    baseline_result = evaluate_classifier(
        baseline_model,
        X_valid,
        y_valid,
        model_name=f"{model_display_name} Baseline - Train Test Split"
    )

    # GridSearch
    grid = GridSearchCV(
        estimator=baseline_model,
        param_grid=get_param_grid(model_type),
        scoring="f1_macro",
        cv=grid_cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    grid_model = grid.best_estimator_

    grid_result = evaluate_classifier(
        grid_model,
        X_valid,
        y_valid,
        model_name=f"{model_display_name} GridSearch - Train Test Split"
    )

    print(f"\n==== MELHORES PARÂMETROS - {model_display_name} ====\n")
    print(grid.best_params_)

    # Cross Validation baseline
    cv_baseline = run_cross_validation(
        baseline_model,
        X,
        y,
        cv=final_cv
    )

    cv_baseline.insert(
        0,
        "model",
        f"{model_display_name} Baseline - Cross Validation"
    )

    # Cross Validation grid
    cv_grid = run_cross_validation(
        grid_model,
        X,
        y,
        cv=final_cv
    )

    cv_grid.insert(
        0,
        "model",
        f"{model_display_name} GridSearch - Cross Validation"
    )

    test_results = pd.DataFrame([
        baseline_result,
        grid_result
    ])

    cv_results = pd.concat(
        [cv_baseline, cv_grid],
        ignore_index=True
    )

    best_cv = cv_results.sort_values(
        by="f1_macro_mean",
        ascending=False
    ).iloc[0]

    if "Baseline" in best_cv["model"]:
        best_model = baseline_model
    else:
        best_model = grid_model

    return {
        "baseline_model": baseline_model,
        "grid_model": grid_model,
        "best_model": best_model,
        "best_result": best_cv,
        "grid_best_params": grid.best_params_,
        "test_results": test_results,
        "cv_results": cv_results,
        "X_valid": X_valid,
        "y_valid": y_valid
    }


# =========================================================
# COMPARISON
# =========================================================

def build_full_comparison(
    test_results_list: list[pd.DataFrame],
    cv_results_list: list[pd.DataFrame]
) -> pd.DataFrame:
    test_comparison_df = pd.concat(
        test_results_list,
        ignore_index=True
    )

    cv_comparison_df = pd.concat(
        cv_results_list,
        ignore_index=True
    )

    test_display = test_comparison_df.copy()
    test_display["evaluation_type"] = "Train Test Split"

    cv_display = cv_comparison_df.copy()
    cv_display["evaluation_type"] = "Cross Validation"

    cv_display = cv_display.rename(columns={
        "accuracy_mean": "accuracy",
        "precision_macro_mean": "precision_macro",
        "recall_macro_mean": "recall_macro",
        "f1_macro_mean": "f1_macro",
        "f1_weighted_mean": "f1_weighted"
    })

    cols = [
        "model",
        "evaluation_type",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted"
    ]

    full_comparison_df = pd.concat(
        [
            test_display[cols],
            cv_display[cols]
        ],
        ignore_index=True
    )

    return full_comparison_df


def select_best_model_from_cv(
    cv_results_list: list[pd.DataFrame]
) -> pd.Series:
    cv_comparison_df = pd.concat(
        cv_results_list,
        ignore_index=True
    )

    best_cv = cv_comparison_df.sort_values(
        by="f1_macro_mean",
        ascending=False
    ).iloc[0]

    return best_cv


# =========================================================
# PREDICTIONS
# =========================================================

def show_predictions_table(
    model,
    X_valid,
    y_valid,
    n_rows: int = 20
) -> pd.DataFrame:
    y_pred = model.predict(X_valid)

    results_df = pd.DataFrame({
        "real_class": y_valid.values,
        "predicted_class": y_pred
    })

    results_df["correct"] = np.where(
        results_df["real_class"] == results_df["predicted_class"],
        "Correct",
        "Wrong"
    )

    print("\n==== PREVISÕES DO MELHOR MODELO ====\n")
    print(results_df.head(n_rows).to_string(index=False))

    return results_df


def create_test_predictions(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    target: str,
    filename: str = "submission.csv"
) -> pd.DataFrame:
    _, _, outputs_dir = get_project_paths()

    X_full = train_df[text_col]
    y_full = train_df[target]

    X_test = test_df[text_col]

    model.fit(X_full, y_full)

    predictions = model.predict(X_test)

    submission = test_df.copy()
    submission["Predicted_Topic"] = predictions

    submission.to_csv(
        outputs_dir / filename,
        index=False
    )

    return submission


# =========================================================
# FULL WORKFLOW
# =========================================================

def full_tfidf_workflow(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    text_col: str = "News Headline",
    target: str = "News Topic",
    test_size: float = 0.2,
    grid_cv: int = 3,
    final_cv: int = 5
) -> dict:
    start_time = time.time()

    _, _, outputs_dir = get_project_paths()

    X = df[text_col]
    y = df[target]

    print("\n==== TREINANDO LOGISTIC REGRESSION ====\n")

    logistic_results = run_model_experiment(
        X=X,
        y=y,
        model_type="logistic",
        model_display_name="Logistic Regression",
        test_size=test_size,
        grid_cv=grid_cv,
        final_cv=final_cv
    )

    print("\n==== TREINANDO SVM ====\n")

    svm_results = run_model_experiment(
        X=X,
        y=y,
        model_type="svm",
        model_display_name="SVM",
        test_size=test_size,
        grid_cv=grid_cv,
        final_cv=final_cv
    )

    full_comparison_df = build_full_comparison(
        test_results_list=[
            logistic_results["test_results"],
            svm_results["test_results"]
        ],
        cv_results_list=[
            logistic_results["cv_results"],
            svm_results["cv_results"]
        ]
    )

    print("\n==== COMPARAÇÃO GERAL: TF-IDF + LOGISTIC VS SVM ====\n")
    print(full_comparison_df.round(4).to_string(index=False))

    best_cv = select_best_model_from_cv([
        logistic_results["cv_results"],
        svm_results["cv_results"]
    ])

    print("\n==== MELHOR RESULTADO GERAL POR CROSS VALIDATION ====\n")
    for key, value in best_cv.items():
        if isinstance(value, (int, float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    best_model_name = best_cv["model"]

    if "Logistic Regression" in best_model_name:
        best_model = logistic_results["best_model"]
        X_valid = logistic_results["X_valid"]
        y_valid = logistic_results["y_valid"]
    else:
        best_model = svm_results["best_model"]
        X_valid = svm_results["X_valid"]
        y_valid = svm_results["y_valid"]

    prediction_table = show_predictions_table(
        model=best_model,
        X_valid=X_valid,
        y_valid=y_valid,
        n_rows=20
    )

    submission = create_test_predictions(
        model=best_model,
        train_df=df,
        test_df=df_test,
        text_col=text_col,
        target=target,
        filename="tfidf_predictions.csv"
    )

    full_comparison_df.to_csv(
        outputs_dir / "tfidf_model_comparison.csv",
        index=False
    )

    prediction_table.to_csv(
        outputs_dir / "tfidf_predictions_table.csv",
        index=False
    )

    elapsed_time = time.time() - start_time

    print("\n==== ARQUIVOS SALVOS ====")
    print(outputs_dir / "tfidf_model_comparison.csv")
    print(outputs_dir / "tfidf_predictions_table.csv")
    print(outputs_dir / "tfidf_predictions.csv")

    print(f"\nTempo total: {elapsed_time / 60:.2f} minutos")

    return {
        "logistic": logistic_results,
        "svm": svm_results,
        "full_comparison": full_comparison_df,
        "best_cv": best_cv,
        "best_model": best_model,
        "prediction_table": prediction_table,
        "submission": submission
    }


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    text_col = "News Headline"
    target = "News Topic"

    df, df_test = load_data()

    print("\n==== DATASET INICIAL ====\n")
    describe_data(df, target=target)
    analyze_text_length(df, text_col=text_col, target=target)

    df, df_test = prepare_text_data(
        df=df,
        df_test=df_test,
        text_col=text_col,
        target=target
    )

    print("\n==== DATASET APÓS LIMPEZA DO TEXTO ====\n")
    print(df[[text_col, target]].head().to_string())

    results = full_tfidf_workflow(
        df=df,
        df_test=df_test,
        text_col=text_col,
        target=target,
        test_size=0.2,
        grid_cv=3,
        final_cv=5
    )


if __name__ == "__main__":
    main()