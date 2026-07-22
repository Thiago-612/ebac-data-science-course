from __future__ import annotations

from pathlib import Path
import time
import warnings
import re

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

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
    """

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    return project_root, data_dir, outputs_dir


# =========================================================
# LOAD DATA
# =========================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega train.csv e test.csv do Titanic.
    """

    _, data_dir, _ = get_project_paths()

    train_df = pd.read_csv(
        data_dir / "train.csv",
        sep=",",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"]
    )

    test_df = pd.read_csv(
        data_dir / "test.csv",
        sep=",",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"]
    )

    return train_df, test_df


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
# FEATURE ENGINEERING
# =========================================================

def create_title_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai o título do passageiro a partir da coluna Name.
    Exemplo: Mr, Mrs, Miss, Master, Rare.
    """

    df = df.copy()

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    rare_titles = [
        "Lady", "Countess", "Capt", "Col",
        "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer", "Dona"
    ]

    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    return df


def create_titanic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features adicionais para o Titanic.
    """

    df = df.copy()

    df["HasCabin"] = df["Cabin"].notnull().astype(int)

    df = create_title_feature(df)

    df["FamilySize"] = (
        df["SibSp"] +
        df["Parch"] +
        1
    )

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

    df["FarePerPerson"] = (
        df["Fare"] / df["FamilySize"]
    )

    return df


def prepare_titanic_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "Survived"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Aplica feature engineering e limpeza em train/test.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    test_passenger_ids = test_df["PassengerId"]

    train_df = create_titanic_features(train_df)
    test_df = create_titanic_features(test_df)

    drop_cols = [
        "PassengerId",
        "Ticket",
        "Cabin",
        "Name"
    ]

    train_df = clean_columns(
        train_df,
        drop_cols=drop_cols,
        to_lower=False,
        remove_special=False,
        remove_unnamed=True
    )

    test_df = clean_columns(
        test_df,
        drop_cols=drop_cols,
        to_lower=False,
        remove_special=False,
        remove_unnamed=True
    )

    return train_df, test_df, test_passenger_ids


# =========================================================
# EDA
# =========================================================

def describe_data(df: pd.DataFrame, target: str) -> None:
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
    Separa features numéricas e categóricas.
    """

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Cria pré-processador com imputação, escala e OneHotEncoder.
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
# MODEL BUILDERS
# =========================================================

def build_xgboost_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Cria pipeline XGBoost.
    """

    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("classifier", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])


def build_svm_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Cria pipeline SVM com probability=True.
    """

    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("classifier", SVC(
            probability=True,
            random_state=42
        ))
    ])


def build_logistic_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Cria pipeline Logistic Regression.
    """

    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])


# =========================================================
# PARAM GRIDS
# =========================================================

def get_param_grid(model_type: str) -> dict:
    """
    Retorna os hiperparâmetros para cada modelo.
    """

    if model_type == "xgboost":
        return {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [2, 3],
            "classifier__learning_rate": [0.05, 0.1]
        }

    if model_type == "svm":
        return [
            {
                "classifier__kernel": ["linear"],
                "classifier__C": [0.1, 1, 10, 100]
            },
            {
                "classifier__kernel": ["rbf"],
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__gamma": ["scale", "auto"]
            },
            {
                "classifier__kernel": ["poly"],
                "classifier__C": [0.1, 1, 10],
                "classifier__degree": [2, 3],
                "classifier__gamma": ["scale", "auto"]
            }
        ]

    if model_type == "logistic":
        return {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"]
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
    """
    Avalia classificador binário.
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


# =========================================================
# MODEL EXPERIMENT
# =========================================================

def run_model_experiment(
    X,
    y,
    model_type: str,
    model_display_name: str,
    test_size: float = 0.2,
    grid_cv: int = 5,
    final_cv: int = 5
) -> dict:
    """
    Executa baseline, GridSearch e Cross Validation para um modelo.
    """

    if model_type == "xgboost":
        baseline_model = build_xgboost_pipeline(X)
    elif model_type == "svm":
        baseline_model = build_svm_pipeline(X)
    elif model_type == "logistic":
        baseline_model = build_logistic_pipeline(X)
    else:
        raise ValueError("model_type inválido.")

    X_train, X_test, y_train, y_test = train_test_split(
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
        X_test,
        y_test,
        model_name=f"{model_display_name} Baseline - Train Test Split"
    )

    # GridSearch
    grid = GridSearchCV(
        estimator=baseline_model,
        param_grid=get_param_grid(model_type),
        scoring="f1",
        cv=grid_cv,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)

    grid_model = grid.best_estimator_

    grid_result = evaluate_classifier(
        grid_model,
        X_test,
        y_test,
        model_name=f"{model_display_name} GridSearch - Train Test Split"
    )

    # CV baseline
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

    # CV grid
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
        by="f1_score_mean",
        ascending=False
    ).iloc[0]

    if "Baseline" in best_cv["model"]:
        best_model = baseline_model
    else:
        best_model = grid_model

    return {
        "model_type": model_type,
        "model_display_name": model_display_name,
        "baseline_model": baseline_model,
        "grid_model": grid_model,
        "best_model": best_model,
        "best_result": best_cv,
        "grid_best_params": grid.best_params_,
        "test_results": test_results,
        "cv_results": cv_results,
        "X_test": X_test,
        "y_test": y_test
    }


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_pipeline_feature_importance(model) -> pd.DataFrame:
    """
    Retorna feature importance para modelos com atributo feature_importances_.
    Funciona principalmente para XGBoost.
    """

    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_
    }).sort_values(
        by="importance",
        ascending=False
    )

    return importance_df


# =========================================================
# ENSEMBLE
# =========================================================

def build_soft_voting_ensemble(
    xgb_model,
    svm_model,
    logistic_model
) -> VotingClassifier:
    """
    Cria ensemble soft voting com XGBoost, SVM e Logistic Regression.
    """

    ensemble = VotingClassifier(
        estimators=[
            ("xgboost", xgb_model),
            ("svm", svm_model),
            ("logistic", logistic_model)
        ],
        voting="soft",
        n_jobs=-1
    )

    return ensemble


def evaluate_ensemble(
    ensemble_model,
    X,
    y,
    test_size: float = 0.2,
    final_cv: int = 5
) -> dict:
    """
    Avalia ensemble com train/test split e cross validation.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    ensemble_model.fit(X_train, y_train)

    test_result = evaluate_classifier(
        ensemble_model,
        X_test,
        y_test,
        model_name="Soft Voting Ensemble - Train Test Split"
    )

    test_results = pd.DataFrame([test_result])

    cv_results = run_cross_validation(
        ensemble_model,
        X,
        y,
        cv=final_cv
    )

    cv_results.insert(
        0,
        "model",
        "Soft Voting Ensemble - Cross Validation"
    )

    return {
        "model": ensemble_model,
        "test_results": test_results,
        "cv_results": cv_results,
        "X_test": X_test,
        "y_test": y_test
    }


# =========================================================
# COMPARISON
# =========================================================

def build_full_comparison(
    test_results_list: list[pd.DataFrame],
    cv_results_list: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    Consolida resultados de train/test e cross validation.
    """

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
            test_display[cols],
            cv_display[cols]
        ],
        ignore_index=True
    )

    return full_comparison_df


def select_best_model_from_cv(
    cv_results_list: list[pd.DataFrame]
) -> pd.Series:
    """
    Seleciona melhor resultado pela média do F1-score na validação cruzada.
    """

    cv_comparison_df = pd.concat(
        cv_results_list,
        ignore_index=True
    )

    best_cv = cv_comparison_df.sort_values(
        by="f1_score_mean",
        ascending=False
    ).iloc[0]

    return best_cv


# =========================================================
# PREDICTIONS TABLE
# =========================================================

def show_predictions_table(
    model,
    X_test,
    y_test,
    threshold: float = 0.5,
    n_rows: int = 20
) -> pd.DataFrame:
    """
    Mostra probabilidades e previsão binária.
    """

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    results_df = pd.DataFrame({
        "real_class": y_test.values,
        "predicted_probability": y_proba,
        "threshold": threshold,
        "binary_prediction": y_pred
    })

    results_df["correct"] = np.where(
        results_df["real_class"] == results_df["binary_prediction"],
        "Correct",
        "Wrong"
    )

    results_df["predicted_probability"] = (
        results_df["predicted_probability"].round(4)
    )

    print("\n==== PREVISÕES DO ENSEMBLE ====\n")
    print(results_df.head(n_rows).to_string(index=False))

    return results_df


# =========================================================
# SUBMISSION
# =========================================================

def create_kaggle_submission(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    passenger_ids: pd.Series,
    target: str = "Survived",
    filename: str = "submission.csv"
) -> pd.DataFrame:
    """
    Treina o modelo em todo o treino e gera submission para Kaggle.
    """

    _, _, outputs_dir = get_project_paths()

    X_full = train_df.drop(columns=[target])
    y_full = train_df[target]

    X_kaggle_test = test_df.copy()

    model.fit(X_full, y_full)

    predictions = model.predict(X_kaggle_test)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int)
    })

    submission.to_csv(
        outputs_dir / filename,
        index=False
    )

    return submission


# =========================================================
# FULL WORKFLOW
# =========================================================

def full_titanic_workflow(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    passenger_ids: pd.Series,
    target: str = "Survived",
    test_size: float = 0.2,
    grid_cv: int = 5,
    final_cv: int = 5
) -> dict:
    """
    Executa XGBoost, SVM, Logistic Regression e Soft Voting Ensemble.
    """

    start_time = time.time()

    _, _, outputs_dir = get_project_paths()

    X = train_df.drop(columns=[target])
    y = train_df[target]

    print("\n==== TREINANDO XGBOOST ====\n")
    xgb_results = run_model_experiment(
        X=X,
        y=y,
        model_type="xgboost",
        model_display_name="XGBoost",
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

    print("\n==== TREINANDO ENSEMBLE ====\n")

    ensemble_model = build_soft_voting_ensemble(
        xgb_model=xgb_results["best_model"],
        svm_model=svm_results["best_model"],
        logistic_model=logistic_results["best_model"]
    )

    ensemble_results = evaluate_ensemble(
        ensemble_model=ensemble_model,
        X=X,
        y=y,
        test_size=test_size,
        final_cv=final_cv
    )

    full_comparison_df = build_full_comparison(
        test_results_list=[
            xgb_results["test_results"],
            svm_results["test_results"],
            logistic_results["test_results"],
            ensemble_results["test_results"]
        ],
        cv_results_list=[
            xgb_results["cv_results"],
            svm_results["cv_results"],
            logistic_results["cv_results"],
            ensemble_results["cv_results"]
        ]
    )

    print("\n==== COMPARAÇÃO GERAL: XGBOOST vs SVM vs LOGISTIC vs ENSEMBLE ====\n")
    print(full_comparison_df.round(4).to_string(index=False))

    best_cv = select_best_model_from_cv([
        xgb_results["cv_results"],
        svm_results["cv_results"],
        logistic_results["cv_results"],
        ensemble_results["cv_results"]
    ])

    print("\n==== MELHOR RESULTADO GERAL POR CROSS VALIDATION ====\n")
    for key, value in best_cv.items():

        if isinstance(value, (int, float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    feature_importance_df = get_pipeline_feature_importance(
        xgb_results["best_model"]
    )

    if not feature_importance_df.empty:
        print("\n==== FEATURE IMPORTANCE - XGBOOST ====\n")
        print(feature_importance_df.round(4).to_string(index=False))

    prediction_table = show_predictions_table(
        model=ensemble_results["model"],
        X_test=ensemble_results["X_test"],
        y_test=ensemble_results["y_test"],
        threshold=0.5,
        n_rows=20
    )

    submission = create_kaggle_submission(
        model=ensemble_results["model"],
        train_df=train_df,
        test_df=test_df,
        passenger_ids=passenger_ids,
        target=target,
        filename="submission_ensemble.csv"
    )

    full_comparison_df.to_csv(
        outputs_dir / "full_model_comparison.csv",
        index=False
    )

    if not feature_importance_df.empty:
        feature_importance_df.to_csv(
            outputs_dir / "xgboost_feature_importance.csv",
            index=False
        )

    prediction_table.to_csv(
        outputs_dir / "ensemble_predictions_table.csv",
        index=False
    )

    elapsed_time = time.time() - start_time

    print("\n==== ARQUIVOS SALVOS ====")
    print(outputs_dir / "full_model_comparison.csv")
    print(outputs_dir / "xgboost_feature_importance.csv")
    print(outputs_dir / "ensemble_predictions_table.csv")
    print(outputs_dir / "submission_ensemble.csv")

    print(f"\nTempo total: {elapsed_time / 60:.2f} minutos")

    return {
        "xgboost": xgb_results,
        "svm": svm_results,
        "logistic": logistic_results,
        "ensemble": ensemble_results,
        "full_comparison": full_comparison_df,
        "best_cv": best_cv,
        "feature_importance": feature_importance_df,
        "prediction_table": prediction_table,
        "submission": submission
    }


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """
    Função principal.
    """

    target = "Survived"

    train_df, test_df = load_data()

    train_df, test_df, passenger_ids = prepare_titanic_data(
        train_df=train_df,
        test_df=test_df,
        target=target
    )

    print("\n==== DATASET APÓS FEATURE ENGINEERING ====\n")
    describe_data(train_df, target=target)
    analyze_by_class(train_df, target=target)

    results = full_titanic_workflow(
        train_df=train_df,
        test_df=test_df,
        passenger_ids=passenger_ids,
        target=target,
        test_size=0.2,
        grid_cv=5,
        final_cv=10
    )


if __name__ == "__main__":
    main()