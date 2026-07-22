from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# =========================================================
# PATHS
# =========================================================
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


# =========================================================
# LOAD DATA
# =========================================================
def load_data() -> pd.DataFrame:
    """
    EN: Load the wine dataset.
    PT: Carrega o dataset de vinho.
    """
    _, data_dir, _ = get_project_paths()
    df = pd.read_csv(data_dir / "winequality-red.csv")
    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    EN: Create engineered features.
    PT: Cria variáveis derivadas.
    """
    df_new = df.copy()

    # Avoid division by zero / Evita divisão por zero
    eps = 1e-6

    # Ratios / Razões
    df_new["free_total_sulfur_ratio"] = (
        df_new["free sulfur dioxide"] / (df_new["total sulfur dioxide"] + eps)
    )
    df_new["alcohol_density_ratio"] = df_new["alcohol"] / (df_new["density"] + eps)
    df_new["acid_ratio"] = df_new["fixed acidity"] / (df_new["volatile acidity"] + eps)
    df_new["sulphates_chlorides_ratio"] = df_new["sulphates"] / (df_new["chlorides"] + eps)

    # Interactions / Interações
    df_new["alcohol_x_sulphates"] = df_new["alcohol"] * df_new["sulphates"]
    df_new["alcohol_x_volatile_acidity"] = df_new["alcohol"] * df_new["volatile acidity"]
    df_new["citric_x_fixed_acidity"] = df_new["citric acid"] * df_new["fixed acidity"]

    # Transformations / Transformações
    df_new["log_total_sulfur_dioxide"] = np.log1p(df_new["total sulfur dioxide"])
    df_new["log_free_sulfur_dioxide"] = np.log1p(df_new["free sulfur dioxide"])
    df_new["sqrt_sulphates"] = np.sqrt(np.clip(df_new["sulphates"], a_min=0, a_max=None))

    # Binning-like feature but numeric / Faixas numéricas
    df_new["alcohol_bin_q"] = pd.qcut(
        df_new["alcohol"],
        q=4,
        labels=False,
        duplicates="drop"
    )

    return df_new


# =========================================================
# FEATURE SELECTION
# =========================================================
def select_features_by_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    top_n: int = 12,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    EN: Select top features based on ExtraTrees importance.
    PT: Seleciona as principais variáveis com base na importância do ExtraTrees.
    """
    selector_model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )
    selector_model.fit(x_train, y_train)

    importance_df = pd.DataFrame({
        "feature": x_train.columns,
        "importance": selector_model.feature_importances_
    }).sort_values("importance", ascending=False)

    selected_features = importance_df.head(top_n)["feature"].tolist()

    x_train_sel = x_train[selected_features].copy()
    x_test_sel = x_test[selected_features].copy()

    return x_train_sel, x_test_sel, importance_df


# =========================================================
# METRICS
# =========================================================
def calculate_metrics(y_true, y_pred) -> dict:
    """
    EN: Calculate evaluation metrics.
    PT: Calcula métricas de avaliação.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def print_metrics(title: str, metrics: dict) -> None:
    """
    EN: Print metrics in a readable format.
    PT: Exibe as métricas em formato legível.
    """
    print(f"\n==== {title.upper()} ====\n")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


# =========================================================
# MODEL EVALUATION
# =========================================================
def evaluate_model_train_test(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """
    EN: Fit model and evaluate on train/test.
    PT: Treina o modelo e avalia no treino/teste.
    """
    fitted_model = clone(model)
    fitted_model.fit(x_train, y_train)

    y_pred_train = fitted_model.predict(x_train)
    y_pred_test = fitted_model.predict(x_test)

    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    print_metrics(f"{model_name} - Train", train_metrics)
    print_metrics(f"{model_name} - Test", test_metrics)

    return {
        "model_name": model_name,
        "model": fitted_model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def cross_validate_model(model, x_train, y_train, cv, scoring="f1_macro") -> float:
    """
    EN: Run cross-validation on training set.
    PT: Executa validação cruzada no conjunto de treino.
    """
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


# =========================================================
# RANDOM SEARCH
# =========================================================
def tune_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    random_state: int = 42,
):
    """
    EN: Tune Random Forest using RandomizedSearchCV.
    PT: Ajusta o Random Forest com RandomizedSearchCV.
    """
    model = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [6, 8, 10, 12, 15, 20, None],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_features": ["sqrt", "log2", None],
        "criterion": ["gini", "entropy", "log_loss"],
        "bootstrap": [True]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="f1_macro",
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    search.fit(x_train, y_train)
    return search


# =========================================================
# OUTPUTS
# =========================================================
def save_metrics_json(metrics_dict: dict, filepath: Path) -> None:
    """
    EN: Save metrics dictionary to JSON.
    PT: Salva o dicionário de métricas em JSON.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)


def save_classification_report(
    y_true,
    y_pred,
    filepath: Path,
) -> None:
    """
    EN: Save classification report as txt.
    PT: Salva o classification report em txt.
    """
    report = classification_report(y_true, y_pred, zero_division=0)
    filepath.write_text(report, encoding="utf-8")


def save_confusion_matrix_figure(
    y_true,
    y_pred,
    filepath: Path,
    title: str,
) -> None:
    """
    EN: Save confusion matrix figure.
    PT: Salva a figura da matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def save_feature_importance(
    model,
    feature_names: list[str],
    filepath_csv: Path,
    filepath_png: Path,
    top_n: int = 15,
) -> None:
    """
    EN: Save feature importances to CSV and PNG.
    PT: Salva as importâncias das variáveis em CSV e PNG.
    """
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(filepath_csv, index=False, encoding="utf-8-sig")

    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(filepath_png, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    """
    EN: Main advanced pipeline.
    PT: Pipeline avançado principal.
    """
    _, _, outputs_dir = get_project_paths()

    # 1) Load
    df = load_data()
    print("\n==== RAW DATA ====\n")
    print(df.head().to_string())

    # 2) Split original data first
    x = df.drop("quality", axis=1)
    y = df["quality"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 3) Feature engineering AFTER split
    x_train_fe = add_features(x_train)
    x_test_fe = add_features(x_test)

    print("\n==== TRAIN SHAPE AFTER FEATURE ENGINEERING ====\n")
    print(x_train_fe.shape)
    print("\n==== TEST SHAPE AFTER FEATURE ENGINEERING ====\n")
    print(x_test_fe.shape)

    # 4) Feature selection only using training data
    x_train_sel, x_test_sel, importance_df = select_features_by_importance(
        x_train=x_train_fe,
        y_train=y_train,
        x_test=x_test_fe,
        top_n=12,
        random_state=42
    )

    print("\n==== SELECTED FEATURES ====\n")
    print(x_train_sel.columns.tolist())

    importance_df.to_csv(
        outputs_dir / "feature_selection_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 5) CV strategy
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 6) Base models
    models = {
        "RandomForest_Base": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1
        ),
        "ExtraTrees_Base": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1
        ),
    }

    comparison_rows = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\n\n{'=' * 20} {model_name} {'=' * 20}")

        cv_score = cross_validate_model(
            model=model,
            x_train=x_train_sel,
            y_train=y_train,
            cv=cv,
            scoring="f1_macro"
        )

        result = evaluate_model_train_test(
            model=model,
            x_train=x_train_sel,
            y_train=y_train,
            x_test=x_test_sel,
            y_test=y_test,
            model_name=model_name
        )

        fitted_models[model_name] = result["model"]

        comparison_rows.append({
            "model_name": model_name,
            "cv_f1_macro": cv_score,
            "train_f1_macro": result["train_metrics"]["f1_macro"],
            "test_f1_macro": result["test_metrics"]["f1_macro"],
            "train_accuracy": result["train_metrics"]["accuracy"],
            "test_accuracy": result["test_metrics"]["accuracy"],
        })

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by="test_f1_macro", ascending=False
    )

    print("\n==== BASE MODEL COMPARISON ====\n")
    print(comparison_df.round(4).to_string(index=False))

    comparison_df.to_csv(
        outputs_dir / "base_model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 7) Choose best base model
    best_base_model_name = comparison_df.iloc[0]["model_name"]
    print("\n==== BEST BASE MODEL ====\n")
    print(best_base_model_name)

    # 8) Tune RandomForest (you can switch to the best model later)
    print("\n==== RANDOMIZED SEARCH - RANDOM FOREST ====\n")
    rf_search = tune_random_forest(
        x_train=x_train_sel,
        y_train=y_train,
        cv=cv,
        random_state=42
    )

    print("\nBest params:")
    print(rf_search.best_params_)
    print(f"\nBest CV f1_macro: {rf_search.best_score_:.4f}")

    best_rf = rf_search.best_estimator_

    # 9) Evaluate tuned RF
    best_rf.fit(x_train_sel, y_train)

    y_pred_train = best_rf.predict(x_train_sel)
    y_pred_test = best_rf.predict(x_test_sel)

    tuned_train_metrics = calculate_metrics(y_train, y_pred_train)
    tuned_test_metrics = calculate_metrics(y_test, y_pred_test)

    print_metrics("Tuned RandomForest - Train", tuned_train_metrics)
    print_metrics("Tuned RandomForest - Test", tuned_test_metrics)

    # 10) Save detailed outputs
    save_metrics_json(
        {
            "best_params": rf_search.best_params_,
            "best_cv_f1_macro": rf_search.best_score_,
            "train_metrics": tuned_train_metrics,
            "test_metrics": tuned_test_metrics,
        },
        outputs_dir / "tuned_random_forest_metrics.json"
    )

    save_classification_report(
        y_true=y_train,
        y_pred=y_pred_train,
        filepath=outputs_dir / "classification_report_train.txt"
    )

    save_classification_report(
        y_true=y_test,
        y_pred=y_pred_test,
        filepath=outputs_dir / "classification_report_test.txt"
    )

    save_confusion_matrix_figure(
        y_true=y_train,
        y_pred=y_pred_train,
        filepath=outputs_dir / "confusion_matrix_train.png",
        title="Confusion Matrix - Train"
    )

    save_confusion_matrix_figure(
        y_true=y_test,
        y_pred=y_pred_test,
        filepath=outputs_dir / "confusion_matrix_test.png",
        title="Confusion Matrix - Test"
    )

    save_feature_importance(
        model=best_rf,
        feature_names=x_train_sel.columns.tolist(),
        filepath_csv=outputs_dir / "feature_importance.csv",
        filepath_png=outputs_dir / "feature_importance.png",
        top_n=15
    )

    # 11) Final comparison: best base vs tuned RF
    best_base_row = comparison_df.iloc[0]

    final_comparison = pd.DataFrame([
        {
            "candidate": "Best_Base_Model",
            "model_name": best_base_row["model_name"],
            "cv_f1_macro": best_base_row["cv_f1_macro"],
            "test_f1_macro": best_base_row["test_f1_macro"],
        },
        {
            "candidate": "Tuned_RandomForest",
            "model_name": "Tuned_RandomForest",
            "cv_f1_macro": rf_search.best_score_,
            "test_f1_macro": tuned_test_metrics["f1_macro"],
        }
    ])

    print("\n==== FINAL COMPARISON ====\n")
    print(final_comparison.round(4).to_string(index=False))

    final_comparison.to_csv(
        outputs_dir / "final_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n==== PIPELINE FINISHED SUCCESSFULLY ====\n")
    print(f"Outputs saved in: {outputs_dir}")


if __name__ == "__main__":
    main()