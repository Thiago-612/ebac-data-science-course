from __future__ import annotations
from pathlib import Path
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from statsmodels.stats.outliers_influence import variance_inflation_factor

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

    df = pd.read_csv(data_dir / 'ALUGUEL_MOD12.csv',
                     sep=';',
                     encoding='utf-8',
                     na_values=['', ' ', 'NA', 'None']
                    )
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


# =========================================================
# DATA CLASS
# =========================================================
@dataclass
class ModelResult:
    model_name: str
    degree: int
    best_params: dict
    mse_train: float
    rmse_train: float
    r2_train: float
    mse_test: float
    rmse_test: float
    r2_test: float
    vif_max: float
    vif_mean: float


# =========================================================
# METRICS
# =========================================================
def calculate_regression_metrics(
    y_train: pd.Series,
    y_pred_train: np.ndarray,
    y_test: pd.Series,
    y_pred_test: np.ndarray,
) -> dict:
    """
    EN: Compute train and test regression metrics.
    PT: Calcula métricas de regressão para treino e teste.
    """
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    return {
        "mse_train": mse_train,
        "rmse_train": np.sqrt(mse_train),
        "r2_train": r2_score(y_train, y_pred_train),
        "mse_test": mse_test,
        "rmse_test": np.sqrt(mse_test),
        "r2_test": r2_score(y_test, y_pred_test),
    }


# =========================================================
# VIF
# =========================================================
def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    EN: Calculate VIF for each feature.
    PT: Calcula o VIF para cada variável.
    """
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)


def build_polynomial_dataframe(
    X: pd.DataFrame,
    degree: int,
) -> pd.DataFrame:
    """
    EN: Create polynomial features as DataFrame.
    PT: Cria as variáveis polinomiais em formato DataFrame.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    feature_names = poly.get_feature_names_out(X.columns)

    return pd.DataFrame(X_poly, columns=feature_names, index=X.index)


# =========================================================
# TRAINING
# =========================================================
def train_and_evaluate_model(
    model_name: str,
    estimator,
    param_grid: dict,
    degree: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_splits: int = 3,
) -> tuple[ModelResult, pd.DataFrame, object]:
    """
    EN: Train a polynomial regression model and evaluate it.
    PT: Treina um modelo de regressão polinomial e o avalia.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", estimator)
    ])

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics = calculate_regression_metrics(
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test
    )

    X_poly_df = build_polynomial_dataframe(X_train, degree)
    vif_df = calculate_vif(X_poly_df)

    result = ModelResult(
        model_name=model_name,
        degree=degree,
        best_params=search.best_params_,
        mse_train=metrics["mse_train"],
        rmse_train=metrics["rmse_train"],
        r2_train=metrics["r2_train"],
        mse_test=metrics["mse_test"],
        rmse_test=metrics["rmse_test"],
        r2_test=metrics["r2_test"],
        vif_max=float(vif_df["VIF"].max()),
        vif_mean=float(vif_df["VIF"].mean()),
    )

    return result, vif_df, best_model


# =========================================================
# MAIN PIPELINE
# =========================================================
def polynomial_regression_pipeline(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    degrees: list[int] = [1, 2, 3],
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    EN: Full polynomial regression pipeline.
    PT: Pipeline completo de regressão polinomial.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    results = []
    vif_tables = {}
    fitted_models = {}

    for degree in degrees:
        print(f"\n{'=' * 25} GRAU {degree} {'=' * 25}")

        # -------------------------------------------------
        # Linear Regression
        # -------------------------------------------------
        result, vif_df, best_model = train_and_evaluate_model(
            model_name="LinearRegression",
            estimator=LinearRegression(),
            param_grid={},
            degree=degree,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result.__dict__)
        vif_tables[f"LinearRegression_degree_{degree}"] = vif_df
        fitted_models[f"LinearRegression_degree_{degree}"] = best_model

        print(f"\nLinearRegression | Grau {degree}")
        print(f"RMSE treino: {result.rmse_train:.4f}")
        print(f"RMSE teste : {result.rmse_test:.4f}")
        print(f"R² treino  : {result.r2_train:.4f}")
        print(f"R² teste   : {result.r2_test:.4f}")
        print(f"VIF máximo : {result.vif_max:.2f}")

        # -------------------------------------------------
        # Ridge
        # -------------------------------------------------
        result, vif_df, best_model = train_and_evaluate_model(
            model_name="Ridge",
            estimator=Ridge(random_state=42),
            param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            degree=degree,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result.__dict__)
        vif_tables[f"Ridge_degree_{degree}"] = vif_df
        fitted_models[f"Ridge_degree_{degree}"] = best_model

        print(f"\nRidge | Grau {degree}")
        print(f"Best params: {result.best_params}")
        print(f"RMSE treino: {result.rmse_train:.4f}")
        print(f"RMSE teste : {result.rmse_test:.4f}")
        print(f"R² treino  : {result.r2_train:.4f}")
        print(f"R² teste   : {result.r2_test:.4f}")
        print(f"VIF máximo : {result.vif_max:.2f}")

        # -------------------------------------------------
        # Lasso
        # -------------------------------------------------
        result, vif_df, best_model = train_and_evaluate_model(
            model_name="Lasso",
            estimator=Lasso(random_state=42, max_iter=100000),
            param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0]},
            degree=degree,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result.__dict__)
        vif_tables[f"Lasso_degree_{degree}"] = vif_df
        fitted_models[f"Lasso_degree_{degree}"] = best_model

        print(f"\nLasso | Grau {degree}")
        print(f"Best params: {result.best_params}")
        print(f"RMSE treino: {result.rmse_train:.4f}")
        print(f"RMSE teste : {result.rmse_test:.4f}")
        print(f"R² treino  : {result.r2_train:.4f}")
        print(f"R² teste   : {result.r2_test:.4f}")
        print(f"VIF máximo : {result.vif_max:.2f}")

        # -------------------------------------------------
        # ElasticNet
        # -------------------------------------------------
        result, vif_df, best_model = train_and_evaluate_model(
            model_name="ElasticNet",
            estimator=ElasticNet(random_state=42, max_iter=100000),
            param_grid={
                "model__alpha": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.2, 0.5, 0.8]
            },
            degree=degree,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result.__dict__)
        vif_tables[f"ElasticNet_degree_{degree}"] = vif_df
        fitted_models[f"ElasticNet_degree_{degree}"] = best_model

        print(f"\nElasticNet | Grau {degree}")
        print(f"Best params: {result.best_params}")
        print(f"RMSE treino: {result.rmse_train:.4f}")
        print(f"RMSE teste : {result.rmse_test:.4f}")
        print(f"R² treino  : {result.r2_train:.4f}")
        print(f"R² teste   : {result.r2_test:.4f}")
        print(f"VIF máximo : {result.vif_max:.2f}")

    results_df = pd.DataFrame(results).sort_values(
        by=["rmse_test", "r2_test"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return results_df, vif_tables, fitted_models


# =========================================================
# PLOTS
# =========================================================
def plot_rmse_comparison(results_df: pd.DataFrame) -> None:
    """
    EN: Plot RMSE comparison for test set.
    PT: Plota comparação de RMSE no conjunto de teste.
    """
    plot_df = results_df.copy()
    plot_df["label"] = plot_df["model_name"] + " | grau " + plot_df["degree"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["label"], plot_df["rmse_test"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Comparação de RMSE no Teste")
    plt.ylabel("RMSE")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.show()


def plot_overfitting(results_df: pd.DataFrame) -> None:
    """
    EN: Plot train vs test RMSE.
    PT: Plota RMSE de treino vs teste para avaliar overfitting.
    """
    plot_df = results_df.copy()
    plot_df["label"] = plot_df["model_name"] + " | grau " + plot_df["degree"].astype(str)

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, plot_df["rmse_train"], width, label="Treino")
    plt.bar(x + width / 2, plot_df["rmse_test"], width, label="Teste")

    plt.xticks(x, plot_df["label"], rotation=45, ha="right")
    plt.title("RMSE Treino vs Teste")
    plt.ylabel("RMSE")
    plt.xlabel("Modelo")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_real_vs_predicted(
    fitted_models: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    best_model_key: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> None:
    """
    EN: Plot actual vs predicted using the best model.
    PT: Plota valores reais vs preditos usando o melhor modelo.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    best_model = fitted_models[best_model_key]
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Valor real")
    plt.ylabel("Valor predito")
    plt.title(f"Real vs Predito - {best_model_key}")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    EN: Main advanced pipeline.
    PT: Pipeline avançado principal.
    """
    _, _, outputs_dir = get_project_paths()

    df = load_data()
    describe_data(df)

    target_col = "Valor_Aluguel"
    feature_cols = ["Metragem", "Valor_Condominio", "N_Quartos", "N_banheiros", "N_Suites", "N_Vagas"]

    results_df, vif_tables, fitted_models = polynomial_regression_pipeline(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        degrees=[1, 2, 4]
    )

    print("\n==== RESULTADOS FINAIS ====\n")
    print(results_df.to_string(index=False))

    best_model_row = results_df.iloc[0]
    print("\n==== MELHOR MODELO ====\n")
    print(best_model_row)

    best_model_key = f"{best_model_row['model_name']}_degree_{best_model_row['degree']}"
    print("\nChave do melhor modelo:", best_model_key)

    print("\n==== VIF DO MELHOR MODELO ====\n")
    print(vif_tables[best_model_key].to_string(index=False))

    plot_rmse_comparison(results_df)
    plot_overfitting(results_df)
    plot_real_vs_predicted(
        fitted_models=fitted_models,
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        best_model_key=best_model_key
    )

if __name__ == "__main__":
    main()