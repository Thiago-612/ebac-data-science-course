from __future__ import annotations
from pathlib import Path
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


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
        data_dir / "ALUGUEL_MOD12.csv",
        sep=";",
        encoding="utf-8",
        na_values=["", " ", "NA", "None"]
    )
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


# =========================================================
# DATA CLASS
# =========================================================
@dataclass
class ModelResult:
    model_name: str
    degree: int
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
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)


def build_polynomial_dataframe(X: pd.DataFrame, degree: int) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    feature_names = poly.get_feature_names_out(X.columns)

    return pd.DataFrame(X_poly, columns=feature_names, index=X.index)


# =========================================================
# TRAIN ONE MODEL
# =========================================================
def train_and_evaluate_fixed_model(
    model_name: str,
    estimator,
    degree: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    vif_max: float,
    vif_mean: float,
) -> tuple[ModelResult, object]:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    metrics = calculate_regression_metrics(
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test
    )

    result = ModelResult(
        model_name=model_name,
        degree=degree,
        mse_train=metrics["mse_train"],
        rmse_train=metrics["rmse_train"],
        r2_train=metrics["r2_train"],
        mse_test=metrics["mse_test"],
        rmse_test=metrics["rmse_test"],
        r2_test=metrics["r2_test"],
        vif_max=vif_max,
        vif_mean=vif_mean,
    )

    return result, pipeline


def compute_vif_by_degree(X_train: pd.DataFrame, degree: int) -> pd.DataFrame:
    X_poly_df = build_polynomial_dataframe(X_train, degree)
    return calculate_vif(X_poly_df)


# =========================================================
# REDUCED PIPELINE
# =========================================================
def reduced_polynomial_pipeline(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict, dict]:
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

    # =========================
    # VIF calculado uma vez por grau
    # =========================
    degrees_to_check = [1, 2, 4]

    for degree in degrees_to_check:
        print(f"\nCalculando VIF do grau {degree}...")
        vif_df = compute_vif_by_degree(X_train, degree)
        vif_tables[f"degree_{degree}"] = vif_df

    tests = [
        ("LinearRegression", 1, LinearRegression()),
        ("LinearRegression", 2, LinearRegression()),
        ("LinearRegression", 4, LinearRegression()),
        ("Ridge", 1, Ridge(alpha=100.0, random_state=42)),
        ("Lasso", 1, Lasso(alpha=10.0, max_iter=100000, random_state=42)),
        ("ElasticNet", 1, ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=100000, random_state=42)),
        ("Ridge", 2, Ridge(alpha=100.0, random_state=42)),
        ("Lasso", 2, Lasso(alpha=10.0, max_iter=100000, random_state=42)),
        ("ElasticNet", 2, ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=100000, random_state=42)),
        ("Ridge", 4, Ridge(alpha=100.0, random_state=42)),
        ("Lasso", 4, Lasso(alpha=10.0, max_iter=100000, random_state=42)),
        ("ElasticNet", 4, ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=100000, random_state=42)),
    ]

    for model_name, degree, estimator in tests:
        print(f"\n{'=' * 20} {model_name} | GRAU {degree} {'=' * 20}")

        vif_df = vif_tables[f"degree_{degree}"]
        vif_max = float(vif_df["VIF"].max())
        vif_mean = float(vif_df["VIF"].mean())

        result, fitted_model = train_and_evaluate_fixed_model(
            model_name=model_name,
            estimator=estimator,
            degree=degree,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            vif_max=vif_max,
            vif_mean=vif_mean,
        )

        results.append(result.__dict__)
        fitted_models[f"{model_name}_degree_{degree}"] = fitted_model

        print(f"RMSE treino: {result.rmse_train:.4f}")
        print(f"RMSE teste : {result.rmse_test:.4f}")
        print(f"R² treino  : {result.r2_train:.4f}")
        print(f"R² teste   : {result.r2_test:.4f}")
        print(f"VIF máximo : {result.vif_max:.2f}")
        print(f"VIF médio  : {result.vif_mean:.2f}")

    results_df = pd.DataFrame(results).sort_values(
        by=["rmse_test", "r2_test"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return results_df, vif_tables, fitted_models


# =========================================================
# PLOTS
# =========================================================
def plot_rmse_comparison(results_df: pd.DataFrame) -> None:
    plot_df = results_df.copy()
    plot_df["label"] = plot_df["model_name"] + " | grau " + plot_df["degree"].astype(str)

    fig = px.bar(
        plot_df,
        x="label",
        y="rmse_test",
        title="Comparação de RMSE no Teste",
        text="rmse_test"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Modelo",
        yaxis_title="RMSE",
        width=1100,
        height=600
    )

    fig.show()


def plot_overfitting(results_df: pd.DataFrame) -> None:
    plot_df = results_df.copy()
    plot_df["label"] = plot_df["model_name"] + " | grau " + plot_df["degree"].astype(str)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["label"],
            y=plot_df["rmse_train"],
            name="Treino"
        )
    )

    fig.add_trace(
        go.Bar(
            x=plot_df["label"],
            y=plot_df["rmse_test"],
            name="Teste"
        )
    )

    fig.update_layout(
        title="RMSE Treino vs Teste",
        xaxis_title="Modelo",
        yaxis_title="RMSE",
        barmode="group",
        width=1100,
        height=600
    )

    fig.show()


def main() -> None:
    df = load_data()
    describe_data(df)

    target_col = "Valor_Aluguel"
    feature_cols = ["Metragem", "Valor_Condominio", "N_Quartos", "N_banheiros", "N_Suites", "N_Vagas"]

    results_df, vif_tables, fitted_models = reduced_polynomial_pipeline(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols
    )

    print("\n==== RESULTADOS FINAIS ====\n")
    print(results_df.to_string(index=False))

    print("\n==== VIF GRAU 1 ====\n")
    print(vif_tables["degree_1"].head(15).to_string(index=False))

    print("\n==== VIF GRAU 2 ====\n")
    print(vif_tables["degree_2"].head(15).to_string(index=False))

    print("\n==== VIF GRAU 4 ====\n")
    print(vif_tables["degree_4"].head(15).to_string(index=False))

    plot_rmse_comparison(results_df)
    plot_overfitting(results_df)


if __name__ == "__main__":
    main()