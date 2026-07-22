from __future__ import annotations

from pathlib import Path
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.base import clone


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.float_format", "{:.4f}".format)


# =========================================================
# GLOBAL CONFIG
# CONFIGURAÇÕES GLOBAIS
# =========================================================

RANDOM_STATE = 42
TARGET_COL = "Class"

# Fraud problem cost assumptions.
# Suposições de custo para o problema de fraude.
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 10

# Threshold grid used to search the best decision point.
# Grade de thresholds usada para buscar o melhor ponto de decisão.
THRESHOLDS = np.arange(0.01, 1.00, 0.01)

# Risk zones for the final risk-oriented ensemble.
# Zonas de risco para o ensemble final orientado a risco.
LOW_RISK_THRESHOLD = 0.20
HIGH_RISK_THRESHOLD = 0.70


# =========================================================
# PATHS
# CAMINHOS DO PROJETO
# =========================================================

def get_project_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Return project paths.
    Retorna os caminhos principais do projeto.

    Expected project structure:
    Estrutura esperada do projeto:

    project/
        data/
            creditcard.csv
        outputs/
        src/
            main.py
    """
    current_file = Path(__file__).resolve()
    project_dir = current_file.parent.parent

    data_dir = project_dir / "data"
    outputs_dir = project_dir / "outputs"
    figures_dir = outputs_dir / "figures"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return project_dir, data_dir, outputs_dir, figures_dir


def find_dataset(data_dir: Path) -> Path:
    """
    Try to find the credit card fraud dataset.
    Tenta localizar o dataset de fraude em cartão de crédito.
    """
    possible_paths = [
        data_dir / "creditcard.csv",
        data_dir / "raw" / "creditcard.csv",
        data_dir / "CreditCardFraud.csv",
        data_dir / "credit_card_fraud.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset não encontrado. Coloque o arquivo em um destes caminhos:\n"
        "- data/creditcard.csv\n"
        "- data/raw/creditcard.csv\n"
        "- data/CreditCardFraud.csv\n"
        "- data/credit_card_fraud.csv"
    )


# =========================================================
# DATA LOADING AND BASIC EDA
# CARREGAMENTO DOS DADOS E EDA BÁSICA
# =========================================================

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV.
    Carrega o dataset a partir de um arquivo CSV.
    """
    print("\n==== DATASET ====\n")
    print(f"Carregando dataset de: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print("\nDataset carregado com sucesso.")
    return df


def show_data_overview(df: pd.DataFrame) -> None:
    """
    Print basic dataset information.
    Exibe informações básicas do dataset.
    """
    print("\n\n==== VISÃO GERAL DOS DADOS ====\n")
    print(df.head())

    print("\n\n==== DIMENSÃO DO DATASET ====\n")
    print(df.shape)

    print("\n\n==== TIPOS DE DADOS ====\n")
    print(df.dtypes)

    print("\n\n==== VALORES NULOS ====\n")
    print(df.isna().sum())

    print("\n\n==== ESTATÍSTICAS DESCRITIVAS ====\n")
    print(df.describe())


def show_target_distribution(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Print and return target distribution.
    Exibe e retorna a distribuição da variável alvo.
    """
    print("\n\n==== DISTRIBUIÇÃO DA VARIÁVEL ALVO ====\n")

    target_count = df[target_col].value_counts().sort_index()
    target_proportion = df[target_col].value_counts(normalize=True).sort_index()

    distribution_df = pd.DataFrame(
        {
            "quantidade": target_count,
            "proporcao": target_proportion,
        }
    )

    print(distribution_df)

    fraud_rate = target_proportion.loc[1] if 1 in target_proportion.index else 0
    print(f"\nTaxa de fraude: {fraud_rate:.6f}")

    return distribution_df


def show_class_summary(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> None:
    """
    Show mean values by class.
    Exibe os valores médios por classe.
    """
    print("\n\n==== RESUMO POR CLASSE ====\n")
    print("Média das variáveis por classe:\n")
    print(df.groupby(target_col).mean(numeric_only=True))


# =========================================================
# PLOTS
# GRÁFICOS
# =========================================================

def save_target_distribution_plot(
    df: pd.DataFrame,
    figures_dir: Path,
    target_col: str = TARGET_COL,
) -> None:
    """
    Save target distribution bar chart.
    Salva o gráfico de barras da distribuição da variável alvo.
    """
    counts = df[target_col].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Distribuição da Variável Alvo")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")
    plt.tight_layout()

    output_path = figures_dir / "target_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nGráfico da distribuição da variável alvo salvo em: {output_path}")


def save_amount_distribution_by_class(
    df: pd.DataFrame,
    figures_dir: Path,
    amount_col: str = "Amount",
    target_col: str = TARGET_COL,
) -> None:
    """
    Save amount distribution by class using log scale.
    Salva a distribuição do valor da transação por classe usando escala logarítmica.
    """
    plt.figure(figsize=(10, 5))

    for class_value in sorted(df[target_col].unique()):
        subset = df.loc[df[target_col] == class_value, amount_col]
        plt.hist(
            subset,
            bins=60,
            alpha=0.5,
            label=f"Classe {class_value}",
        )

    plt.title("Distribuição do Valor da Transação por Classe")
    plt.xlabel("Valor da Transação")
    plt.ylabel("Frequência")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    output_path = figures_dir / "amount_distribution_by_class.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Gráfico da distribuição do valor por classe salvo em: {output_path}")


def save_precision_recall_curve_plot(
    y_true: pd.Series,
    y_proba: np.ndarray,
    figures_dir: Path,
    model_name: str,
) -> None:
    """
    Save Precision-Recall curve.
    Salva a curva Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label=f"Average Precision = {avg_precision:.4f}")
    plt.title(f"Curva Precision-Recall - {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_path = figures_dir / f"precision_recall_curve_{safe_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Curva Precision-Recall salva em: {output_path}")


def save_roc_curve_plot(
    y_true: pd.Series,
    y_proba: np.ndarray,
    figures_dir: Path,
    model_name: str,
) -> None:
    """
    Save ROC curve.
    Salva a curva ROC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"Curva ROC - {model_name}")
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.legend()
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_path = figures_dir / f"roc_curve_{safe_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Curva ROC salva em: {output_path}")


# =========================================================
# DATA SPLIT AND PREPROCESSING
# SEPARAÇÃO DOS DADOS E PRÉ-PROCESSAMENTO
# =========================================================

def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation and test sets.
    Separa os dados em treino, validação e teste.

    Final split:
    Divisão final:

    - 60% train / treino
    - 20% validation / validação
    - 20% test / teste
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\n\n==== SEPARAÇÃO DOS DADOS ====\n")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_valid: {X_valid.shape} | y_valid: {y_valid.shape}")
    print(f"X_test : {X_test.shape} | y_test : {y_test.shape}")

    print("\nProporção da variável alvo no treino:")
    print(y_train.value_counts(normalize=True).sort_index())

    print("\nProporção da variável alvo na validação:")
    print(y_valid.value_counts(normalize=True).sort_index())

    print("\nProporção da variável alvo no teste:")
    print(y_test.value_counts(normalize=True).sort_index())

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def scale_time_and_amount(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale only Time and Amount.
    Escala apenas as colunas Time e Amount.

    The V1-V28 variables are already PCA components.
    As variáveis V1-V28 já são componentes PCA.

    Time and Amount are not PCA transformed, so they should be scaled.
    Time e Amount não foram transformadas por PCA, por isso serão escaladas.
    """
    columns_to_scale = ["Time", "Amount"]

    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    X_test_scaled = X_test.copy()

    scaler = StandardScaler()

    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_valid_scaled[columns_to_scale] = scaler.transform(X_valid[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    print("\n\n==== PRÉ-PROCESSAMENTO ====\n")
    print("Colunas escaladas: Time, Amount")
    print("Colunas PCA V1-V28 mantidas sem alteração.")

    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


# =========================================================
# MODELING
# MODELAGEM
# =========================================================

def get_base_models() -> Dict[str, object]:
    """
    Create base models.
    Cria os modelos base.

    These models represent different classification strategies.
    Estes modelos representam diferentes estratégias de classificação.
    """
    models = {
        "Logistic Regression Balanced": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Decision Tree Balanced": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest Balanced": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees Balanced": ExtraTreesClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }

    return models


def fit_models(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, object]:
    """
    Fit all models.
    Treina todos os modelos.
    """
    fitted_models = {}

    print("\n\n==== TREINAMENTO DOS MODELOS ====\n")

    for model_name, model in models.items():
        print(f"Treinando modelo: {model_name}")

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        fitted_models[model_name] = fitted_model

    print("\nTodos os modelos foram treinados com sucesso.")

    return fitted_models


def get_positive_class_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    """
    Return probability for positive class.
    Retorna a probabilidade da classe positiva.

    Some classifiers may not have predict_proba.
    Alguns classificadores podem não possuir predict_proba.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min())

    raise AttributeError("O modelo não possui predict_proba nem decision_function.")


# =========================================================
# EVALUATION
# AVALIAÇÃO
# =========================================================

def calculate_binary_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
    strategy: str,
    false_positive_cost: int = FALSE_POSITIVE_COST,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
) -> Dict[str, float]:
    """
    Calculate binary classification metrics using a custom threshold.
    Calcula métricas de classificação binária usando um threshold personalizado.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fp * false_positive_cost) + (fn * false_negative_cost)

    metrics = {
        "modelo": model_name,
        "estrategia": strategy,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fraude": precision_score(y_true, y_pred, zero_division=0),
        "recall_fraude": recall_score(y_true, y_pred, zero_division=0),
        "f1_fraude": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "taxa_falso_positivo": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "taxa_falso_negativo": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "custo_total": total_cost,
    }

    return metrics


def find_best_thresholds_for_model(
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """
    Search thresholds according to different strategies.
    Busca thresholds conforme diferentes estratégias.

    Strategies:
    Estratégias:

    - Max Recall with minimum precision constraint.
      Máximo recall com restrição mínima de precision.

    - Max Precision with minimum recall constraint.
      Máxima precision com restrição mínima de recall.

    - Max F1-score.
      Máximo F1-score.

    - Min Cost.
      Menor custo estimado.
    """
    rows = []

    for threshold in THRESHOLDS:
        metrics = calculate_binary_metrics(
            y_true=y_true,
            y_proba=y_proba,
            threshold=float(threshold),
            model_name=model_name,
            strategy="Busca de Threshold",
        )
        rows.append(metrics)

    threshold_df = pd.DataFrame(rows)

    # Recall-oriented strategy.
    # Estratégia orientada para recall.
    recall_candidates = threshold_df[threshold_df["precision_fraude"] >= 0.05].copy()

    if len(recall_candidates) == 0:
        best_recall = threshold_df.sort_values(
            by=["recall_fraude", "precision_fraude"],
            ascending=[False, False],
        ).head(1)
    else:
        best_recall = recall_candidates.sort_values(
            by=["recall_fraude", "precision_fraude"],
            ascending=[False, False],
        ).head(1)

    best_recall = best_recall.copy()
    best_recall["estrategia"] = "Orientado para Recall"

    # Precision-oriented strategy.
    # Estratégia orientada para precision.
    precision_candidates = threshold_df[threshold_df["recall_fraude"] >= 0.20].copy()

    if len(precision_candidates) == 0:
        best_precision = threshold_df.sort_values(
            by=["precision_fraude", "recall_fraude"],
            ascending=[False, False],
        ).head(1)
    else:
        best_precision = precision_candidates.sort_values(
            by=["precision_fraude", "recall_fraude"],
            ascending=[False, False],
        ).head(1)

    best_precision = best_precision.copy()
    best_precision["estrategia"] = "Orientado para Precision"

    # F1-oriented strategy.
    # Estratégia orientada para F1-score.
    best_f1 = threshold_df.sort_values(
        by=["f1_fraude", "recall_fraude"],
        ascending=[False, False],
    ).head(1)

    best_f1 = best_f1.copy()
    best_f1["estrategia"] = "Orientado para F1"

    # Cost-oriented strategy.
    # Estratégia orientada para menor custo.
    best_cost = threshold_df.sort_values(
        by=["custo_total", "recall_fraude"],
        ascending=[True, False],
    ).head(1)

    best_cost = best_cost.copy()
    best_cost["estrategia"] = "Orientado para Custo"

    best_thresholds = pd.concat(
        [best_recall, best_precision, best_f1, best_cost],
        axis=0,
        ignore_index=True,
    )

    return best_thresholds


# =========================================================
# CUSTOM RISK-ORIENTED ENSEMBLE
# ENSEMBLE PERSONALIZADO ORIENTADO A RISCO
# =========================================================

def build_probability_matrix(
    fitted_models: Dict[str, object],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a matrix with fraud probabilities from all fitted models.
    Cria uma matriz com as probabilidades de fraude de todos os modelos treinados.
    """
    proba_dict = {}

    for model_name, model in fitted_models.items():
        proba_dict[model_name] = get_positive_class_proba(model, X)

    proba_df = pd.DataFrame(proba_dict, index=X.index)

    return proba_df


def calculate_model_weights_from_validation(
    fitted_models: Dict[str, object],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Dict[str, float]:
    """
    Calculate model weights based on Average Precision in validation set.
    Calcula os pesos dos modelos com base no Average Precision no conjunto de validação.
    """
    scores = {}

    for model_name, model in fitted_models.items():
        y_proba = get_positive_class_proba(model, X_valid)
        ap_score = average_precision_score(y_valid, y_proba)
        scores[model_name] = ap_score

    total_score = sum(scores.values())

    if total_score == 0:
        weights = {model_name: 1 / len(scores) for model_name in scores}
    else:
        weights = {
            model_name: score / total_score
            for model_name, score in scores.items()
        }

    print("\n\n==== PESOS DOS MODELOS NO ENSEMBLE PERSONALIZADO ====\n")
    for model_name, weight in weights.items():
        print(f"{model_name}: {weight:.4f}")

    return weights


def calculate_custom_risk_score(
    proba_df: pd.DataFrame,
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Calculate weighted fraud risk score.
    Calcula o score ponderado de risco de fraude.
    """
    risk_score = np.zeros(len(proba_df))

    for model_name, weight in weights.items():
        risk_score += proba_df[model_name].values * weight

    return risk_score


def classify_risk_zone(
    risk_score: np.ndarray,
    low_threshold: float = LOW_RISK_THRESHOLD,
    high_threshold: float = HIGH_RISK_THRESHOLD,
) -> np.ndarray:
    """
    Convert numeric risk score into risk zones.
    Converte o score numérico de risco em zonas de risco.

    Low Risk:
    Baixo Risco:
        Transaction can be approved.
        A transação pode ser aprovada.

    Medium Risk:
    Risco Médio:
        Transaction should be sent to manual review or extra authentication.
        A transação deve ir para revisão manual ou autenticação extra.

    High Risk:
    Alto Risco:
        Transaction should be blocked or flagged as fraud.
        A transação deve ser bloqueada ou sinalizada como fraude.
    """
    zones = np.where(
        risk_score < low_threshold,
        "Baixo Risco",
        np.where(risk_score < high_threshold, "Risco Médio", "Alto Risco"),
    )

    return zones


def evaluate_custom_risk_ensemble(
    fitted_models: Dict[str, object],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate custom risk-oriented ensemble.
    Avalia o ensemble personalizado orientado a risco.
    """
    print("\n\n==== ENSEMBLE PERSONALIZADO ORIENTADO A RISCO ====\n")

    weights = calculate_model_weights_from_validation(
        fitted_models=fitted_models,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    valid_proba_df = build_probability_matrix(fitted_models, X_valid)
    valid_risk_score = calculate_custom_risk_score(valid_proba_df, weights)

    custom_validation_thresholds = find_best_thresholds_for_model(
        y_true=y_valid,
        y_proba=valid_risk_score,
        model_name="Custom Risk-Oriented Ensemble",
    )

    print("\nMelhores thresholds do ensemble personalizado na validação:\n")
    print(custom_validation_thresholds)

    test_proba_df = build_probability_matrix(fitted_models, X_test)
    test_risk_score = calculate_custom_risk_score(test_proba_df, weights)

    test_results = []

    for _, row in custom_validation_thresholds.iterrows():
        threshold = row["threshold"]
        strategy = row["estrategia"]

        metrics = calculate_binary_metrics(
            y_true=y_test,
            y_proba=test_risk_score,
            threshold=float(threshold),
            model_name="Custom Risk-Oriented Ensemble",
            strategy=strategy,
        )

        test_results.append(metrics)

    risk_zones = classify_risk_zone(test_risk_score)

    high_risk_as_fraud = (risk_zones == "Alto Risco").astype(int)
    medium_or_high_as_fraud = np.isin(risk_zones, ["Risco Médio", "Alto Risco"]).astype(int)

    policy_predictions = {
        "Política - Apenas Alto Risco": high_risk_as_fraud,
        "Política - Risco Médio ou Alto": medium_or_high_as_fraud,
    }

    for policy_name, y_pred_policy in policy_predictions.items():
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_policy).ravel()
        total_cost = (fp * FALSE_POSITIVE_COST) + (fn * FALSE_NEGATIVE_COST)

        metrics = {
            "modelo": "Custom Risk-Oriented Ensemble",
            "estrategia": policy_name,
            "threshold": np.nan,
            "accuracy": accuracy_score(y_test, y_pred_policy),
            "precision_fraude": precision_score(y_test, y_pred_policy, zero_division=0),
            "recall_fraude": recall_score(y_test, y_pred_policy, zero_division=0),
            "f1_fraude": f1_score(y_test, y_pred_policy, zero_division=0),
            "roc_auc": roc_auc_score(y_test, test_risk_score),
            "average_precision": average_precision_score(y_test, test_risk_score),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "taxa_falso_positivo": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "taxa_falso_negativo": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "custo_total": total_cost,
        }

        test_results.append(metrics)

    custom_test_results_df = pd.DataFrame(test_results)
    custom_test_results_df = custom_test_results_df.sort_values(
        by=["f1_fraude", "average_precision"],
        ascending=[False, False],
    )

    risk_zone_df = pd.DataFrame(
        {
            "score_risco": test_risk_score,
            "zona_risco": risk_zones,
            "classe_real": y_test.values,
        },
        index=y_test.index,
    )

    print("\nResultados do Ensemble Personalizado Orientado a Risco no teste:\n")
    print(custom_test_results_df)

    print("\nDistribuição das zonas de risco:\n")
    print(risk_zone_df["zona_risco"].value_counts())

    print("\nZona de risco x classe real:")
    print(pd.crosstab(risk_zone_df["zona_risco"], risk_zone_df["classe_real"]))

    return custom_test_results_df, risk_zone_df


def save_final_ensemble_outputs(
    outputs_dir: Path,
    y_test: pd.Series,
    risk_zone_df: pd.DataFrame,
    custom_test_results_df: pd.DataFrame,
) -> None:
    """
    Save final ensemble outputs as a single final model.
    Salva as saídas finais do ensemble como se fosse um único modelo final.
    """
    print("\n\n==== SALVANDO SAÍDA FINAL DO ENSEMBLE ====\n")

    # Select the final decision policy.
    # Selecionar a política de decisão final.
    final_strategy = "Política - Apenas Alto Risco"

    final_metrics_df = custom_test_results_df[
        custom_test_results_df["estrategia"] == final_strategy
    ].copy()

    if final_metrics_df.empty:
        raise ValueError(
            f"A estratégia final '{final_strategy}' não foi encontrada nos resultados do ensemble."
        )

    final_metrics_df = final_metrics_df.rename(
        columns={
            "modelo": "modelo_final",
            "estrategia": "politica_decisao",
        }
    )

    # Save final metrics.
    # Salvar métricas finais.
    final_metrics_path = outputs_dir / "final_ensemble_metrics.csv"
    final_metrics_df.to_csv(final_metrics_path, index=False)

    print(f"Métricas finais do ensemble salvas em: {final_metrics_path}")

    # Create final predictions.
    # Criar predições finais.
    final_predictions_df = risk_zone_df.copy()
    final_predictions_df["classe_predita"] = (
        final_predictions_df["zona_risco"] == "Alto Risco"
    ).astype(int)

    final_predictions_df = final_predictions_df.reset_index(names="indice_original")

    final_predictions_path = outputs_dir / "final_ensemble_predictions.csv"
    final_predictions_df.to_csv(final_predictions_path, index=False)

    print(f"Predições finais do ensemble salvas em: {final_predictions_path}")

    # Create confusion matrix as a readable table.
    # Criar matriz de confusão em formato legível.
    cm = confusion_matrix(
        final_predictions_df["classe_real"],
        final_predictions_df["classe_predita"],
    )

    confusion_matrix_df = pd.DataFrame(
        cm,
        index=["Real Não Fraude", "Real Fraude"],
        columns=["Predito Não Fraude", "Predito Fraude"],
    )

    confusion_matrix_path = outputs_dir / "final_ensemble_confusion_matrix.csv"
    confusion_matrix_df.to_csv(confusion_matrix_path)

    print(f"Matriz de confusão final salva em: {confusion_matrix_path}")

    print("\nMétricas finais do ensemble:")
    print(final_metrics_df)

    print("\nMatriz de confusão final:")
    print(confusion_matrix_df)


# =========================================================
# OUTPUTS
# SAÍDAS
# =========================================================

def save_results(
    outputs_dir: Path,
    **dataframes: pd.DataFrame,
) -> None:
    """
    Save result DataFrames to CSV.
    Salva os DataFrames de resultados em arquivos CSV.
    """
    print("\n\n==== SALVANDO RESULTADOS ====\n")

    for name, df in dataframes.items():
        output_path = outputs_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Arquivo salvo: {output_path}")


# =========================================================
# MAIN
# EXECUÇÃO PRINCIPAL
# =========================================================

def main() -> None:
    """
    Main project pipeline.
    Pipeline principal do projeto.
    """
    start_time = time.time()

    project_dir, data_dir, outputs_dir, figures_dir = get_project_paths()

    print("\n==== CAMINHOS DO PROJETO ====\n")
    print(f"Pasta do projeto : {project_dir}")
    print(f"Pasta dos dados  : {data_dir}")
    print(f"Pasta de saídas  : {outputs_dir}")
    print(f"Pasta de figuras : {figures_dir}")

    dataset_path = find_dataset(data_dir)

    # Load and inspect data.
    # Carregar e inspecionar os dados.
    df = load_dataset(dataset_path)

    show_data_overview(df)
    target_distribution_df = show_target_distribution(df)
    show_class_summary(df)

    # Save basic plots.
    # Salvar gráficos básicos.
    save_target_distribution_plot(df, figures_dir)
    save_amount_distribution_by_class(df, figures_dir)

    # Split and preprocess.
    # Separar e pré-processar os dados.
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

    X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_time_and_amount(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
    )

    # Train base models.
    # Treinar modelos base.
    base_models = get_base_models()
    fitted_models = fit_models(base_models, X_train_scaled, y_train)

    # Train and evaluate the final custom risk-oriented ensemble.
    # Treinar e avaliar o ensemble final personalizado orientado a risco.
    custom_test_results_df, risk_zone_df = evaluate_custom_risk_ensemble(
        fitted_models=fitted_models,
        X_valid=X_valid_scaled,
        y_valid=y_valid,
        X_test=X_test_scaled,
        y_test=y_test,
    )

    # Save final ensemble outputs as a single final model.
    # Salvar as saídas finais do ensemble como um único modelo final.
    save_final_ensemble_outputs(
        outputs_dir=outputs_dir,
        y_test=y_test,
        risk_zone_df=risk_zone_df,
        custom_test_results_df=custom_test_results_df,
    )

    # Build custom risk score for final curves.
    # Criar score de risco personalizado para as curvas finais.
    custom_weights = calculate_model_weights_from_validation(
        fitted_models=fitted_models,
        X_valid=X_valid_scaled,
        y_valid=y_valid,
    )

    test_proba_df = build_probability_matrix(fitted_models, X_test_scaled)
    test_risk_score = calculate_custom_risk_score(test_proba_df, custom_weights)

    save_precision_recall_curve_plot(
        y_true=y_test,
        y_proba=test_risk_score,
        figures_dir=figures_dir,
        model_name="Custom Risk-Oriented Ensemble",
    )

    save_roc_curve_plot(
        y_true=y_test,
        y_proba=test_risk_score,
        figures_dir=figures_dir,
        model_name="Custom Risk-Oriented Ensemble",
    )

    # Save additional outputs.
    # Salvar saídas adicionais.
    save_results(
        outputs_dir=outputs_dir,
        target_distribution=target_distribution_df.reset_index(names="classe"),
        test_custom_ensemble_results=custom_test_results_df,
        risk_zone_results=risk_zone_df.reset_index(names="indice_original"),
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n\n==== EXECUÇÃO FINALIZADA ====\n")
    print(f"Tempo total de execução: {elapsed_time:.2f} segundos")


if __name__ == "__main__":
    main()
