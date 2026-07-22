from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

import plotly.figure_factory as ff


# =========================
# Configurações
# =========================
MAP_CLASSES = {0: "Alto", 1: "Baixo", 2: "Médio"}
CLASS_ORDER = [0, 1, 2]
CLASS_NAMES = [MAP_CLASSES[c] for c in CLASS_ORDER]


def plot_confusion(cm, title):
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predito",
        yaxis_title="Real",
        template="plotly_white"
    )
    fig.show()


def load_data():
    # raiz do projeto = 1 nível acima de /src
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    x_train = pd.read_csv(data_dir / "x_train_bal.csv")
    y_train = pd.read_csv(data_dir / "y_train_bal.csv").squeeze()

    x_test = pd.read_csv(data_dir / "x_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    return x_train, y_train, x_test, y_test


def main():
    # =========================
    # Leitura dos dados
    # =========================
    x_train, y_train, x_test, y_test = load_data()

    print("\nVERIFICAÇÃO DAS BASES DE TREINO E TESTE:\n")
    print("Tamanho do x_train:", x_train.shape)
    print("Tamanho do y_train:", y_train.shape)
    print("Tamanho do x_test :", x_test.shape)
    print("Tamanho do y_test :", y_test.shape)

    print("\nBalanceamento em y_train:")
    print(y_train.value_counts().sort_index())

    print("\nBalanceamento em y_test:")
    print(y_test.value_counts().sort_index())

    # =========================
    # Modelo
    # =========================
    modelo = GaussianNB()
    modelo.fit(x_train, y_train)

    # =========================
    # TREINO
    # =========================
    y_pred_train = modelo.predict(x_train)

    acc_train = accuracy_score(y_train, y_pred_train)
    recall_train_macro = recall_score(y_train, y_pred_train, average="macro", labels=CLASS_ORDER)
    recall_train_weighted = recall_score(y_train, y_pred_train, average="weighted", labels=CLASS_ORDER)
    recall_train_per_class = recall_score(y_train, y_pred_train, average=None, labels=CLASS_ORDER)

    print("\n=== TREINO ===")
    print("Acurácia:", acc_train)
    print("Recall (macro):", recall_train_macro)
    print("Recall (weighted):", recall_train_weighted)
    print("Recall por classe [0,1,2]:", recall_train_per_class)

    cm_train = confusion_matrix(y_train, y_pred_train, labels=CLASS_ORDER)
    plot_confusion(cm_train, "Matriz de Confusão — TREINO (Naive Bayes)")

    # =========================
    # TESTE
    # =========================
    y_pred_test = modelo.predict(x_test)

    acc_test = accuracy_score(y_test, y_pred_test)
    recall_test_macro = recall_score(y_test, y_pred_test, average="macro", labels=CLASS_ORDER)
    recall_test_weighted = recall_score(y_test, y_pred_test, average="weighted", labels=CLASS_ORDER)
    recall_test_per_class = recall_score(y_test, y_pred_test, average=None, labels=CLASS_ORDER)

    print("\n=== TESTE ===")
    print("Acurácia:", acc_test)
    print("Recall (macro):", recall_test_macro)
    print("Recall (weighted):", recall_test_weighted)
    print("Recall por classe [0,1,2]:", recall_test_per_class)

    cm_test = confusion_matrix(y_test, y_pred_test, labels=CLASS_ORDER)
    plot_confusion(cm_test, "Matriz de Confusão — TESTE (Naive Bayes)")

    # =========================
    # MELHORIA 1: Cross-validation
    # =========================
    print("\n=== CROSS-VALIDATION (StratifiedKFold) no TREINO ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores_acc = cross_val_score(GaussianNB(), x_train, y_train, cv=cv, scoring="accuracy")
    scores_rec = cross_val_score(GaussianNB(), x_train, y_train, cv=cv, scoring="recall_macro")

    print("Accuracy CV: média =", scores_acc.mean(), "| std =", scores_acc.std())
    print("Recall macro CV: média =", scores_rec.mean(), "| std =", scores_rec.std())

    # =========================
    # MELHORIA 2: Probabilidades e confiança
    # =========================
    print("\n=== PROBABILIDADES (predict_proba) no TESTE ===")
    probas = modelo.predict_proba(x_test)  # (n_amostras, n_classes)
    classes = modelo.classes_

    df_proba = pd.DataFrame(probas, columns=[f"P(classe={c})" for c in classes])
    df_proba["Real"] = y_test.values
    df_proba["Predito"] = y_pred_test

    # confiança = maior probabilidade da linha
    prob_cols = [c for c in df_proba.columns if c.startswith("P(")]
    df_proba["Confiança"] = df_proba[prob_cols].max(axis=1)

    # mostrar os 10 casos mais "incertos"
    print("\nTop 10 previsões com MENOR confiança:")
    print(df_proba.sort_values("Confiança").head(10).to_string(index=False))

    limiar = 0.60
    incertos = df_proba[df_proba["Confiança"] < limiar]
    print(f"\nTotal de previsões incertas (conf < {limiar}): {len(incertos)}")


if __name__ == "__main__":
    main()
