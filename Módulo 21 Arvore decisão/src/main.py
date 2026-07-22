from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report


# =========================
# Config / Configuracao
# =========================
RANDOM_STATE = 0
CRITERION = "gini"

# Multiclass (Credit Score) / Multiclasse (Credit Score)
MAP_CLASSES = {0: "Alto", 1: "Baixo", 2: "Médio"}
CLASS_ORDER = [0, 1, 2]
CLASS_NAMES = [MAP_CLASSES[c] for c in CLASS_ORDER]


def load_data():
    """
    EN: Load train/test datasets from /data (project root is one level above /src).
    PT: Carrega os datasets de treino/teste da pasta /data (raiz do projeto fica 1 nivel acima de /src).
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    # EN: Training set (balanced) / PT: Base de treino (balanceada)
    x_train = pd.read_csv(data_dir / "x_train_bal.csv")
    y_train = pd.read_csv(data_dir / "y_train_bal.csv").squeeze()

    # EN: Test set (original distribution) / PT: Base de teste (distribuicao original)
    x_test = pd.read_csv(data_dir / "x_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    return x_train, y_train, x_test, y_test


def evaluate(model_name: str, model, X: pd.DataFrame, y: pd.Series):
    """
    EN: Print key metrics and return them in a dict (fixed multiclass order).
    PT: Imprime as metricas principais e retorna em dict (ordem fixa para multiclasse).
    """
    # EN: Predict / PT: Predicao
    y_pred = model.predict(X)

    # EN: Metrics / PT: Metricas
    acc = accuracy_score(y, y_pred)
    rec_macro = recall_score(y, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
    rec_weighted = recall_score(y, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0)

    # EN: Confusion matrix with fixed label order / PT: Matriz de confusao com ordem fixa
    cm = confusion_matrix(y, y_pred, labels=CLASS_ORDER)

    # EN: Console output / PT: Saida no console
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall (macro): {rec_macro:.4f}")
    print(f"Recall (weighted): {rec_weighted:.4f}")

    print("\nConfusion Matrix (labels = 0/1/2):")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            labels=CLASS_ORDER,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    return {
        "model": model_name,
        "accuracy": acc,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
    }


def save_tree_plot(model, feature_names, out_path: Path, title: str):
    """
    EN: Save a decision tree plot as PNG (Matplotlib).
    PT: Salva o grafico da arvore de decisao como PNG (Matplotlib).
    """
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        proportion=False,
        max_depth=4,  # EN: visual limit / PT: limite visual
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_tree_plotly(model, feature_names, class_names, out_html_path, title="Decision Tree", max_depth=4):
    """
    EN: Plot a sklearn DecisionTreeClassifier using Plotly (interactive) and save as HTML.
    PT: Plota uma arvore do sklearn usando Plotly (interativo) e salva em HTML.

    Parameters / Parametros:
    - model: trained DecisionTreeClassifier / modelo treinado
    - feature_names: list of feature names / lista com nomes das features
    - class_names: list of class labels / lista de nomes das classes
    - out_html_path: output HTML path / caminho do HTML de saida
    - title: chart title / titulo do grafico
    - max_depth: depth limit for visualization / limite de profundidade para visualizacao
    """
    tree = model.tree_
    n_classes = len(class_names)

    # -------------------------
    # EN: Helper to build node label (rule + predicted class + class distribution)
    # PT: Helper para montar texto do no (regra + classe prevista + distribuicao)
    # -------------------------
    def node_label(node_id: int) -> str:
        value = tree.value[node_id][0]  # class counts / contagens por classe
        total = value.sum()
        probs = value / total if total > 0 else value

        # EN: Internal node vs leaf / PT: No interno vs folha
        if tree.feature[node_id] != -2:
            fname = feature_names[tree.feature[node_id]]
            thr = tree.threshold[node_id]
            rule = f"{fname} ≤ {thr:.3f}"
        else:
            rule = "Leaf"

        pred = int(np.argmax(value)) if total > 0 else -1
        pred_name = class_names[pred] if pred >= 0 else "?"

        dist = "<br>".join(
            [f"{class_names[i]}: {int(value[i])} ({probs[i]:.2f})" for i in range(n_classes)]
        )
        return f"<b>{rule}</b><br>Pred: <b>{pred_name}</b><br>{dist}"

    # -------------------------
    # EN: Build a pruned view up to max_depth (avoid huge graphs)
    # PT: Monta uma subarvore ate max_depth (evita graficos gigantes)
    # -------------------------
    from collections import deque

    q = deque([(0, 0)])  # (node_id, depth) / (id do no, profundidade)
    keep = set()

    while q:
        node_id, depth = q.popleft()
        if depth > max_depth:
            continue
        keep.add(node_id)

        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left != -1 and right != -1 and depth < max_depth:
            q.append((left, depth + 1))
            q.append((right, depth + 1))

    keep_list = sorted(list(keep))

    # EN/PT: Nodes and edges list
    nodes = [{"id": nid, "label": node_label(nid)} for nid in keep_list]
    edges = []

    for nid in keep_list:
        left = tree.children_left[nid]
        right = tree.children_right[nid]
        if left in keep and right in keep:
            # EN: "True" means condition is satisfied (<= threshold)
            # PT: "True" significa que a condicao foi satisfeita (<= threshold)
            edges.append((nid, left, "True"))
            # EN: "False" means condition is NOT satisfied (> threshold)
            # PT: "False" significa que a condicao NAO foi satisfeita (> threshold)
            edges.append((nid, right, "False"))

    # -------------------------
    # EN: Simple hierarchical layout via DFS (leaves spread in X, depth in Y)
    # PT: Layout hierarquico simples via DFS (folhas espalhadas em X, profundidade em Y)
    # -------------------------
    positions = {}

    def dfs(nid, depth, x_cursor):
        left = tree.children_left[nid]
        right = tree.children_right[nid]

        is_leaf = (tree.feature[nid] == -2) or (left not in keep) or (right not in keep)
        if is_leaf:
            positions[nid] = (x_cursor[0], -depth)
            x_cursor[0] += 1
            return positions[nid][0]

        xl = dfs(left, depth + 1, x_cursor)
        xr = dfs(right, depth + 1, x_cursor)
        xmid = (xl + xr) / 2
        positions[nid] = (xmid, -depth)
        return xmid

    if 0 not in keep:
        raise ValueError("Root node not in keep set (unexpected).")

    dfs(0, 0, [0])

    # -------------------------
    # EN: Build Plotly traces (edges, edge labels, nodes)
    # PT: Monta os traces do Plotly (arestas, rotulos das arestas, nos)
    # -------------------------
    edge_x, edge_y = [], []
    edge_text_x, edge_text_y, edge_text = [], [], []

    for src, dst, label in edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        edge_text_x.append((x0 + x1) / 2)
        edge_text_y.append((y0 + y1) / 2)
        edge_text.append(label)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="skip",
    )

    edge_label_trace = go.Scatter(
        x=edge_text_x,
        y=edge_text_y,
        mode="text",
        text=edge_text,
        hoverinfo="skip",
        textposition="middle center",
    )

    node_x, node_y, node_hover, node_text = [], [], [], []

    for n in nodes:
        nid = n["id"]
        x, y = positions[nid]
        node_x.append(x)
        node_y.append(y)
        node_hover.append(n["label"])

        # EN: short visible text / PT: texto curto visivel
        if tree.feature[nid] != -2:
            fname = feature_names[tree.feature[nid]]
            thr = tree.threshold[nid]
            node_text.append(f"{fname} ≤ {thr:.2f}")
        else:
            node_text.append("Leaf")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(size=18),
    )

    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.write_html(str(out_html_path))
    print(f"🧩 Plotly tree saved to: {out_html_path}")


def main():
    # =========================
    # 0) Load data / Carregar dados
    # =========================
    X_train, y_train, X_test, y_test = load_data()

    print("Train shape:", X_train.shape, " | Test shape:", X_test.shape)
    print("\nTarget distribution (train):")
    print(y_train.value_counts(dropna=False).sort_index())
    print("\nTarget distribution (test):")
    print(y_test.value_counts(dropna=False).sort_index())

    # EN: Create outputs folder / PT: Criar pasta outputs
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    # =========================
    # 1) Baseline model (TRAIN) - rs=0, gini
    #    EN: Train baseline decision tree
    #    PT: Treinar arvore baseline
    # =========================
    baseline = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        criterion=CRITERION,
    )
    baseline.fit(X_train, y_train)

    # EN: Metrics before tuning / PT: Metricas antes do tuning
    m_train_base = evaluate("1) Baseline (gini, rs=0) - TRAIN", baseline, X_train, y_train)

    # =========================
    # 2) Baseline on TEST / Aplicar baseline no TESTE
    # =========================
    m_test_base = evaluate("2) Baseline (gini, rs=0) - TEST", baseline, X_test, y_test)

    # =========================
    # 3) Plot baseline tree (PNG + Plotly HTML)
    #    EN: Save baseline plots
    #    PT: Salvar graficos do baseline
    # =========================
    baseline_tree_path = out_dir / "tree_baseline.png"
    save_tree_plot(
        baseline,
        feature_names=list(X_train.columns),
        out_path=baseline_tree_path,
        title="Decision Tree - Baseline (gini, rs=0) [max_depth=4 view]",
    )
    print(f"\n🖼️ Baseline tree saved to: {baseline_tree_path}")

    plot_tree_plotly(
        baseline,
        feature_names=list(X_train.columns),
        class_names=CLASS_NAMES,
        out_html_path=out_dir / "tree_baseline_plotly.html",
        title="Decision Tree (Baseline) - Plotly",
        max_depth=4,
    )

    # =========================
    # 4) Identify top features / Identificar features mais importantes
    # =========================
    fi_base = pd.Series(baseline.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    top2_features = fi_base.head(2).index.tolist()

    print("\nTop 15 Feature Importances (Baseline):")
    print(fi_base.head(15))

    print(f"\n✅ Top 2 features selected for the 2-feature model: {top2_features}")

    # =========================
    # 5) Train 2-feature model / Treinar modelo com 2 features
    # =========================
    X_train_2 = X_train[top2_features].copy()
    X_test_2 = X_test[top2_features].copy()

    model_2feat = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        criterion=CRITERION,
    )
    model_2feat.fit(X_train_2, y_train)

    # =========================
    # 6) Evaluate 2-feature model (TRAIN + TEST)
    #    EN: Compare performance with reduced features
    #    PT: Comparar performance com features reduzidas
    # =========================
    m_train_2 = evaluate("3) 2-Feature Tree (gini, rs=0) - TRAIN", model_2feat, X_train_2, y_train)
    m_test_2 = evaluate("4) 2-Feature Tree (gini, rs=0) - TEST", model_2feat, X_test_2, y_test)

    tree_2feat_path = out_dir / "tree_2_features.png"
    save_tree_plot(
        model_2feat,
        feature_names=top2_features,
        out_path=tree_2feat_path,
        title="Decision Tree - Top 2 Features (gini, rs=0) [max_depth=4 view]",
    )
    print(f"\n🖼️ 2-feature tree saved to: {tree_2feat_path}")

    plot_tree_plotly(
        model_2feat,
        feature_names=top2_features,
        class_names=CLASS_NAMES,
        out_html_path=out_dir / "tree_2_features_plotly.html",
        title="Decision Tree (Top 2 Features) - Plotly",
        max_depth=4,
    )

    # =========================
    # 7) Hyperparameter tuning (GridSearchCV)
    #    EN: Search best tree settings using CV
    #    PT: Buscar melhores hiperparametros com validacao cruzada
    # =========================
    param_grid = {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 10, 30, 50],
        "min_samples_leaf": [1, 5, 10, 20],
        "class_weight": [None, "balanced"],
        "criterion": [CRITERION],          # EN: force gini / PT: forcar gini
        "random_state": [RANDOM_STATE],    # EN: force rs=0 / PT: forcar rs=0
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=param_grid,
        scoring="recall_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)
    tuned = grid.best_estimator_

    print("\n===== TUNING RESULTS =====")
    print("Best params:", grid.best_params_)
    print("Best CV score (recall_macro):", f"{grid.best_score_:.4f}")

    # EN/PT: tuned metrics
    m_train_tuned = evaluate("5) Tuned Tree (GridSearchCV) - TRAIN", tuned, X_train, y_train)
    m_test_tuned = evaluate("6) Tuned Tree (GridSearchCV) - TEST", tuned, X_test, y_test)

    # =========================
    # 8) Interpretability (importances + rules) / Interpretabilidade
    # =========================
    fi_tuned = pd.Series(tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 15 Feature Importances (Tuned):")
    print(fi_tuned.head(15))

    rules = export_text(tuned, feature_names=list(X_train.columns), max_depth=3)
    print("\nDecision Rules (Tuned, depth <= 3):")
    print(rules)

    tuned_tree_path = out_dir / "tree_tuned.png"
    save_tree_plot(
        tuned,
        feature_names=list(X_train.columns),
        out_path=tuned_tree_path,
        title="Decision Tree - Tuned (gini, rs=0) [max_depth=4 view]",
    )
    print(f"\n🖼️ Tuned tree saved to: {tuned_tree_path}")

    plot_tree_plotly(
        tuned,
        feature_names=list(X_train.columns),
        class_names=CLASS_NAMES,
        out_html_path=out_dir / "tree_tuned_plotly.html",
        title="Decision Tree (Tuned) - Plotly",
        max_depth=4,
    )

    # =========================
    # 9) Save outputs / Salvar saidas
    # =========================
    metrics_df = pd.DataFrame([
        m_train_base,
        m_test_base,
        m_train_2,
        m_test_2,
        m_train_tuned,
        m_test_tuned,
    ])
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fi_base_path = out_dir / "feature_importances_baseline.csv"
    fi_tuned_path = out_dir / "feature_importances_tuned.csv"
    fi_base.to_csv(fi_base_path, header=["importance"])
    fi_tuned.to_csv(fi_tuned_path, header=["importance"])

    rules_path = out_dir / "rules_tuned_depth3.txt"
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(rules)

    top2_path = out_dir / "top2_features.txt"
    with open(top2_path, "w", encoding="utf-8") as f:
        f.write("\n".join(top2_features))

    print("\n✅ Outputs saved:")
    print(f"- {metrics_path}")
    print(f"- {fi_base_path}")
    print(f"- {fi_tuned_path}")
    print(f"- {rules_path}")
    print(f"- {top2_path}")
    print(f"- {baseline_tree_path}")
    print(f"- {tree_2feat_path}")
    print(f"- {tuned_tree_path}")


if __name__ == "__main__":
    main()
