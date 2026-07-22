from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

"""
CONCLUSÕES:

O código antigo está logo após o código novo e marcado como comentário para possibilitar a exemplificação das conclusões.

A exclusão dos outliers prejudica muito a divisão das classes da target, valores raros são excluídos e prejudicam a classificação do modelo.

Possibilidade de data leakage por causa da correlação que utiliza todo o dataset utilizando as features em comparação com a target e depois gera outro dataframe.
O modelo olha todos os dados por causa da correlação, inclusive os dados de teste.
A correlação será realizada somente nos dados de treino.
Após a alteração o melhor modelo deixou de ser correlação com balanceamento para somente correlação.

Smote pode favorecer o overfitting criando muitos exemplos artificiais no treino enquanto os dados do teste são desbalanceados.
A criação de novos dados gera uma realidade diferente da habitual, as classes raras se tornam normais e o modelo se ajusta muito ao treino e não consegue prever resultados futuros.
O modelo sem smote foi superior ao modelo com smote.

A utilização do paramgrid permissivo está deixando o modelo com as arvores muito profundas, resultando em overfitting.
O treino excessivo pode fazer com que o modelo revise muito os dados e acabe decorando.

Apesar do modelo sem o GridSearch ter tido um desempenho melhor, o modelo com o GridSearch é mais realista, pois a diferença entre o treino e o teste diminuiu.

O modelo pode ser melhorado utilizando feature engineering, criando novas interações (porcentagem de alcool por acidez), novas transformações utilizando escalas logarítmicas,
utilização de binning para definir intervalos nas variáveis.
Também pode ser melhorado com a utilização de feature selection para escolher as melhores variáveis no lugar de usar correlação, pois as arvores capturam
não-linearidade enquanto a correlação capta a linearidade com a target.
"""

"""
Dicas de Melhoria:

Considere utilizar a classe Pipeline do scikit-learn. Ela ajuda a organizar as etapas de pré-processamento (como o SMOTE e a seleção de features) e 
garante que as transformações sejam aplicadas corretamente durante a validação cruzada.
Após o melhor modelo ser definido pelo GridSearch, utilize o atributo feature_importances_ do Random Forest para plotar um gráfico. 
Isso ajuda a explicar o "porquê" das decisões do modelo, agregando valor à análise de negócio.
"""


def load_data() -> pd.DataFrame:
    """
    EN: Load dataset from /data (project root is one level above /src).
    PT: Carrega o dataset da pasta /data (a raiz do projeto fica um nível acima de /src).
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    df = pd.read_csv(data_dir / "winequality-red.csv", delimiter=",")
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


def remove_outliers_iqr(df: pd.DataFrame, colunas, fator: float = 1.5) -> pd.DataFrame:
    """
    EN: Remove outliers using IQR rule.
    PT: Remove outliers usando a regra do IQR.
    """
    df_limpo = df.copy()

    for col in colunas:
        q1 = df_limpo[col].quantile(0.25)
        q3 = df_limpo[col].quantile(0.75)
        iqr = q3 - q1

        limite_baixo = q1 - fator * iqr
        limite_alto = q3 + fator * iqr

        mask_outlier = (df_limpo[col] < limite_baixo) | (df_limpo[col] > limite_alto)
        qtd_outliers = int(mask_outlier.sum())
        total_antes = df_limpo.shape[0]

        df_limpo = df_limpo[~mask_outlier]

        print(f"\nCOLUNA: {col}")
        print(f"Limite inferior: {limite_baixo}")
        print(f"Limite superior: {limite_alto}")
        print(f"Outliers na coluna: {qtd_outliers} ({qtd_outliers / total_antes:.2%})")
        print(f"Linhas restantes após remover: {df_limpo.shape[0]}")

    return df_limpo


def correlations_target(
    df: pd.DataFrame,
    target: str,
    method: str = "pearson",
    top_n: int | None = None,
    threshold: float | None = None,
    absolute: bool = True
) -> pd.DataFrame:
    """
    EN: Return a DataFrame with feature correlations to the target.
    PT: Retorna um DataFrame com as variáveis mais correlacionadas com a target.
    """
    corr_matrix = df.corr(method=method, numeric_only=True)
    corr_target = corr_matrix[target].drop(labels=[target])

    if absolute:
        corr_target = corr_target.abs()

    corr_df = corr_target.reset_index()
    corr_df.columns = ["Variavel", "Correlacao"]
    corr_df = corr_df.sort_values(by="Correlacao", ascending=False)

    if threshold is not None:
        corr_df = corr_df[corr_df["Correlacao"] >= threshold]

    if top_n is not None:
        corr_df = corr_df.head(top_n)

    return corr_df


def train_random_forest(
    x_train,
    y_train,
    n_estimators: int = 200,
    random_state: int = 42,
    class_weight=None,
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features="sqrt"
):
    """
    EN: Train Random Forest model.
    PT: Treina o modelo Random Forest.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x, y, dataset_name: str = "Dataset") -> dict:
    """
    EN: Evaluate the model on a dataset.
    PT: Avalia o modelo em um conjunto de dados.
    """
    y_pred = model.predict(x)

    results = {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
    }

    print(f"\n==== {dataset_name.upper()} ====\n")
    for k, v in results.items():
        if k != "dataset":
            print(f"{k}: {v:.4f}")

    return results


def evaluate_train_test(model, x_train, y_train, x_test, y_test, model_name: str = "Modelo") -> dict:
    """
    EN: Evaluate model on train and test.
    PT: Avalia o modelo no treino e no teste.
    """
    train_results = evaluate_model(model, x_train, y_train, dataset_name=f"{model_name} - Treino")
    test_results = evaluate_model(model, x_test, y_test, dataset_name=f"{model_name} - Teste")

    combined = {
        "model_name": model_name,
        "accuracy_train": train_results["accuracy"],
        "precision_macro_train": train_results["precision_macro"],
        "recall_macro_train": train_results["recall_macro"],
        "f1_macro_train": train_results["f1_macro"],
        "accuracy_test": test_results["accuracy"],
        "precision_macro_test": test_results["precision_macro"],
        "recall_macro_test": test_results["recall_macro"],
        "f1_macro_test": test_results["f1_macro"],
    }

    return combined


def grid_search_random_forest(x_train, y_train, x_test, y_test, class_weight=None) -> dict:
    """
    EN: Perform GridSearchCV for Random Forest with a more regularized grid.
    PT: Executa GridSearchCV para Random Forest com um grid mais regularizado.
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 8, 10, 12],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 4, 8],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
        "bootstrap": [True]
    }

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    rf = RandomForestClassifier(
        random_state=42,
        class_weight=class_weight,
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv_strategy,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )

    grid.fit(x_train, y_train)

    print("\n==== GRID SEARCH RANDOM FOREST ====\n")
    print("Melhores hiperparâmetros:")
    print(grid.best_params_)
    print(f"\nMelhor f1_macro médio na validação cruzada: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    print("\n==== ANÁLISE DO MELHOR MODELO - TREINO ====\n")
    train_metrics = evaluate_model(best_model, x_train, y_train, dataset_name="Treino")

    print("\n==== ANÁLISE DO MELHOR MODELO - TESTE ====\n")
    test_metrics = evaluate_model(best_model, x_test, y_test, dataset_name="Teste")

    return {
        "best_params": grid.best_params_,
        "best_cv_score_f1_macro": grid.best_score_,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_estimator": best_model
    }


def main() -> None:
    """
    EN: Main pipeline.
    PT: Pipeline principal.
    """
    df = load_data()
    describe_data(df)

    print("\n==== VERIFICAÇÃO DOS OUTLIERS COM IQR ====\n")
    colunas_numericas = df.select_dtypes(include=["int64", "float64"]).columns
    df_sem_outliers = remove_outliers_iqr(df, colunas_numericas)

    print("\n==== BALANCEAMENTO TARGET ====\n")
    print("\nDataframe original:")
    print(df["quality"].value_counts(normalize=True) * 100)


    print("\nDataframe sem outliers:")
    print(df_sem_outliers["quality"].value_counts(normalize=True) * 100)

    # Mantemos o dataframe original porque a remoção de outliers prejudicou classes raras.
    x = df.drop("quality", axis=1)
    y = df["quality"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    print("\n==== VERIFICAÇÃO DAS CORRELAÇÕES NO TREINO ====\n")
    print("Somente colunas com correlação maior que 10 %")
    train_df = x_train.copy()
    train_df["quality"] = y_train.values

    corr_train = correlations_target(
        train_df,
        target="quality",
        threshold=0.1,
        method="spearman"
    )

    print(corr_train.to_string(index=False))

    selected_features = corr_train["Variavel"].tolist()

    print("\nColunas selecionadas:")
    print(selected_features)

    x_train_corr = x_train[selected_features].copy()
    x_test_corr = x_test[selected_features].copy()

    print("\nTamanho do x_train:", x_train.shape)
    print("Tamanho do y_train:", y_train.shape)
    print("Tamanho do x_test:", x_test.shape)
    print("Tamanho do y_test:", y_test.shape)

    # SMOTE apenas para comparação exploratória, não para o modelo final
    smote = SMOTE(k_neighbors=3, random_state=42)
    x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)
    x_train_corr_bal, y_train_corr_bal = smote.fit_resample(x_train_corr, y_train)

    print("\n==== BALANCEAMENTO ====\n")
    print("Após SMOTE:")
    print("x_train_bal:", x_train_bal.shape)
    print("y_train_bal:", y_train_bal.shape)
    print("x_train_corr_bal:", x_train_corr_bal.shape)
    print("y_train_corr_bal:", y_train_corr_bal.shape)

    results = []

    model_original = train_random_forest(
        x_train, y_train, class_weight="balanced",
        max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    results.append(
        evaluate_train_test(model_original, x_train, y_train, x_test, y_test, model_name="Original")
    )

    model_original_balanceado = train_random_forest(
        x_train_bal, y_train_bal, class_weight=None,
        max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    results.append(
        evaluate_train_test(
            model_original_balanceado, x_train, y_train, x_test, y_test, model_name="Original Balanceado"
        )
    )

    model_corr = train_random_forest(
        x_train_corr, y_train, class_weight="balanced",
        max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    results.append(
        evaluate_train_test(model_corr, x_train_corr, y_train, x_test_corr, y_test, model_name="Correlacao")
    )

    model_corr_balanceado = train_random_forest(
        x_train_corr_bal, y_train_corr_bal, class_weight=None,
        max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    results.append(
        evaluate_train_test(
            model_corr_balanceado, x_train_corr, y_train, x_test_corr, y_test, model_name="Correlacao Balanceado"
        )
    )

    results_df = pd.DataFrame(results)

    print("\n==== COMPARAÇÃO DOS MODELOS ====\n")
    print(results_df.round(4).to_string(index=False))

    best_row = results_df.sort_values("f1_macro_test", ascending=False).iloc[0]

    print("\n==== MELHOR MODELO BASE ====\n")
    print(best_row.to_string())

    print("\n==== GRID SEARCH FINAL (SEM SMOTE, COM CLASS_WEIGHT, FEATURES POR CORRELAÇÃO) ====\n")
    resultado_grid = grid_search_random_forest(
        x_train_corr,
        y_train,
        x_test_corr,
        y_test,
        class_weight="balanced"
    )

    print("\n==== RESUMO FINAL ====\n")
    print("Best params:", resultado_grid["best_params"])
    print(f"CV f1_macro: {resultado_grid['best_cv_score_f1_macro']:.4f}")
    print(f"Train f1_macro: {resultado_grid['train_metrics']['f1_macro']:.4f}")
    print(f"Test f1_macro: {resultado_grid['test_metrics']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()


# CÓDIGO ANTIGO

# def load_data():
#     """
#     EN: Load datasets from /data (project root is one level above /src).
#     PT: Carrega os datasets da pasta /data (raiz do projeto fica 1 nivel acima de /src).
#     """
#     project_root = Path(__file__).resolve().parents[1]
#     data_dir = project_root / "data"
#
#     df = pd.read_csv(data_dir / "winequality-red.csv", delimiter=',')
#
#     return df
#
#
# def describe_data(df: pd.DataFrame) -> None:
#     """
#     Exibe visão geral do dataset.
#     """
#     print("\n==== VISÃO GERAL DOS DADOS ====\n")
#     print(df.head().to_string())
#
#     print("\n==== SHAPE ====\n")
#     print(df.shape)
#
#     print("\n==== TIPOS DE DADOS ====\n")
#     print(df.dtypes)
#
#     print("\n==== VALORES NULOS ====\n")
#     print(df.isnull().sum())
#
#     print("\n==== ANÁLISE DOS DADOS ====\n")
#     print(df.describe().to_string())
#
#
# def remove_outliers_iqr(df, colunas, fator=1.5):
#     """
#     Testar modelo com e sem outliers.
#     """
#     df_limpo = df.copy()
#
#     for col in colunas:
#         Q1 = df_limpo[col].quantile(0.25)
#         Q3 = df_limpo[col].quantile(0.75)
#         IQR = Q3 - Q1
#
#         limite_baixo = Q1 - fator * IQR
#         limite_alto = Q3 + fator * IQR
#
#         mask_outlier = (df_limpo[col] < limite_baixo) | (df_limpo[col] > limite_alto)
#         qtd_outliers = int(mask_outlier.sum())
#         total_antes = df_limpo.shape[0]
#
#         # agora sim filtra removendo outliers dessa coluna
#         df_limpo = df_limpo[~mask_outlier]
#
#         print(f"\nCOLUNA: {col}")
#         print(f"Limite inferior: {limite_baixo}")
#         print(f"Limite superior: {limite_alto}")
#         print(f"Outliers na coluna: {qtd_outliers} ({qtd_outliers/total_antes:.2%})")
#         print(f"Linhas restantes após remover: {df_limpo.shape[0]}")
#
#     return df_limpo
#
#
# def correlations_target(
#     df: pd.DataFrame,
#     target: str,
#     method: str = "pearson",
#     top_n: int | None = None,
#     threshold: float | None = None,
#     absolute: bool = True
# ) -> pd.DataFrame:
#     """
#     Retorna um DataFrame com as variáveis mais correlacionadas com a target.
#
#     Parâmetros:
#     ----------
#     df : DataFrame completo
#     target : nome da variável alvo
#     method : 'pearson', 'spearman' ou 'kendall'
#     top_n : retorna as N mais relevantes
#     threshold : retorna variáveis com correlação >= threshold
#     absolute : usa valor absoluto da correlação
#
#     Retorno:
#     -------
#     DataFrame com colunas:
#     - Variavel
#     - Correlacao
#     """
#
#     # calcula matriz de correlação
#     corr_matrix = df.corr(method=method, numeric_only=True)
#
#     # pega só a correlação com a target
#     corr_target = corr_matrix[target].drop(labels=[target])
#
#     # aplica valor absoluto se desejado
#     if absolute:
#         corr_target = corr_target.abs()
#
#     # transforma em DataFrame
#     corr_df = corr_target.reset_index()
#     corr_df.columns = ["Variavel", "Correlacao"]
#
#     # ordena
#     corr_df = corr_df.sort_values(by="Correlacao", ascending=False)
#
#     # aplica threshold
#     if threshold is not None:
#         corr_df = corr_df[corr_df["Correlacao"] >= threshold]
#
#     # aplica top_n
#     if top_n is not None:
#         corr_df = corr_df.head(top_n)
#
#     return corr_df
#
#
# #treina o modelo
# def train_random_forest(
#     x_train,
#     y_train,
#     n_estimators=200,
#     random_state=42,
#     class_weight=None
# ):
#     model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         random_state=random_state,
#         class_weight=class_weight
#     )
#     model.fit(x_train, y_train)
#     return model
#
#
# #avalia o modelo
# def evaluate_model(model, x, y, dataset_name="Dataset"):
#     y_pred = model.predict(x)
#
#     results = {
#         "dataset": dataset_name,
#         "accuracy": accuracy_score(y, y_pred),
#         "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
#         "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
#         "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
#     }
#
#     print(f"\n==== {dataset_name.upper()} ====\n")
#     for k, v in results.items():
#         if k != "dataset":
#             print(f"{k}: {v:.4f}")
#
#     return results
#
#
# #avalia treino com teste
# def evaluate_train_test(model, x_train, y_train, x_test, y_test, model_name="Modelo"):
#     train_results = evaluate_model(model, x_train, y_train, dataset_name=f"{model_name} - Treino")
#     test_results = evaluate_model(model, x_test, y_test, dataset_name=f"{model_name} - Teste")
#
#     combined = {
#         "model_name": model_name,
#         "accuracy_train": train_results["accuracy"],
#         "precision_macro_train": train_results["precision_macro"],
#         "recall_macro_train": train_results["recall_macro"],
#         "f1_macro_train": train_results["f1_macro"],
#         "accuracy_test": test_results["accuracy"],
#         "precision_macro_test": test_results["precision_macro"],
#         "recall_macro_test": test_results["recall_macro"],
#         "f1_macro_test": test_results["f1_macro"],
#     }
#
#     return combined
#
#
# # Possibilidade de trocar os parâmetros pelos dos comentários.
# # Utilizar combinação para calcular o tempo. ((3x3x3x3x2x2x2)x10) = 6480 modelos
# # Média de 0.3 segundos para um processador moderno. 6480 x 0,3 = 1944 segundos, 32 minutos.
# def grid_search_random_forest(x_train, y_train, x_test, y_test, class_weight=None):
#     param_grid = {
#         # Muito demorado,
#         # "n_estimators": [100, 200, 300, 500],
#         # "max_depth": [None, 10, 20, 30, 50],
#         # "min_samples_split": [2, 5, 10, 15],
#         # "min_samples_leaf": [1, 2, 4, 6],
#         # "max_features": ["sqrt", "log2", None],
#         # "criterion": ["gini", "entropy", "log_loss"],
#         # "bootstrap": [True, False]
#
#         # Deu overfitting, f1 treino de 1 e f1 teste 0,38.
#         # "n_estimators": [200, 300, 500],
#         # "max_depth": [None, 20, 30],
#         # "min_samples_split": [2, 5, 10],
#         # "min_samples_leaf": [1, 2, 4],
#         # "max_features": ["sqrt", "log2"],
#         # "criterion": ["gini", "entropy"],
#         # "bootstrap": [True, False]
#
#         # Deu overfitting, f1 treino de 0,99 e f1 teste 0,41.
#         # "n_estimators": [200, 300],
#         # "max_depth": [None, 10, 20, 30],
#         # "min_samples_split": [2, 5, 10],
#         # "min_samples_leaf": [2, 4, 6],
#         # "max_features": ["sqrt", "log2"],
#         # "criterion": ["gini", "entropy"],
#         # "bootstrap": [True]
#
#         # Deu overfitting, splits 5, com smote treino f1 0,99 e teste 0,38, sem smote treino f1 0,93 e teste 0,41
#         # "n_estimators": [200, 300],
#         # "max_depth": [10, 15, 20, 30],
#         # "min_samples_split": [2, 5, 10],
#         # "min_samples_leaf": [2, 4, 6],
#         # "max_features": ["sqrt", "log2"],
#         # "criterion": ["gini", "entropy"],
#         # "bootstrap": [True]
#
#
#         "n_estimators": [200, 300],
#         "max_depth": [10, 15, 20, 30],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [2, 4, 6],
#         "max_features": ["sqrt", "log2"],
#         "criterion": ["gini", "entropy"],
#         "bootstrap": [True]
#     }
#
#     cv_strategy = StratifiedKFold(
#         n_splits=5,
#         #n_splits=10,
#         shuffle=True,
#         random_state=42
#     )
#
#     rf = RandomForestClassifier(
#         random_state=42,
#         class_weight=class_weight
#     )
#
#     grid = GridSearchCV(
#         estimator=rf,
#         param_grid=param_grid,
#         scoring="f1_macro",
#         cv=cv_strategy,
#         #n_jobs=10,
#         n_jobs=-1,
#         refit=True,
#         return_train_score=True
#     )
#
#     grid.fit(x_train, y_train)
#
#     print("\n==== GRID SEARCH RANDOM FOREST ====\n")
#     print("Melhores hiperparâmetros:")
#     print(grid.best_params_)
#
#     print(f"\nMelhor f1_macro médio na validação cruzada: {grid.best_score_:.4f}")
#
#     melhor_modelo = grid.best_estimator_
#
#     print("\n==== ANÁLISE DO MELHOR MODELO - TREINO ====\n")
#     resultados_treino = evaluate_model(
#         melhor_modelo,
#         x_train,
#         y_train,
#         dataset_name="Treino"
#     )
#
#     print("\n==== ANÁLISE DO MELHOR MODELO - TESTE ====\n")
#     resultados_teste = evaluate_model(
#         melhor_modelo,
#         x_test,
#         y_test,
#         dataset_name="Teste"
#     )
#
#     resultados_finais = {
#         "best_params": grid.best_params_,
#         "best_cv_score_f1_macro": grid.best_score_,
#         "train_metrics": resultados_treino,
#         "test_metrics": resultados_teste,
#         "best_estimator": melhor_modelo
#     }
#
#     return resultados_finais
#
#
# def main() -> None:
#     """
#     Pipeline principal do projeto.
#
#     dataframes utilizados:
#     df
#     df_corr
#     """
#     df = load_data()
#     describe_data(df)
#
#     print('\n==== VERIFICAÇÃO DOS OUTLIERS COM IQR: ====')
#     colunas_numericas = df.select_dtypes(include=['int64','float64']).columns
#     df_sem_outliers = remove_outliers_iqr(df, colunas_numericas)
#     print("\nDATAFRAME SEM OUTLIERS (IQR):")
#     print(df_sem_outliers.describe())
#
#     print('\n==== BALANCEAMENTO TARGET: ====')
#     print("\nDataframe original:")
#     print(df['quality'].value_counts(normalize=True) * 100)
#
#     print("\nDataframe sem outliers:")
#     print(df_sem_outliers['quality'].value_counts(normalize=True) * 100)
#
#     """
#     Com a retirada dos outliers a target fica com as classes mais prejudicadas do que já estão no dataframe original por conta do desbalanceamento.
#     Decisão de continuar com os outliers e gerar quatro modelos, um com balanceamento e outro sem balanceamento, para cada dataframe.
#     """
#
#     print('\n==== VERIFICAÇÃO DAS CORRELAÇÕES: ====')
#     print('\nDATAFRAME COM OUTLIERS:')
#     corr = correlations_target(
#         df,
#         target="quality",
#         method="spearman"
#     )
#     print(corr.to_string(index=False))
#
#     print('\n==== DATAFRAME ATUALIZADO: ====')
#
#     corr_df = correlations_target(
#         df,
#         target="quality",
#         threshold=0.1,
#         method="spearman"
#     )
#
#     features = corr_df["Variavel"].tolist()
#     df_corr = df[features + ["quality"]]
#
#     print(df_corr.head())
#     print("\nColunas selecionadas:")
#     print(df_corr.columns.tolist())
#
#     #MODELO
#     x = df.drop("quality", axis=1)
#     y = df["quality"]
#
#     x_corr = df_corr.drop("quality", axis=1)
#     y_corr = df_corr["quality"]
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
#     x_train_corr, x_test_corr, y_train_corr, y_test_corr = train_test_split(x_corr, y_corr, test_size=0.25, random_state=42, stratify=y)
#
#     # Sem necessidade de verificar o tamanho do df_corr, pois somente as colunas foram excluídas.
#     print('\n Tamanho do x_train:', x_train.shape)
#     print('\n Tamanho do y_train:', y_train.shape)
#     print('\n Tamanho do x_test:', x_test.shape)
#     print('\n Tamanho do y_test:', y_test.shape)
#
#     smote = SMOTE(k_neighbors=3, random_state=42)
#
#     x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)
#     x_train_corr_bal, y_train_corr_bal = smote.fit_resample(x_train_corr, y_train_corr)
#
#     print('\n BALANCEAMENTO:')
#     print("\nApós SMOTE:")
#     print("x_train_bal:", x_train_bal.shape)
#     print("y_train_bal:", y_train_bal.shape)
#     print("x_train_corr_bal:", x_train_corr_bal.shape)
#     print("y_train_corr_bal:", y_train_corr_bal.shape)
#
#     #treino
#     results = []
#
#     model = train_random_forest(x_train, y_train, class_weight="balanced")
#     results.append(
#         evaluate_train_test(model, x_train, y_train, x_test, y_test, model_name="Original")
#     )
#
#     # original balanceado: treina no balanceado, avalia treino no original
#     model_1 = train_random_forest(x_train_bal, y_train_bal, class_weight=None)
#     results.append(
#         evaluate_train_test(model_1, x_train, y_train, x_test, y_test, model_name="Original Balanceado")
#     )
#
#     model_2 = train_random_forest(x_train_corr, y_train_corr, class_weight="balanced")
#     results.append(
#         evaluate_train_test(model_2, x_train_corr, y_train_corr, x_test_corr, y_test_corr, model_name="Correlacao")
#     )
#
#     # correlação balanceado: treina no balanceado, avalia treino no original
#     model_3 = train_random_forest(x_train_corr_bal, y_train_corr_bal, class_weight=None)
#     results.append(
#         evaluate_train_test(model_3, x_train_corr, y_train_corr, x_test_corr, y_test_corr, model_name="Correlacao Balanceado")
#     )
#
#     results_df = pd.DataFrame(results)
#
#     print("\n==== COMPARAÇÃO DOS MODELOS ====\n")
#     print(results_df.round(4).to_string(index=False))
#
#     best_model_row = results_df.sort_values("f1_macro_test", ascending=False).iloc[0]
#
#     print("\n==== MELHOR MODELO ====\n")
#     print(best_model_row.to_string())
#
#     best_model_name = best_model_row["model_name"]
#
#
#     # GridSearch
#     if best_model_name == "Original":
#         resultado_grid = grid_search_random_forest(
#             x_train, y_train, x_test, y_test, class_weight="balanced"
#         )
#     elif best_model_name == "Original Balanceado":
#         resultado_grid = grid_search_random_forest(
#             x_train_bal, y_train_bal, x_test, y_test, class_weight=None
#         )
#     elif best_model_name == "Correlacao":
#         resultado_grid = grid_search_random_forest(
#             x_train_corr, y_train_corr, x_test_corr, y_test_corr, class_weight="balanced"
#         )
#     elif best_model_name == "Correlacao Balanceado":
#         resultado_grid = grid_search_random_forest(
#             x_train_corr_bal, y_train_corr_bal, x_test_corr, y_test_corr, class_weight=None
#         )
#
#     print("\n==== GRID SEARCH RANDOM FOREST SEM BALANCEAMENTO ====\n")
#     resultado_grid = grid_search_random_forest(
#         x_train_corr, y_train_corr, x_test_corr, y_test_corr, class_weight="balanced"
#     )
#
#
# if __name__ == "__main__":
#     main()