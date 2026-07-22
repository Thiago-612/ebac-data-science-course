import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =====================================================
# 1) LIMPEZA DE COLUNAS
# =====================================================

def clean_columns(
    df: pd.DataFrame,
    drop_cols: list = None,
    to_lower: bool = True,
    remove_special: bool = True,
    remove_unnamed: bool = True
) -> pd.DataFrame:
    """
    Limpa e padroniza nomes das colunas + remove colunas indesejadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    drop_cols : list, opcional
        Lista de colunas que deseja remover (nomes originais).
        A função automaticamente padroniza esses nomes antes de remover.

    to_lower : bool, default=True
        Converte os nomes das colunas para minúsculo.

    remove_special : bool, default=True
        Remove caracteres especiais (mantém apenas letras, números e "_").

    remove_unnamed : bool, default=True
        Remove automaticamente colunas do tipo "Unnamed: 0" (erro comum de CSV).

    Returns
    -------
    pd.DataFrame
        DataFrame com nomes padronizados e colunas removidas.
    """

    df = df.copy()

    # ---------- 1) Padronizar nomes ----------
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

    # ---------- 2) Remover colunas "Unnamed" ----------
    if remove_unnamed:
        df = df.loc[:, ~df.columns.str.contains("^unnamed")]

    # ---------- 3) Remover colunas escolhidas ----------
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


# =====================================================
# 2) TRATAMENTO DE MISSING VALUES
# =====================================================

def handle_missing(
    df: pd.DataFrame,
    num_strategy: str = "median",
    cat_strategy: str = "mode"
) -> pd.DataFrame:
    """
    Trata valores nulos separando colunas numéricas e categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    num_strategy : str, default="median"
        Estratégia para colunas numéricas:
        - "mean"   → média
        - "median" → mediana (mais robusto a outliers)
        - "zero"   → preenche com 0

    cat_strategy : str, default="mode"
        Estratégia para colunas categóricas:
        - "mode"     → valor mais frequente
        - "unknown"  → preenche com "unknown"

    Returns
    -------
    pd.DataFrame
        DataFrame sem valores nulos.
    """

    df = df.copy()

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    # ---------- Numéricos ----------
    for col in num_cols:
        if num_strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())

        elif num_strategy == "median":
            df[col] = df[col].fillna(df[col].median())

        elif num_strategy == "zero":
            df[col] = df[col].fillna(0)

    # ---------- Categóricos ----------
    for col in cat_cols:
        if cat_strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])

        elif cat_strategy == "unknown":
            df[col] = df[col].fillna("unknown")

    return df


# =====================================================
# 3) ENCODING
# =====================================================

def encode_categoricals(
    df: pd.DataFrame,
    drop_first: bool = True
) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding nas variáveis categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    drop_first : bool, default=True
        Remove uma categoria (evita multicolinearidade em modelos lineares).

    Returns
    -------
    pd.DataFrame
        DataFrame com variáveis categóricas transformadas.
    """

    df = df.copy()

    cat_cols = df.select_dtypes(exclude=np.number).columns

    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)

    return df


# =====================================================
# 4) SCALING
# =====================================================

def scale_features(
    df: pd.DataFrame,
    method: str = "standard"
):
    """
    Aplica normalização nas variáveis numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    method : str, default="standard"
        Método de escala:
        - "standard" → StandardScaler (média=0, desvio=1)
        - "minmax"   → MinMaxScaler (0 a 1)

    Returns
    -------
    df_scaled : pd.DataFrame
        DataFrame escalado

    scaler : objeto sklearn
        Objeto scaler treinado (importante para produção)
    """

    df = df.copy()

    num_cols = df.select_dtypes(include=np.number).columns

    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError("Método inválido. Use 'standard' ou 'minmax'.")

    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler



df = clean_columns(df)

df = handle_missing(df)

df = encode_categoricals(df)

df, scaler = scale_features(df)