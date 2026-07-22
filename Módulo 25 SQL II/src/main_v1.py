import sqlite3
from pathlib import Path
import pandas as pd


# =========================
# Config / Configuracao
# =========================
CSV_NAME = "TB_VENDAS_TAREFA.csv"
TABLE_NAME = "tb_vendas"
CHUNKSIZE = 100_000  # EN: adjust for your machine / PT: ajuste conforme sua maquina


def load_data_in_chunks(chunksize: int = CHUNKSIZE):
    """
    EN: Load a CSV into an in-memory SQLite database using pandas chunksize to avoid RAM issues.
        Also applies basic data normalization (decimal comma, trimming, case normalization).
    PT: Carrega um CSV em um SQLite em memoria usando chunksize para evitar estourar a RAM.
        Tambem aplica normalizacao basica (virgula decimal, trim, padronizacao de caixa).
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    csv_path = data_dir / CSV_NAME

    # EN: Create SQLite database in memory / PT: Cria SQLite em memoria
    conn = sqlite3.connect(":memory:")

    first_chunk = True

    # EN: Read in chunks / PT: Leitura em blocos
    for chunk in pd.read_csv(csv_path, delimiter=";", chunksize=chunksize):
        # -------------------------
        # EN: Basic text normalization
        # PT: Normalizacao basica de texto
        # -------------------------
        if "PRODUTO" in chunk.columns:
            chunk["PRODUTO"] = chunk["PRODUTO"].astype(str).str.strip()
            chunk["PRODUTO_NORM"] = chunk["PRODUTO"].str.upper()

        # -------------------------
        # EN: Fix decimal comma -> float for unit value
        # PT: Corrige virgula decimal -> float no valor unitario
        # -------------------------
        if "VALOR_UNID" in chunk.columns:
            chunk["VALOR_UNID"] = (
                chunk["VALOR_UNID"]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            chunk["VALOR_UNID"] = pd.to_numeric(chunk["VALOR_UNID"], errors="coerce")

        # -------------------------
        # EN: Ensure UNIDADES is numeric
        # PT: Garante UNIDADES como numerico
        # -------------------------
        if "UNIDADES" in chunk.columns:
            chunk["UNIDADES"] = pd.to_numeric(chunk["UNIDADES"], errors="coerce").fillna(0).astype(int)

        # -------------------------
        # EN: Append chunk to SQLite (replace on first chunk)
        # PT: Insere no SQLite (replace no primeiro chunk, append nos demais)
        # -------------------------
        chunk.to_sql(
            TABLE_NAME,
            conn,
            index=False,
            if_exists="replace" if first_chunk else "append",
        )
        first_chunk = False

    return conn


def run_query(query: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    EN: Execute SQL query and return as DataFrame.
    PT: Executa uma query SQL e retorna como DataFrame.
    """
    return pd.read_sql_query(query, conn)


def main():
    # =========================
    # 0) Load data (chunked) / Carregar dados (em blocos)
    # =========================
    conn = load_data_in_chunks(chunksize=CHUNKSIZE)

    # 1) Full table (sample)
    query = f"""
    SELECT *
    FROM {TABLE_NAME}
    LIMIT 30
    """
    result_df = run_query(query, conn)
    print("\n==== FULL TABLE (SAMPLE) ====\n", result_df)

    # 2) Distinct products (normalized)
    query = f"""
    SELECT DISTINCT PRODUTO_NORM AS PRODUTO
    FROM {TABLE_NAME}
    ORDER BY PRODUTO
    """
    result_df = run_query(query, conn)
    print("\n==== DISTINCT PRODUCTS (NORMALIZED) ====\n", result_df)

    # 3) Count distinct customers
    query = f"""
    SELECT COUNT(DISTINCT ID_CLIENTE) AS CLIENTES_DISTINTOS
    FROM {TABLE_NAME}
    """
    result_df = run_query(query, conn)
    print("\n==== DISTINCT CUSTOMERS COUNT ====\n", result_df)

    # 4) Distinct products where unit value >= 50
    query = f"""
    SELECT DISTINCT
        PRODUTO_NORM AS PRODUTO,
        VALOR_UNID
    FROM {TABLE_NAME}
    WHERE VALOR_UNID >= 50
    ORDER BY VALOR_UNID DESC
    """
    result_df = run_query(query, conn)
    print("\n==== DISTINCT PRODUCTS WITH UNIT VALUE >= 50 ====\n", result_df)

    # 5) Top 5 purchases by total value
    query = f"""
    SELECT
        ID_COMPRA,
        (VALOR_UNID * UNIDADES) AS VALOR_TOTAL
    FROM {TABLE_NAME}
    ORDER BY VALOR_TOTAL DESC
    LIMIT 5
    """
    result_df = run_query(query, conn)
    print("\n==== TOP 5 PURCHASES BY TOTAL VALUE ====\n", result_df)

    # 6) Average unit price by product (normalized)
    query = f"""
    SELECT
        PRODUTO_NORM AS PRODUTO,
        AVG(VALOR_UNID) AS PRECO_MEDIO
    FROM {TABLE_NAME}
    GROUP BY PRODUTO_NORM
    ORDER BY PRECO_MEDIO DESC
    """
    result_df = run_query(query, conn)
    print("\n==== AVERAGE UNIT PRICE BY PRODUCT ====\n", result_df)

    # 7) Top 3 customers by number of purchases
    query = f"""
    SELECT
        ID_CLIENTE,
        COUNT(ID_COMPRA) AS QUANTIDADE_COMPRAS
    FROM {TABLE_NAME}
    GROUP BY ID_CLIENTE
    ORDER BY QUANTIDADE_COMPRAS DESC
    LIMIT 3
    """
    result_df = run_query(query, conn)
    print("\n==== TOP 3 CUSTOMERS BY NUMBER OF PURCHASES ====\n", result_df)

    # 8) Average total value per purchase (robust aggregate with subquery)
    query = f"""
    SELECT AVG(valor_total_compra) AS MEDIA_VALOR_TOTAL
    FROM (
        SELECT ID_COMPRA, SUM(VALOR_UNID * UNIDADES) AS valor_total_compra
        FROM {TABLE_NAME}
        GROUP BY ID_COMPRA
    )
    """
    result_df = run_query(query, conn)
    print("\n==== AVERAGE TOTAL VALUE PER PURCHASE ====\n", result_df)

    conn.close()


if __name__ == "__main__":
    main()
