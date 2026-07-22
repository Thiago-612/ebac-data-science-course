import sqlite3
from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

def normalize_chunk(chunk: pd.DataFrame, table_kind: str) -> pd.DataFrame:
    # 1) padroniza nomes de colunas
    chunk = chunk.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))

    # 2) trim em todas as colunas texto
    for col in chunk.select_dtypes(include="object").columns:
        chunk[col] = chunk[col].astype(str).str.strip()

    if table_kind == "clientes":
        # id_client
        if "id_client" in chunk.columns:
            chunk["id_client"] = pd.to_numeric(chunk["id_client"], errors="coerce").astype("Int64")

        # state_name
        if "state_name" in chunk.columns:
            chunk["state_name"] = chunk["state_name"].str.upper()

        # gender
        if "gender" in chunk.columns:
            chunk["gender"] = chunk["gender"].str.lower()

        # first_name / job_title: apenas strip já ajuda (você pode manter como está)

    elif table_kind == "compras":
        # id_client
        if "id_client" in chunk.columns:
            chunk["id_client"] = pd.to_numeric(chunk["id_client"], errors="coerce").astype("Int64")

        # price: vírgula decimal
        if "price" in chunk.columns:
            chunk["price"] = (
                chunk["price"].astype(str).str.replace(",", ".", regex=False)
            )
            chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")

        # category
        if "category" in chunk.columns:
            chunk["category_norm"] = chunk["category"].str.strip().str.upper()

        # card_type
        if "card_type" in chunk.columns:
            chunk["card_type"] = chunk["card_type"].str.lower()

    return chunk

def load_csv_to_sqlite(conn, csv_path: Path, table_name: str, table_kind: str, chunksize=100_000):
    first = True
    for chunk in pd.read_csv(csv_path, delimiter=";", chunksize=chunksize):
        chunk = normalize_chunk(chunk, table_kind)

        chunk.to_sql(
            table_name, conn, index=False,
            if_exists="replace" if first else "append"
        )
        first = False


def load_all_data(chunksize=100_000):
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    conn = sqlite3.connect(":memory:")

    # CSV 1
    load_csv_to_sqlite(
        conn=conn,
        csv_path=data_dir / "TB_CLIENTES_PROJETO_ECOMM.csv",
        table_name="tb_clientes",
        table_kind="clientes",
        chunksize=chunksize
    )

    # CSV 2
    load_csv_to_sqlite(
        conn=conn,
        csv_path=data_dir / "TB_TRANSACOES_PROJETO_ECOMM.csv",
        table_name="tb_transacoes",
        table_kind="compras",
        chunksize=chunksize
    )

    return conn

def run_query(query: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    EN: Execute SQL query and return as DataFrame.
    PT: Executa uma query SQL e retorna como DataFrame.
    """
    return pd.read_sql_query(query, conn)

def main():
    conn = load_all_data(chunksize=100_000)

    query = f"""
    SELECT *
    FROM tb_clientes
    """
    result_df = run_query(query, conn)
    print("\n==== FIRST FULL TABLE (SAMPLE) ====\n", result_df)

    query = f"""
    SELECT *
    FROM tb_transacoes
    """
    result_df = run_query(query, conn)
    print("\n==== SECOND FULL TABLE (SAMPLE) ====\n", result_df)

    query = """
    WITH
    c AS (SELECT DISTINCT id_client FROM tb_clientes WHERE id_client IS NOT NULL),
    t AS (SELECT DISTINCT id_client FROM tb_transacoes WHERE id_client IS NOT NULL),
    inter AS (SELECT c.id_client FROM c INNER JOIN t ON c.id_client = t.id_client),
    only_c AS (SELECT c.id_client FROM c LEFT JOIN t ON c.id_client = t.id_client WHERE t.id_client IS NULL),
    only_t AS (SELECT t.id_client FROM t LEFT JOIN c ON t.id_client = c.id_client WHERE c.id_client IS NULL),
    union_all AS (
        SELECT id_client FROM c
        UNION
        SELECT id_client FROM t
    )
    SELECT
        (SELECT COUNT(*) FROM c)          AS clientes_tb_clientes,
        (SELECT COUNT(*) FROM t)          AS clientes_tb_transacoes,
        (SELECT COUNT(*) FROM inter)      AS clientes_em_ambas,
        (SELECT COUNT(*) FROM only_c)     AS clientes_so_clientes,
        (SELECT COUNT(*) FROM only_t)     AS clientes_so_transacoes,
        (SELECT COUNT(*) FROM union_all)  AS clientes_total_uniao;
    """
    venn_df = run_query(query, conn)
    print("\n==== VENN SUMMARY ====\n", venn_df.to_string(index=False))

    dim_clientes_query = """
    SELECT
      id_client,
      state_name,
      first_name,
      gender,
      job_title
    FROM tb_clientes
    """
    dim_df = run_query(dim_clientes_query, conn)
    print("\n==== DIMENSION TABLE ====\n", dim_df.head(20).to_string(index=False))

    fact_transaction_query = """
    SELECT
      id_client,
      category,
      category_norm,
      price,
      card_type
    FROM tb_transacoes
    """
    fact_df = run_query(fact_transaction_query, conn)
    print("\n==== FACT TABLE ====\n", fact_df.head(20).to_string(index=False))

    # full_join_query = """
    # SELECT
    #     c.id_client                                    AS id_client,
    #     c.state_name                                   AS state_name,
    #     c.first_name                                   AS first_name,
    #     c.gender                                       AS gender,
    #     c.job_title                                    AS job_title,
    #     t.category                                     AS category,
    #     t.category_norm                                AS category_norm,
    #     t.price                                        AS price,
    #     t.card_type                                    AS card_type
    # FROM tb_clientes c
    # LEFT JOIN tb_transacoes t
    #     ON c.id_client = t.id_client
    #
    # UNION ALL
    #
    # SELECT
    #     t.id_client                                    AS id_client,
    #     NULL                                           AS state_name,
    #     NULL                                           AS first_name,
    #     NULL                                           AS gender,
    #     NULL                                           AS job_title,
    #     t.category                                     AS category,
    #     t.category_norm                                AS category_norm,
    #     t.price                                        AS price,
    #     t.card_type                                    AS card_type
    # FROM tb_transacoes t
    # LEFT JOIN tb_clientes c
    #     ON t.id_client = c.id_client
    # WHERE c.id_client IS NULL;
    # """
    # full_df = run_query(full_join_query, conn)
    # print("\n==== FULL JOIN (EMULATED) SAMPLE ====\n", full_df.head(20).to_string(index=False))

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Export
    venn_path = out_dir / "venn_summary.csv"
    venn_df.to_csv(venn_path, index=False, encoding="utf-8-sig")

    fact_path = out_dir / "fact_table.csv"
    fact_df.to_csv(fact_path, index=False, encoding="utf-8-sig")

    dim_path = out_dir / "dimension_table.csv"
    dim_df.to_csv(dim_path, index=False, encoding="utf-8-sig")

    # # Export full join
    # full_path = out_dir / "clientes_transacoes_full_join.csv"
    # full_df.to_csv(full_path, index=False, encoding="utf-8-sig")

    print("\n✅ Saved:")
    print(f"- {venn_path}")
    print(f"- {fact_path}")
    print(f"- {dim_path}")
    # print(f"- {full_path}")

    conn.close()

if __name__ == "__main__":
    main()