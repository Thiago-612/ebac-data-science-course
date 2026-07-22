import sqlite3
from pathlib import Path
import pandas as pd

def load_data():

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    df_vendas = pd.read_csv(data_dir / "TB_VENDAS_TAREFA.csv", delimiter=';')

    df_vendas["VALOR_UNID"] = (
        df_vendas["VALOR_UNID"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    df_vendas["UNIDADES"] = pd.to_numeric(df_vendas["UNIDADES"], errors="coerce").astype(int)

    conn = sqlite3.connect(':memory:')
    df_vendas.to_sql('tb_vendas', conn, index=False, if_exists='replace')

    return df_vendas, conn

def run_query(query, conn):
    return pd.read_sql_query(query, conn)

def main():

    df_vendas, conn = load_data()

    query = """
    SELECT *
        FROM tb_vendas
    """
    result_df = run_query(query, conn)
    print('\n====TABELA TOTAL:====\n', result_df)

    query = """
    SELECT DISTINCT PRODUTO
        FROM tb_vendas
    """
    result_df = run_query(query, conn)
    print('\n====PRODUTOS DISTINTOS:====\n', result_df)

    query = """
    SELECT COUNT(DISTINCT ID_CLIENTE) AS CLIENTES_DISTINTOS
        FROM tb_vendas
    """
    result_df = run_query(query, conn)
    print('\n====CONTAGEM DOS CLIENTES DISTINTOS:====\n', result_df)

    query = """
    SELECT
    DISTINCT PRODUTO,
    VALOR_UNID
    FROM tb_vendas
    WHERE VALOR_UNID >= 50
    """
    result_df = run_query(query, conn)
    print('\n====PRODUTOS DISTINTOS MAIS DE R$ 50:====\n', result_df)

    query = """
    SELECT
    ID_COMPRA,
    VALOR_UNID * UNIDADES AS VALOR_TOTAL
    FROM tb_vendas
    ORDER BY VALOR_TOTAL DESC 
    LIMIT 5
    """
    result_df = run_query(query, conn)
    print('\n====CINCO MAIORES COMPRAS:====\n', result_df)

    query = """
    SELECT
    PRODUTO,
    AVG (VALOR_UNID) AS PREÇO_MÉDIO
    FROM tb_vendas
    GROUP BY PRODUTO
    ORDER BY PREÇO_MÉDIO DESC 
    """
    result_df = run_query(query, conn)
    print('\n====PREÇO MÉDIO DA UNIDADE POR PRODUTO:====\n', result_df)

    query = """
    SELECT
    ID_CLIENTE,
    COUNT(ID_COMPRA) AS QUANTIDADE_COMPRAS
    FROM tb_vendas
    GROUP BY ID_CLIENTE
    ORDER BY QUANTIDADE_COMPRAS DESC
    LIMIT 3
    """
    result_df = run_query(query, conn)
    print('\n====TOP 3 CLIENTES POR NÚMERO DE COMPRA:====\n', result_df)

if __name__ == '__main__':
    main()