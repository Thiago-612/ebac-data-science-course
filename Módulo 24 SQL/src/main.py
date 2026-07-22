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

    #def run_query(query):
        #return pd.read_sql_query(query, conn)

    query = "SELECT * FROM tb_vendas"
    result_df = run_query(query, conn)
    print('\nCONSULTA GERAL:\n', result_df)

    query = "SELECT PRODUTO FROM tb_vendas limit 10"
    result_df = run_query(query, conn)
    print('\nCONSULTA PRODUTO 10 LINHAS:\n', result_df)

    query = ("SELECT AVG(VALOR_UNID) AS MÉDIA_VALOR_UNIDADE, AVG(UNIDADES) AS MÉDIA_UNIDADES_VENDIDAS FROM tb_vendas")
    result_df = run_query(query, conn)
    print('\nCONSULTA DAS MÉDIAS:\n', result_df)

    query = ("SELECT ID_COMPRA, ID_CLIENTE, VALOR_UNID * UNIDADES AS VALOR_TOTAL FROM tb_vendas")
    result_df = run_query(query, conn)
    print('\nTOTAL GASTO POR COMPRAS:\n', result_df)

    query = """
    SELECT AVG(valor_total_compra) AS MEDIA_VALOR_TOTAL
    FROM (
        SELECT ID_COMPRA, SUM(VALOR_UNID * UNIDADES) AS valor_total_compra
        FROM tb_vendas
        GROUP BY ID_COMPRA
    )
    """
    result_df = run_query(query, conn)
    print('\nMÉDIA DO PREÇO DE TODAS AS COMPRAS:\n', result_df)

if __name__ == '__main__':
    main()