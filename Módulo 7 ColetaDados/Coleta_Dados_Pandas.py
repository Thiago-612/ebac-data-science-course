import pandas as pd

# url = 'https://br.financas.yahoo.com/quote/%5EBVSP/history/'
# PANDAS ACEITA SOMENTE SITES COM HTML ESTÁTICO, O LINK ACIMA USA JAVA SCRIPT
url = 'https://pt.wikipedia.org/wiki/Lista_de_empresas_cujo_nome_come%C3%A7a_com_a_letra_A_na_B3'

print('\nPANDAS\n')
url_dados = pd.read_html(url)
print(url_dados[0].head())
