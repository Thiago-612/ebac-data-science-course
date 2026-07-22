import requests
from bs4 import BeautifulSoup

#URL SOLICITADA NO EXERCÍCIO GERA ERRO, ESSA OUTRA NÃO GERA
url = 'https://br.financas.yahoo.com/quote/%5EBVSP/history/'

cabecalho = {

  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, como Gecko) Chrome/91.0.4472.124 Safari/537.36'

}

response = requests.get(url, headers=cabecalho)

print('\n---REQUEST---\n')
print(response.text[:2000])

print('\n---BEAUTIFUL_SOUP---\n')
extracao = BeautifulSoup(response.text, 'html.parser')
print(extracao.prettify()[:2000])
