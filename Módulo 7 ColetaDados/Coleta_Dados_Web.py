import requests
from bs4 import BeautifulSoup

url = 'https://python.org.br/web/'

cabecalho = {

  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, como Gecko) Chrome/91.0.4472.124 Safari/537.36'

}

response = requests.get(url, headers=cabecalho)

print('\n---BEAUTIFUL_SOUP---\n')
soup = BeautifulSoup(response.text, 'html.parser')

#EXIBIR O TEXTO
print(soup.text.strip())

#FILTRAR PELAS TAGS
for linha_texto in soup.find_all('h2'):
    titulo = linha_texto.text.strip()
    print('Titulo: ', titulo)

#CONTAR QUANTIDADE DE TÍTULOS E PARÁGRAFOS
Contador_Titulo = 0
Contador_Paragrafo = 0

for linha_texto in soup.find_all(['h2','p']):
    if linha_texto.name == 'h2':
        Contador_Titulo += 1
    elif linha_texto.name == 'p':
        Contador_Paragrafo += 1

print('Quantidade de Títulos: ' ,Contador_Titulo, 'Quantidade de Parágrafos: ' ,Contador_Paragrafo)

#EXIBIR O TEXTO DAS TAGS
for linha_texto in soup.find_all(['h2','p']):
    if linha_texto.name == 'h2':
        print('Titulo:\n ', linha_texto.text.strip())
    elif linha_texto.name == 'p':
        print('Parágrafo:\n ', linha_texto.text.strip())

#EXIBIR TAGS ANINHADA
for titulo in soup.find_all('h2'):
    print('\n Titulo: ' , titulo.text.strip())
    for link in titulo.find_next_siblings ('p'):
        for a in link.find_all('a', href=True):
            print('Texto link: ', a.text.strip(), 'URL:', a["href"])




