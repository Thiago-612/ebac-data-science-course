import requests

url = 'https://br.financas.yahoo.com/quote/%5EBVSP/history/'

#SIMULAR UM NAVEGADOR E NÃO ACHAR QUE É BOT
#CABEÇALHO USER-AGENT
cabecalho = {

  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, como Gecko) Chrome/91.0.4472.124 Safari/537.36'

}

response = requests.get(url, headers=cabecalho)
print('CÓDIGO DE ERRO :')
print(response.status_code)
print("\n--- CONTEÚDO HTML RECEBIDO ---")
print(response.text[:500])
