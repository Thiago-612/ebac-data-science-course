import requests

def enviar_arquivo():
    caminho = 'C:/Users/thiag/Downloads/produtos_informatica.xlsx'

    #ENVIAR ARQUIVO
    requisicao = requests.post( url='https://upload.gofile.io/uploadFile', files={'file': open(caminho, 'rb')})
    saida_requisicao = requisicao.json()

    print(saida_requisicao)
    url = saida_requisicao['data']['downloadPage']
    print('Arquivo enviado. Link para acesso: ', url)

def enviar_arquivo_chave():
    caminho = 'C:/Users/thiag/Downloads/produtos_informatica.xlsx'
    chave_acesso =
    requisicao = requests.post(
        url='https://upload.gofile.io/uploadFile',
        files={'file': open(caminho, 'rb')}),
        headers={'Authorization': 'Bearer ' + chave_acesso}
    saida_requisicao = requisicao.json()
    print(saida_requisicao)
    url = saida_requisicao['data']['downloadPage']
    print('Arquivo enviado. Link para acesso: ', url)

def receber_arquivo(file_url):
    #RECEBER O ARQUIVO
    requisicao = requests.get(file_url)

    #SALVAR O ARQUIVO
    if requisicao.ok:
        with open(file_url, 'wb') as file:
            file.write(requisicao.content)
        print('Arquivo baixado com sucesso.')
    else:
        print('Erro ao baixar o arquivo: ', requisicao.json())




enviar_arquivo()
enviar_arquivo_chave()
receber_arquivo(file_url)






