import pandas as pd

#USAR LAMBDA PARA ECONOMIZAR LINHAS
#USAR SOMENTE COM FUNÇÕES SIMPLES

def eleva_cubo(x):
    return x**3

df = pd.DataFrame({'numeros': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]})

#EXEMPLO USANDO A FUNÇÃO
# df ['cubo_funcao'] = df['numeros'].apply(eleva_cubo)
# print(df)

df ['cubo_funcao'] = df['numeros'].apply(lambda x: x ** 3)
print(df)
