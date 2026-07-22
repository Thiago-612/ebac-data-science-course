import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import plotly.figure_factory as ff
from sklearn.model_selection import StratifiedKFold, cross_val_score

x_train = pd.read_csv('x_train_bal.csv')
y_train = pd.read_csv('y_train_bal.csv')
x_test  = pd.read_csv('x_test.csv')
y_test  = pd.read_csv('y_test.csv')

# transformar y em vetor 1D
y_train = y_train.squeeze()
y_test  = y_test.squeeze()

print('\nVERIFICAÇÃO DAS BASES DE TREINO E TESTE: \n')
print('Tamanho do x_train:', x_train.shape)
print('Tamanho do y_train:', y_train.shape)
print('Tamanho do x_test:', x_test.shape)
print('Tamanho do y_test:', y_test.shape)

print("\nBalanceamento em y_train:")
print(y_train.value_counts())

print("\nBalanceamento em y_test:")
print(y_test.value_counts())

# Modelo
naive_churn = GaussianNB()
naive_churn.fit(x_train, y_train)

#TREINO
# Previsão treino
y_pred_train = naive_churn.predict(x_train)

# Acurácia treino
acc_train = accuracy_score(y_train, y_pred_train)
recall_train_macro = recall_score(y_train, y_pred_train, average="macro")
recall_train_weighted = recall_score(y_train, y_pred_train, average="weighted")
recall_train_por_classe = recall_score(y_train, y_pred_train, average=None)

print("\n=== TREINO ===")
print("Acurácia:", acc_train)
print("Recall (macro):", recall_train_macro)
print("Recall (weighted):", recall_train_weighted)
print("Recall por classe [0,1,2]:", recall_train_por_classe)

# Matriz de confusão (treino)
conf_matrix = confusion_matrix(y_train, y_pred_train)

# Mapeamento correto das classes
# 0 = Alto | 1 = Baixo | 2 = Médio
class_names = ['Alto', 'Baixo', 'Médio']

fig = ff.create_annotated_heatmap(
    z=conf_matrix,
    x=class_names,
    y=class_names,
    colorscale='Blues',
    showscale=True
)

fig.update_layout(
    title='Matriz de Confusão — Treino (Naive Bayes)',
    xaxis_title='Predito',
    yaxis_title='Real',
    template='plotly_white'
)

fig.show()

#TESTE
y_pred_test = naive_churn.predict(x_test)

acc_test = accuracy_score(y_test, y_pred_test)
recall_test_macro = recall_score(y_test, y_pred_test, average="macro")
recall_test_weighted = recall_score(y_test, y_pred_test, average="weighted")
recall_test_por_classe = recall_score(y_test, y_pred_test, average=None)

print("\n=== TESTE ===")
print("Acurácia:", acc_test)
print("Recall (macro):", recall_test_macro)
print("Recall (weighted):", recall_test_weighted)
print("Recall por classe [0,1,2]:", recall_test_por_classe)

cm_test = confusion_matrix(y_test, y_pred_test)

fig_test = ff.create_annotated_heatmap(
    z=cm_test,
    x=class_names,
    y=class_names,
    colorscale='Blues',
    showscale=True
)
fig_test.update_layout(
    title='Matriz de Confusão — TESTE (Naive Bayes)',
    xaxis_title='Predito',
    yaxis_title='Real',
    template='plotly_white'
)
fig_test.show()

"""
Análise:
Acurácia do treino de 0,984; de um total de 252 acertou 248.

O recall de cada classe demonstra a porcentagem de quantos verdadeiros foram identificados dentro
do conjunto de todos os verdadeiros da respectiva classe.

Matriz de confusão demonstra a contagem de erros e acertos de cada classe.

Métricas de treino altas podem ser indícios de overfitting, porém as métricas de testes também são altas e
isso indica que o modelo está bom. Além disso, a base de treino foi balanceada com SMOTE para 
reduzir viés do classificador em direção à classe majoritária.

A acurácia do teste foi de 0,975; de um total de 41 acertou 40.

O recall de cada classe do teste foi de 100% para as classes altas e médias.
Porém, a classe baixa teve um recall de 83% num conjunto de apenas 6 dados.
Ou seja, no universo de 6 dados teve um erro.
Pela análise desse erro percebe-se que um cliente com o score credit baixo foi classificado como médio.
Isso gera uma situação de risco, pois o cliente pode apresentar problemas financeiros.

O Teorema de Bayes mostra a probabilidade de um evento acontecer dado que outro evento aconteceu.
Tem-se uma probabilidade inicial(priori) que pode ser alterada pela ocorrência de novas evidências, isso
gera uma probabilidade a posteriori.
No contexto da ciência de dados é usado para fazer classificações. 
O objeto a ser classificado tem sua probabilidade calculada com base em evidências, que podem ser palavras, comportamentos
ou características.
Na base de estudo análisada, as evidências são características de clientes de acordo com o credit score.
O algoritmo de Naive Bayes classifica o credit score dos clientes.
As métricas de teste servem para analisar se o modelo está classificando corretamente, pois é comparada a acurácia
da base de treino com a base de teste. Isso permite a utilização do modelo em clientes novos que serão
classificados de acordo com a probabilidade gerada.
Uma frase que resume o algoritmo: Qual a probabilidade de um cliente ter um determinado credit score dado que
outras características (eventos) aconteceram. 
"""

#CROSS VALIDATION
modelo = GaussianNB()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_acc = cross_val_score(modelo, x_train, y_train, cv=cv, scoring="accuracy")
scores_rec = cross_val_score(modelo, x_train, y_train, cv=cv, scoring="recall_macro")

print("\n=== CROSS-VALIDATION (TREINO) ===")
print("Accuracy CV: média =", scores_acc.mean(), " | std =", scores_acc.std())
print("Recall macro CV: média =", scores_rec.mean(), " | std =", scores_rec.std())

#PREDICT PROBA

probas = naive_churn.predict_proba(x_test)  # matriz (n_amostras x n_classes)
preds = naive_churn.predict(x_test)

df_proba = pd.DataFrame(probas, columns=[f"P(classe={c})" for c in naive_churn.classes_])
df_proba["Real"] = y_test.values
df_proba["Predito"] = preds

# confiança = maior probabilidade
df_proba["Confiança"] = df_proba[[c for c in df_proba.columns if c.startswith("P(")]].max(axis=1)

print("\n=== AMOSTRAS COM BAIXA CONFIANÇA (TOP 10) ===")
print(df_proba.sort_values("Confiança").head(10).to_string(index=False))

limiar = 0.60
incertos = df_proba[df_proba["Confiança"] < limiar]
print(f"\nTotal de previsões incertas (conf < {limiar}):", len(incertos))
