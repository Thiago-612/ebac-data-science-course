# EBAC Data Science Course

Repositório com projetos, exercícios e estudos desenvolvidos durante a formação profissional em **Ciência de Dados pela EBAC**.

O conteúdo registra minha evolução ao longo do curso, desde coleta, tratamento e visualização de dados até Machine Learning, redução de dimensionalidade, Processamento de Linguagem Natural e desenvolvimento de modelos orientados a problemas de negócio.

---

## Projetos em Destaque

### 1. Detecção de Fraudes em Cartão de Crédito

**Projeto Final — Ciência de Dados**

[🔗 Acessar projeto](./Módulo%2043%20Projeto%20Final)

Desenvolvimento de um sistema de classificação para identificação de transações fraudulentas em uma base altamente desbalanceada.

Foi construído um **ensemble customizado orientado a risco**, com avaliação baseada em métricas adequadas ao problema de fraude e análise dos custos associados a falsos positivos e falsos negativos.

#### Principais resultados

| Métrica            |  Resultado |
| ------------------ | ---------: |
| Recall — Fraude    | **83,67%** |
| Precision — Fraude | **79,61%** |
| F1-score           | **81,59%** |
| ROC AUC            | **97,32%** |
| Average Precision  | **82,74%** |
| Fraudes detectadas |     **82** |
| Falsos negativos   |     **16** |
| Falsos positivos   |     **21** |

#### Técnicas utilizadas

* Classificação supervisionada
* Ensemble Learning
* Tratamento de classes desbalanceadas
* Matriz de confusão
* Precision-Recall Curve
* ROC Curve
* ROC AUC
* Average Precision
* Análise de threshold
* Modelagem orientada a risco

**Tecnologias:** Python · Pandas · NumPy · Scikit-learn · Matplotlib

---

### 2. Classificação de Tópicos de Notícias com NLP

**Processamento de Linguagem Natural**

[🔗 Acessar projeto](./Módulo%2042%20PLN)

Projeto de classificação automática de notícias a partir de seus títulos utilizando técnicas de **Natural Language Processing — NLP**.

Foi desenvolvido um pipeline envolvendo limpeza e normalização dos textos, transformação com **TF-IDF**, treinamento de modelos de Regressão Logística e Support Vector Machine e otimização de hiperparâmetros.

#### Principais etapas

* Limpeza e normalização de textos
* TF-IDF Vectorization
* Logistic Regression
* Linear SVM
* CalibratedClassifierCV
* Pipeline do Scikit-learn
* GridSearchCV
* Stratified K-Fold
* Cross Validation
* Comparação entre modelos

O **SVM** apresentou o melhor desempenho geral, alcançando aproximadamente:

* **92,7% de acurácia em validação cruzada**
* **91,6% de F1-macro**

O projeto demonstra a aplicação de Machine Learning na transformação de dados textuais não estruturados em informações utilizáveis para classificação automática.

**Tecnologias:** Python · Pandas · Scikit-learn · NLP · TF-IDF · SVM · Logistic Regression

---

### 3. Predição de Compras Online com PCA e Random Forest

**Machine Learning e Redução de Dimensionalidade**

[🔗 Acessar projeto](./Módulo%2037%20PCA)

Projeto de Machine Learning aplicado a dados de campanhas de marketing, com comparação entre diferentes pipelines para predição do comportamento de compras online.

Foram comparados modelos de **Regressão Logística e Random Forest**, com e sem aplicação de **Principal Component Analysis — PCA**.

#### Pipeline desenvolvido

* Tratamento de valores ausentes
* StandardScaler
* One-Hot Encoding
* ColumnTransformer
* PCA
* Logistic Regression
* Random Forest
* GridSearchCV
* Stratified Cross Validation
* Análise de métricas

O **Random Forest sem PCA** apresentou o melhor desempenho no conjunto de teste:

| Métrica  |  Resultado |
| -------- | ---------: |
| Recall   | **96,02%** |
| F1-score | **92,14%** |
| ROC AUC  | **97,42%** |

O experimento também demonstrou que a redução de dimensionalidade com PCA não necessariamente melhora o desempenho preditivo, reforçando a importância da comparação empírica entre diferentes pipelines.

**Tecnologias:** Python · Pandas · Scikit-learn · PCA · Random Forest · Logistic Regression · Plotly

---

# Conteúdo da Formação

## Fundamentos de Dados

### Módulo 7 — Coleta de Dados

[🔗 Acessar](./Módulo%207%20ColetaDados)

Estudos e exercícios envolvendo diferentes métodos de aquisição de dados.

**Conteúdo:**

* Pandas
* APIs
* Web scraping
* Beautiful Soup
* Dados financeiros com yfinance
* Manipulação de DataFrames

---

### Módulo 8 — Tratamento de Dados

[🔗 Acessar](./Módulo%208%20TratamentoDados)

Aplicação de técnicas de limpeza e tratamento de inconsistências.

**Conteúdo:**

* Valores ausentes
* Inconsistências
* Outliers
* Funções lambda
* Limpeza e transformação de dados

---

### Módulo 9 — Preparação de Dados

[🔗 Acessar](./Módulo%209%20PreparaçãoDados)

Preparação de dados para análise e modelagem.

**Conteúdo:**

* Análise exploratória
* Normalização
* Padronização
* Codificação de variáveis categóricas
* Feature transformation

---

### Módulo 10 — Visualização de Dados

[🔗 Acessar](./Módulo%2010%20VisualizaçãoDados)

Visualização e exploração gráfica de dados.

**Tecnologias:** Matplotlib · Seaborn · Pandas

---

# Estatística e Pré-Modelagem

### Módulo 14 — Pré-Modelagem: Churn

[🔗 Acessar](./Módulo%2014%20Pré%20Modelagem)

Preparação e análise de uma base de clientes para estudo de churn.

---

### Módulo 15 — Pré-Modelagem

[🔗 Acessar](./Módulo%2015%20Pré%20Modelagem)

Continuação do processo de preparação de dados para modelagem.

---

### Módulo 17 — Credit Score: Preparação dos Dados

[🔗 Acessar](./Módulo%2017%20Pré%20Modelagem)

Preparação de uma base de Credit Score para classificação multiclasse.

**Técnicas:**

* Encoding
* Separação treino/teste
* Balanceamento de classes
* SMOTE
* Preparação para Machine Learning

---

### Módulo 18 — Regressão Linear

[🔗 Acessar](./Módulo%2018%20Regressão%20Linear)

Aplicação de Regressão Linear na previsão de valores de aluguel.

---

### Módulo 19 — Estatística Aplicada

[🔗 Acessar](./Módulo%2019%20Estatística%20Aplicada)

Aplicação de conceitos estatísticos no contexto de Ciência de Dados.

---

# Machine Learning

### Módulo 20 — Naive Bayes

[🔗 Acessar](./Módulo%2020%20Aprendizagem%20Baysiana)

Aplicação de classificação probabilística com Naive Bayes sobre o problema de Credit Score.

---

### Módulo 21 — Árvore de Decisão

[🔗 Acessar](./Módulo%2021%20Arvore%20decisão)

Classificação multiclasse de Credit Score utilizando Decision Tree.

O projeto envolveu:

* Modelo baseline
* Feature Importance
* Seleção de variáveis
* GridSearchCV
* Validação cruzada
* Análise de overfitting
* Matriz de confusão
* Correção de data leakage

O modelo final atingiu aproximadamente **97,56% de acurácia no conjunto de teste**.

---

# SQL e Business Intelligence

### Módulo 24 — SQL

[🔗 Acessar](./Módulo%2024%20SQL)

Consultas e manipulação de dados utilizando SQL.

---

### Módulo 25 — SQL II

[🔗 Acessar](./Módulo%2025%20SQL%20II)

Continuação dos estudos de banco de dados e consultas SQL.

---

### Módulo 26 — SQL III e Power BI

[🔗 Acessar](./Módulo%2026%20SQL%20III)

Projeto envolvendo dados de e-commerce, estruturação de tabelas e visualização em Power BI.

**Conteúdo:**

* SQL
* Full Join
* Tabela fato
* Tabelas dimensão
* Modelagem de dados
* Power BI

---

# Modelos de Classificação e Agrupamento

### Módulo 27 — Regressão Logística

[🔗 Acessar](./Módulo%2027%20Regressão%20Logística)

Projeto aplicado a uma base cardiovascular.

**Técnicas:**

* Logistic Regression
* Padronização
* SMOTE
* ROC AUC
* Odds Ratio
* Threshold
* Análise de multicolinearidade

---

### Módulo 30 — Agrupamento com K-Means

[🔗 Acessar](./Módulo%2030%20Agrupamento%20Kmeans)

Aplicação inicial de técnicas de clustering.

---

### Módulo 32 — Random Forest

[🔗 Acessar](./Módulo%2032%20Random%20Forest)

Classificação utilizando Random Forest, avaliação de desempenho e análise de importância das variáveis.

**Técnicas:**

* Random Forest
* Feature Importance
* Feature Selection
* Matriz de confusão
* Hyperparameter Tuning

---

### Módulo 33 — Segmentação de Clientes com K-Means

[🔗 Acessar](./Módulo%2033%20Kmeans)

Segmentação de clientes utilizando o dataset Mall Customers.

**Técnicas:**

* K-Means
* Clusterização
* Análise de centroides
* Segmentação de clientes

---

### Módulo 34 — Regressão Polinomial

[🔗 Acessar](./Módulo%2034%20Regressão%20Polinomial)

Comparação de modelos de regressão aplicados à previsão de aluguel.

**Conteúdo:**

* Regressão Linear
* Regressão Polinomial
* Ridge
* Lasso
* ElasticNet
* VIF
* Análise de overfitting

---

# Validação e Redução de Dimensionalidade

### Módulo 35 — Cross Validation

[🔗 Acessar](./Módulo%2035%20Cross%20Validation)

Projeto de classificação de alarmes utilizando dados de sensores IoT.

**Técnicas:**

* Cross Validation
* Random Forest
* GridSearchCV
* Feature Importance
* Comparação de modelos

---

### Módulo 36 — Classificação

[🔗 Acessar](./Módulo%2036)

Exercícios utilizando o dataset Fetal Health.

---

### Módulo 37 — PCA

[🔗 Acessar](./Módulo%2037%20PCA)

Comparação de pipelines com e sem redução de dimensionalidade utilizando Principal Component Analysis.

---

# Modelos Avançados

### Módulo 39 — XGBoost

[🔗 Acessar](./Módulo%2039%20Xgboost)

Classificação com Gradient Boosting utilizando XGBoost.

**Técnicas:**

* XGBoost
* GridSearchCV
* Cross Validation
* Feature Importance
* Hyperparameter Tuning

---

### Módulo 40 — Support Vector Machine

[🔗 Acessar](./Módulo%2040%20SVM)

Aplicação de Support Vector Machine em problemas de classificação.

**Técnicas:**

* SVM
* Cross Validation
* Feature Importance
* Hyperparameter Optimization

---

### Módulo 41 — Comparação entre Modelos e Ensemble

[🔗 Acessar](./Módulo%2041%20Xgbost%20X%20SVM)

Comparação entre diferentes algoritmos de Machine Learning e desenvolvimento de ensemble.

**Modelos avaliados:**

* XGBoost
* SVM
* Logistic Regression
* Soft Voting Ensemble

O projeto incluiu avaliação por Train/Test Split e Cross Validation.

---

# Processamento de Linguagem Natural

### Módulo 42 — NLP

[🔗 Acessar](./Módulo%2042%20PLN)

Projeto de classificação de textos utilizando TF-IDF, Regressão Logística e SVM.

---

# Projeto Final

### Módulo 43 — Detecção de Fraudes em Cartão de Crédito

[🔗 Acessar](./Módulo%2043%20Projeto%20Final)

Projeto final da formação, reunindo conceitos de preparação de dados, modelagem, avaliação, comparação de algoritmos e tomada de decisão orientada a risco.

---

# Principais Tecnologias

### Linguagens e manipulação de dados

* Python
* SQL
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* Logistic Regression
* Decision Trees
* Random Forest
* Naive Bayes
* Support Vector Machines
* K-Means
* Ensemble Learning

### Estatística e modelagem

* Regressão Linear
* Regressão Polinomial
* PCA
* Cross Validation
* GridSearchCV
* Feature Engineering
* Feature Selection
* SMOTE

### Visualização

* Matplotlib
* Seaborn
* Plotly
* Power BI

### NLP

* TF-IDF
* Text Preprocessing
* Logistic Regression
* SVM

### Ferramentas

* Git
* GitHub
* Jupyter Notebook
* Visual Studio Code

---

# Estrutura do Repositório

```text
ebac-data-science-course/
│
├── Módulo 7 ColetaDados/
├── Módulo 8 TratamentoDados/
├── Módulo 9 PreparaçãoDados/
├── Módulo 10 VisualizaçãoDados/
│
├── Módulo 14 Pré Modelagem/
├── Módulo 15 Pré Modelagem/
├── Módulo 17 Pré Modelagem/
├── Módulo 18 Regressão Linear/
├── Módulo 19 Estatística Aplicada/
├── Módulo 20 Aprendizagem Baysiana/
├── Módulo 21 Arvore decisão/
│
├── Módulo 24 SQL/
├── Módulo 25 SQL II/
├── Módulo 26 SQL III/
├── Módulo 27 Regressão Logística/
│
├── Módulo 30 Agrupamento Kmeans/
├── Módulo 32 Random Forest/
├── Módulo 33 Kmeans/
├── Módulo 34 Regressão Polinomial/
├── Módulo 35 Cross Validation/
├── Módulo 36/
├── Módulo 37 PCA/
│
├── Módulo 39 Xgboost/
├── Módulo 40 SVM/
├── Módulo 41 Xgbost X SVM/
├── Módulo 42 PLN/
└── Módulo 43 Projeto Final/
```

---

# Dados de Grande Porte

O dataset utilizado no projeto final de detecção de fraudes possui aproximadamente **150 MB** e é versionado utilizando **Git Large File Storage — Git LFS**.

Após clonar o repositório, usuários que desejarem obter o dataset completo devem possuir o Git LFS instalado.

```bash
git lfs install
git clone https://github.com/Thiago-612/ebac-data-science-course.git
```

---

# Projeto Complementar

Além dos projetos desenvolvidos durante a formação, também desenvolvi um projeto independente aplicado ao mercado imobiliário:

## Análise de Oportunidades Imobiliárias em Brasília e Entorno

Pipeline de Ciência de Dados aplicado à avaliação de oportunidades de investimento imobiliário, envolvendo coleta de anúncios públicos, tratamento e validação de dados, modelagem preditiva de aluguel e simulação de viabilidade financeira.

[🔗 Acessar Projeto Imobiliário](https://github.com/Thiago-612/projeto-imobiliario-brasilia)

---

# Autor

**Thiago Patrício**

Cientista de Dados | Engenheiro Civil | Licenciado em Matemática

* GitHub: [github.com/Thiago-612](https://github.com/Thiago-612)
* LinkedIn: [linkedin.com/in/thiago-patricio-data-science](https://www.linkedin.com/in/thiago-patricio-data-science/)

---

## Sobre este repositório

Este repositório tem finalidade acadêmica e de portfólio, documentando a evolução técnica ao longo da formação em Ciência de Dados.

Os projetos foram desenvolvidos com foco não apenas na aplicação de algoritmos, mas também na compreensão dos problemas, preparação adequada dos dados, comparação de abordagens, validação dos modelos e interpretação dos resultados.
