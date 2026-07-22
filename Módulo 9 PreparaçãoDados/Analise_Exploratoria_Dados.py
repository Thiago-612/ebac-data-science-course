import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('clientes-v2.csv')
# Realizando EDA
print(data.describe())
sns.histplot(data['idade'])
plt.show()
