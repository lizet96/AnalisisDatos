import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

datos = pd.read_csv(filepath_or_buffer = "/users/Liz/comprar_alquilar.csv")
datos.head()
datos.dropna(inplace=True)

X = datos.drop('comprar', axis=1)
y = datos['comprar']

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_lda.flatten(), y=[0]*len(X_lda), hue=y, palette='coolwarm')
plt.title('Reducción LDA: Proyección 1D para Comprar/Alquilar')
plt.xlabel('Componente LDA')
plt.yticks([])
plt.show()