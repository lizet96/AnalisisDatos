import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

datos = pd.read_csv(filepath_or_buffer = "/users/Liz/Downloads/breast-cancer.csv")
datos.head()

datos = datos.drop('id', axis=1)
datos['diagnosis'] = datos['diagnosis'].map({'M': 1, 'B': 0})
datos.dropna(inplace=True)
print(datos.describe())

X = datos.drop('diagnosis', axis=1)
y = datos['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(modelo, filled=True, feature_names=X.columns, class_names=['B', 'M'])
plt.title('Árbol de Decisión para Diagnóstico de Cáncer')
plt.show()
