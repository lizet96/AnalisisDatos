import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

datos = pd.read_csv(filepath_or_buffer = "/users/Liz/Downloads/samsung.csv")
datos.head()

print("Columnas del DataFrame:", datos.columns.tolist())

print(datos.head())

print(datos.isnull().sum())

try:
    datos['Date'] = pd.to_datetime(datos['Date'], format='%d/%m/%Y')
except KeyError:
    print("Error: La columna 'Date' no existe. Verifica los nombres de las columnas.")
datos = datos[['Close', 'Volume']]
datos.dropna(inplace=True)

scaler = StandardScaler()
datos_scaled = scaler.fit_transform(datos)

modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
modelo.fit(datos_scaled)
datos['Cluster'] = modelo.labels_

plt.figure(figsize=(10, 6))
sns.scatterplot(x=datos['Close'], y=datos['Volume'], hue=datos['Cluster'], palette='viridis')
plt.title('Clustering K-Means en Datos de Samsung (Close vs Volume)')
plt.xlabel('Close Price')
plt.ylabel('Volume')
plt.show()