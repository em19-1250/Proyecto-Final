# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos
url = "https://raw.githubusercontent.com/datasets/housing-prices/main/housing.csv"  # Cambia a la ruta de Kaggle si la descargas
df = pd.read_csv(url)

# Exploración inicial
print(df.head())
print(df.info())
print(df.describe())

# Visualización de valores nulos
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Valores nulos en el dataset")
plt.show()

# Manejo de valores nulos (relleno con la media para este ejemplo)
df.fillna(df.mean(), inplace=True)

# Selección de variables (simplificando para un ejemplo práctico)
X = df.drop(columns=['Price'])  # Variables predictoras
y = df['Price']  # Variable objetivo

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### 1. Modelo de Regresión Lineal
# Entrenamiento
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predicción
y_pred_linear = linear_model.predict(X_test_scaled)

# Evaluación
mse = mean_squared_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

print("Regresión Lineal:")
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}\n")

### 2. Modelo de Regresión Logística
# Preparar datos para clasificación
# Clasificamos como "1" casas con precio por encima de la mediana, "0" en caso contrario
y_class = (y > y.median()).astype(int)
y_train_class = y_class[y_train.index]
y_test_class = y_class[y_test.index]

# Entrenamiento
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train_class)

# Predicción
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_pred_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]

# Evaluación
accuracy = accuracy_score(y_test_class, y_pred_logistic)
auc_roc = roc_auc_score(y_test_class, y_pred_proba)

print("Regresión Logística:")
print(f"Precisión (Accuracy): {accuracy}")
print(f"AUC-ROC: {auc_roc}")
print("\nInforme de Clasificación:")
print(classification_report(y_test_class, y_pred_logistic))

### Comparación de modelos
print("\n--- Comparación de Modelos ---")
print(f"Regresión Lineal - R^2: {r2}")
print(f"Regresión Logística - AUC-ROC: {auc_roc}")

# Visualización de resultados
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_linear)
plt.title("Regresión Lineal - Precio Real vs Predicho")
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")

plt.subplot(1, 2, 2)
sns.histplot(y_pred_proba, kde=True, bins=20, color='orange')
plt.title("Regresión Logística - Distribución de Probabilidades")
plt.xlabel("Probabilidad Predicha")
plt.ylabel("Frecuencia")
plt.show()
