# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Cargar los datasets
ruta_train = r"C:\Users\Emil Martinez\Desktop\Piton-assemble\Proyecto-Final\train.csv"
ruta_test = r"C:\Users\Emil Martinez\Desktop\Piton-assemble\Proyecto-Final\test.csv"
ruta_sample = r"C:\Users\Emil Martinez\Desktop\Piton-assemble\Proyecto-Final\sample_submission.csv"

df_train = pd.read_csv(ruta_train)
df_test = pd.read_csv(ruta_test)
df_sample = pd.read_csv(ruta_sample)

# Inspección inicial
print("Train Dataset:")
print(df_train.head())
print(df_train.info())
print("Test Dataset:")
print(df_test.head())
print(df_test.info())
print("Sample Submission:")
print(df_sample.head())

# Limpieza y preparación de datos
# Verificamos si hay valores nulos en el conjunto de entrenamiento
print("Valores nulos en train.csv:")
print(df_train.isnull().sum())

# Rellenar valores nulos con la media (o algún otro enfoque)
df_train.fillna(df_train.mean(), inplace=True)

# Seleccionar las columnas predictoras y la columna objetivo
X = df_train.drop(columns=["SalePrice"])  # Cambiar "SalePrice" según el dataset
y = df_train["SalePrice"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### 1. Modelo de Regresión Lineal
# Entrenar el modelo
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predicción
y_pred_linear = linear_model.predict(X_test_scaled)

# Evaluación
mse = mean_squared_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

print("\nRegresión Lineal:")
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")

### 2. Modelo de Regresión Logística
# Convertir el problema a clasificación: Categorías "1" (caras) y "0" (baratas)
y_class = (y > y.median()).astype(int)
y_train_class = y_class[y_train.index]
y_test_class = y_class[y_test.index]

# Entrenar el modelo
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train_class)

# Predicción
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_pred_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]

# Evaluación
accuracy = accuracy_score(y_test_class, y_pred_logistic)
auc_roc = roc_auc_score(y_test_class, y_pred_proba)

print("\nRegresión Logística:")
print(f"Precisión (Accuracy): {accuracy}")
print(f"AUC-ROC: {auc_roc}")
print("\nInforme de Clasificación:")
print(classification_report(y_test_class, y_pred_logistic))

### Comparación de Modelos
print("\n--- Comparación de Modelos ---")
print(f"Regresión Lineal - R^2: {r2}")
print(f"Regresión Logística - AUC-ROC: {auc_roc}")

# Visualización de Resultados
plt.figure(figsize=(14, 6))

# Gráfico 1: Regresión Lineal - Real vs Predicho
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_linear, alpha=0.7, edgecolor=None)
plt.title("Regresión Lineal - Precio Real vs Predicho")
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")

# Gráfico 2: Regresión Logística - Distribución de Probabilidades
plt.subplot(1, 2, 2)
sns.histplot(y_pred_proba, kde=True, bins=20, color='orange')
plt.title("Regresión Logística - Distribución de Probabilidades")
plt.xlabel("Probabilidad Predicha")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()
