import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Cargar los datos
train_path = r'C:\Users\Emil Martinez\Desktop\Piton-assemble\Proyecto-Final\train.csv'
test_path = r'C:\Users\Emil Martinez\Desktop\Piton-assemble\Proyecto-Final\test.csv'
submission_path = r'C:\Users\Emil Martinez\Desktop\Piton-assemble\Proyecto-Final\sample_submission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Separar características y la variable objetivo
X = train.drop('TARGET(PRICE_IN_LACS)', axis=1)
y = train['TARGET(PRICE_IN_LACS)']

# Convertir columnas categóricas en variables dummies
X = pd.get_dummies(X, drop_first=True)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regresión lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Calcular el error cuadrático medio (MSE)
mse_linear = mean_squared_error(y_test, y_pred_linear)

# Modelo de regresión logística
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, np.round(y_train))  # Asegúrate de redondear los valores objetivo para la regresión logística
y_pred_logistic = logistic_model.predict(X_test)

# Calcular la precisión (Accuracy)
accuracy_logistic = accuracy_score(np.round(y_test), y_pred_logistic)

# Imprimir resultados
print(f'Linear Regression MSE: {mse_linear}')
print(f'Logistic Regression Accuracy: {accuracy_logistic}')
print('\nClassification Report:\n')
print(classification_report(np.round(y_test), y_pred_logistic))

# Graficar resultados
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression: Actual vs Predicted')

plt.subplot(1, 2, 2)
plt.scatter(np.round(y_test), y_pred_logistic, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Logistic Regression: Actual vs Predicted')

plt.show()
