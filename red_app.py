import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Título de la aplicación
st.title("API Perceptrón")

with st.container():
    # Definir las dos columnas
    col1, col2 = st.columns(2)

    # ============================================
    # COLUMNA IZQUIERDA: Suma de dos números
    # ============================================

    with col1:
        st.header("Conjunto de datos simulados")
        num1 = st.number_input("Ingrese número de simulaciones", value=0)
        num2 = st.number_input("Ingrese número de clases", value=0)

        if st.button("Generar"):

            x, gt = make_blobs(n_samples= num1, centers= num2, n_features=2,random_state=42)
            gt = gt.reshape((len(gt), 1)) # convertir y en vector
            #plt.scatter(x[:,0], x[:,1], c=gt, cmap=plt.cm.Paired, edgecolors='k', marker='o')
            #plt.show()

            st.session_state.x = x
            st.session_state.gt = gt
        
            #st.write(f"El triple de {num1} es: {gt}")
            fig, ax = plt.subplots()
            ax.scatter(x[:, 0], x[:, 1], c=gt, cmap=plt.cm.Paired, edgecolors='k', marker='o')
        
            # Mostrar gráfico en Streamlit
            st.pyplot(fig)

    # ============================================
    # COLUMNA DERECHA: Triplicar un número
    # ============================================
    with col2:
        st.header("Perceptrón")
        num3 = st.number_input("Ingrese taza de aprendizaje", value=0.0, step=0.01)
        num4 = st.number_input("Ingreso numero de iteraciones", value=0)
        num5 = st.number_input("Ingreso numero de clases", value=0)

        if st.button("Estimate Neural Network"):
            class PerceptronMulticlass:
                def __init__(self, num3, num4, num5):
                    self.lr = num3
                    self.n_iters = num4
                    self.n_class = num5
                    self.activation_func = self._unit_step_func
                    self.weights = None
                    self.bias = None

                def _unit_step_func(self, x):
                    return np.where(x >= 0, 1, 0)

                def fit(self, X, y):
                    n_samples, n_features = X.shape
                    # Inicialización de parámetros
                    self.weights = np.zeros((self.n_class, n_features))  # Matriz de pesos para cada clase
                    self.bias = np.zeros(self.n_class)  # Vector de sesgos para cada clase

                    # Iteración a través de épocas
                    for _ in range(self.n_iters):
                        for i, x_i in enumerate(X):  # Para cada muestra
                            for idx in range(self.n_class):  # Para cada clase
                                y_ = 1 if y[i] == idx else 0  # Etiqueta binaria "One vs Rest"

                                # Salida lineal para la clase idx
                                linear_output = np.dot(x_i, self.weights[idx]) + self.bias[idx]
                                y_predicted = self.activation_func(linear_output)  # Predicción binaria

                                # Actualización de pesos y bias si la predicción no coincide con la etiqueta real
                                update = self.lr * (y_ - y_predicted)
                                self.weights[idx] += update * x_i
                                self.bias[idx] += update

                def predict(self, X):
                    # Asegúrate de que X sea un arreglo 2D
                    if X.ndim == 1:
                        X = X.reshape(1, -1)  # Cambiar a forma (1, n_features)

                    linear_outputs = np.dot(X, self.weights.T) + self.bias  # (n_samples, n_classes)
                    y_predicted = np.argmax(linear_outputs, axis=1)
    
                    return y_predicted

                def plot_decision_boundary(self, X, y, title="Frontera de Decisión del Perceptrón"):
                    plt.figure(figsize=(8, 6))

                    # Scatter plot para puntos de datos
                    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')

                    # Definir límites para el gráfico
                    x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
                    y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1

                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
                    Z = np.array([self.predict(np.array([xi, yi])) for xi, yi in zip(xx.ravel(), yy.ravel())])
                    Z = Z.reshape(xx.shape)

                    # Graficar las fronteras de decisión
                    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)

                    plt.title(title)
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                    st.pyplot(plt)

            # Comprobar si x y gt están en session_state
            if 'x' in st.session_state and 'gt' in st.session_state:
                x = st.session_state.x
                gt = st.session_state.gt

                # Entrenar el perceptrón utilizando los datos generados
                perceptron = PerceptronMulticlass(num3, num4, num5)
                perceptron.fit(x, gt)

                # Mostrar la frontera de decisión
                perceptron.plot_decision_boundary(x, gt)
            else:
                st.warning("Primero genere los datos en la columna izquierda.")