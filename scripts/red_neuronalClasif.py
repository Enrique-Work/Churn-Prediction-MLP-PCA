import numpy as np

class RedNeuronalClsif:
    
    def sigmoid(self,z):
        res = 1 / (1 + np.exp(-z))
        return res

    def layer_sizes(self, X, Y):
        """
        Calcula el tamaño de las capas de la red neuronal.

        Argumentos:
        X -- dataset de entrada de forma (n_x, m)
        Y -- etiquetas de forma (n_y, m)
        
        Retorna:
        n_x -- tamaño de la capa de entrada
        n_h -- tamaño de la capa oculta
        n_y -- tamaño de la capa de salida
        """
        # Obtenemos el tamaño de entrada de la primera dimensión de X (filas)
        n_x = X.shape[0] 
        
        # Definimos 2 neuronas ocultas
        n_h = 2 
        
        # Obtenemos el tamaño de salida de la primera dimensión de Y
        n_y = Y.shape[0] 
        
        return (n_x, n_h, n_y)

    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argumentos:
        n_x -- tamaño de la capa de entrada
        n_h -- tamaño de la capa oculta
        n_y -- tamaño de la capa de salida
        """
        # semilla fijada al azar
        np.random.seed(3) 
        
        # Inicialización de pesos y sesgos usando 'self'
        self.W1 = np.random.randn(n_h, n_x) * 0.01
        self.b1 = np.zeros((n_h, 1))
        self.W2 = np.random.randn(n_y, n_h) * 0.01
        self.b2 = np.zeros((n_y, 1))
        
        # Ayudan a que si algo sale mal con los 5 inputs, el código se detenga ahí mismo.
        assert (self.W1.shape == (n_h, n_x))
        assert (self.b1.shape == (n_h, 1))
        assert (self.W2.shape == (n_y, n_h))
        assert (self.b2.shape == (n_y, 1))    


    def forward_propagation(self, X):
        """
        Argumento:
        X -- datos de entrada de tamaño (n_x, m)
        
        Retorna:
        A2 -- la salida sigmoide de la segunda activación (predicciones)
        """
        # Implementamos el cálculo de la Capa 1
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(self.Z1) 
        
        # Implementamos el cálculo de la Capa 2
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        # El assert ahora usa self.W2.shape[0] para saber n_y
        assert(self.A2.shape == (self.W2.shape[0], X.shape[1]))
        

        return self.A2     


    def compute_cost(self, A2, Y):
        """
        Calcula la función de costo Log Loss.

        Argumentos:
        A2 -- La salida de la red (predicciones) de forma (1, m)
        Y -- Vector de etiquetas reales de forma (1, m)
        
        Retorna:
        cost -- el valor escalar de la pérdida logarítmica
        """
        # Número de ejemplos de entrenamiento
        m = Y.shape[1]
        
        # Calculamos el log loss elemento por elemento
        # Usamos np.multiply o simplemente '*' para el producto de Hadamard
        logloss = - (Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
        
        # Promediamos la suma de todas las pérdidas
        cost = np.sum(logloss) / m
        
        # Forzamos que el resultado sea un float escalar (por si acaso viene como array)
        cost = float(np.squeeze(cost))
        
        assert(isinstance(cost, float))
        
        return cost
    
    def backward_propagation(self, X, Y):
        """
        Implementa el backpropagation para calcular gradientes.
        
        Argumentos:
        X -- datos de entrada de forma (n_x, m)
        Y -- etiquetas reales de forma (n_y, m)
        
        Retorna:
        No retorna nada, guarda los gradientes en 'self'
        """
        m = X.shape[1]
        
        # 1. Cálculo del error en la salida (Capa 2)
        # dZ2 es (n_y, m)
        dZ2 = self.A2 - Y
        
        # 2. Gradientes de la Capa 2
        # dW2 será (n_y, n_h) -> (1, 2)
        self.dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        self.db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # 3. Cálculo del error en la capa oculta (Capa 1)
        # dZ1 será (n_h, m) -> (2, m)
        dZ1 = np.dot(self.W2.T, dZ2) * self.A1 * (1 - self.A1)
        
        # 4. Gradientes de la Capa 1
        self.dW1 = (1/m) * np.dot(dZ1, X.T)
        self.db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)



    def update_parameters(self, learning_rate=1.2):
        """
        Actualiza los parámetros usando la regla del descenso del gradiente.
        
        Argumentos:
        learning_rate -- tasa de aprendizaje para el descenso del gradiente
        """
        # Actualización de la Capa 1
        self.W1 = self.W1 - learning_rate * self.dW1
        self.b1 = self.b1 - learning_rate * self.db1
        
        # Actualización de la Capa 2
        self.W2 = self.W2 - learning_rate * self.dW2
        self.b2 = self.b2 - learning_rate * self.db2
        
        # No hace falta return, los cambios ya están guardados en el objeto 'self'

    def fit(self, X, Y, n_h=2, num_iterations=10, learning_rate=1.2, print_cost=False):
        """
        Entrena el modelo y resetea los parámetros al iniciar.
        """
        # 1. Detectamos tamaños de capa
        n_x, _, n_y = self.layer_sizes(X, Y)
        
        # 2. RESETEAR PESOS: Llamamos a la función que ya definimos arriba
        self.initialize_parameters(n_x, n_h, n_y)
        
        # 3. Bucle de entrenamiento (Gradiente Descendente)
        for i in range(0, num_iterations):
             
            # Forward propagation.
            A2 = self.forward_propagation(X)
            
            # Cálculo del costo.
            cost = self.compute_cost(A2, Y)
            
            # Backpropagation (calcula gradientes y los guarda en self).
            self.backward_propagation(X, Y)
            
            # Actualización de parámetros (usa los gradientes de self).
            self.update_parameters(learning_rate)
            
            # Imprimir el costo.
            if print_cost and i % 100 == 0:
                print(f"Costo después de la iteración {i}: {cost}")

        
        return self




    def predict(self, X):
        """
        Utiliza los parámetros entrenados en 'self' para predecir la clase.
        
        Argumentos:
        X -- datos de entrada de forma (n_x, m)
        
        Retorna:
        predictions -- vector de predicciones (True/1 si > 0.5, False/0 en caso contrario)
        """
        # 1. Ejecutamos el forward propagation con los pesos actuales de la red
        A2 = self.forward_propagation(X)
        
        # 2. Convertimos las probabilidades en clases binarias (0 o 1)
        # Si A2 > 0.5, la predicción es True (1), de lo contrario es False (0)
        predictions = (A2 > 0.5)
        
        return predictions

