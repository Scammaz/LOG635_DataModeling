from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import pickle

class NeuralNetwork():
    def __init__(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_nodes_per_layer, learning_rate):
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.nb_hidden_layers = nb_hidden_layers
        self.nb_nodes_per_layer = nb_nodes_per_layer
        self.learning_rate = learning_rate
        self.loss = []
        self.A = np.zeros(nb_hidden_layers+1)
        self.Z = np.zeros(nb_hidden_layers+1)
        
        # Weight and Bias
        self.init_weights_bias()
        
    
    def init_weights_bias(self):
        # Weights
        self.Wi = np.random.randn(self.nb_inputs, self.nb_nodes_per_layer)
        self.Wo = np.random.randn(self.nb_nodes_per_layer, self.nb_outputs)
        self.Wh = []
            
        # Bias
        self.bi = np.ones(self.nb_nodes_per_layer)
        self.bo = np.ones(self.nb_outputs)
        self.bh = []
        
        #Fill weight and bias for hidden layers
        for i in range(self.nb_hidden_layers - 1):
            self.Wh.append(np.random.randn(self.nb_nodes_per_layer, self.nb_nodes_per_layer))
            self.bh.append(np.ones(self.nb_nodes_per_layer))
         
        
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s
    
    def derived_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def softmax(self, z):
        s = (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
        return s
    
    def entropy_loss(self, y, y_pred):
        eps = np.finfo(float).eps
        return -np.sum(y * np.log(y_pred + eps))
    
    def forward(self, X):
        # Forward propagation through our network
        # input to hidden
        self.A0 = X
        self.Z[0] = np.dot(self.A0, self.Wi) + self.bi
        self.A[0] = self.sigmoid(self.Z[0])
        
        #traverse hidden
        for i in range(1, self.nb_hidden_layers):
            self.Z[i] = np.dot(self.A[i-1], self.Wh[i-1]) + self.bh[i-1]
            self.A[i] = self.sigmoid(self.Z[i])
        
        # hidden to output
        self.Z[self.nb_hidden_layers] = np.dot(self.A[self.nb_hidden_layers - 1], self.Wo) + self.bo
        self.A[self.nb_hidden_layers] = self.sigmoid(self.Z[self.nb_hidden_layers])
        
        return self.A[self.nb_hidden_layers]
    
    def backward(self, X, y):
        m = X.shape[0]  # number of examples
        
        dZ = np.zeros(self.nb_hidden_layers + 1)
        dA = np.zeros(self.nb_hidden_layers + 1)
        dW = np.zeros(self.nb_hidden_layers + 1)
        db = np.zeros(self.nb_hidden_layers + 1)
        
        #Output layers
        # Error in output
        dZ[self.nb_hidden_layers] = self.A[self.nb_hidden_layers]-y
        # Delta for the weights w2
        dW[self.nb_hidden_layers - 1] = (1./m) * np.dot(self.A[self.nb_hidden_layers - 1].T, dZ[self.nb_hidden_layers])
        # Delta for the bias b2
        db[self.nb_hidden_layers - 1] = np.sum(dZ[self.nb_hidden_layers], axis=0)  # sum across columns
        # Weights/bias update
        self.Wo -= self.learning_rate * dW[self.nb_hidden_layers]
        self.bo -= self.learning_rate * db[self.nb_hidden_layers]
        
        #Hidden layers
        for i in reversed(range(1, self.nb_hidden_layers)):
            # Error in hidden layer n
            dA[i] = np.dot(dZ[i], self.Wh[i].T)
            dZ[i] = self.A[i] * self.derived_sigmoid(self.Z[i])
            # Delta for the weights wn
            dW[i] = (1./m) * np.dot(self.A[i].T, dZ[i])
            # Delta for the bias b2
            db[i] = np.sum(dZ[i], axis=0)  # sum across columns
            # Update weights/bias
            self.Wh[] -= self.learning_rate * dW[i]
            self.bh[] -= self.learning_rate * db[i]

        #Input layers
        # d2
        dA[1] = np.dot(dZ[2], self.Wh[1].T)
        dZ[1] = dA[1] * self.derived_sigmoid(self.Z[1])
        # Delta for the weights w1
        dW[0] = (1./m) * np.dot(X.T, dZ[1])
        # Delta for the bias b1
        db[0] = (1./m) * np.sum(dZ[1], axis=0)  # sum across columns

        # Wights and biases update
        self.Wi -= self.learning_rate * dW[0]
        self.bi -= self.learning_rate * db[0]

    def train (self, X, y, nb_iterations):
        
        for i in range(nb_iterations):
            y_pred = self.forward(X)
            loss = self.entropy_loss(y, y_pred)
            self.loss.append(loss)
            self.backward(X, y)

            if i == 0 or i == nb_iterations-1:
                print(f"Iteration: {i+1}")
                print(tabulate(zip(X, y, [np.round(y_pred) for y_pred in self.A3] ), headers=["Input", "Actual", "Predicted"]))
                print(f"Loss: {loss}")                
                print("\n")

    def predict(self, X):
        return np.round(self.forward(X))