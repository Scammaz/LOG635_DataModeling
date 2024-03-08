from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

class NeuralNetwork():
    def __init__(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_nodes_per_layer, learning_rate):
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.nb_hidden_layers = nb_hidden_layers
        self.nb_nodes_per_layer = nb_nodes_per_layer
        self.learning_rate = learning_rate
        self.loss = []
        self.A = np.zeros(nb_hidden_layers + 1, dtype=object)
        self.Z = np.zeros(nb_hidden_layers + 1, dtype=object)
        self.W = np.zeros(nb_hidden_layers + 1, dtype=object)
        self.B = np.zeros(nb_hidden_layers + 1, dtype=object)
        
        # Weight and Bias
        self.init_weights_bias()
        
    
    def init_weights_bias(self):
        # Weights
        #print("LE NOMBRE DE IMPUT EST DE ", self.nb_inputs)
        self.W[0] = np.random.randn(self.nb_inputs, self.nb_nodes_per_layer)
        self.W[self.nb_hidden_layers] = np.random.randn(self.nb_nodes_per_layer, self.nb_outputs)
            
        # Bias
        self.B[0] = np.ones((self.nb_nodes_per_layer))
        self.B[self.nb_hidden_layers] = np.ones(self.nb_outputs)
        
        # Fill weight and bias for hidden layers
        for i in range(1, self.nb_hidden_layers - 1):
            self.W[i] = np.random.randn(self.nb_nodes_per_layer, self.nb_nodes_per_layer)
            self.B[i] = np.ones((self.nb_nodes_per_layer))
         
        
    def sigmoid(self, z):
        positive = z >= 0
        negative = ~positive

        result = np.empty_like(z, dtype=float)
        result[positive] = self._pos_sigmoid(z[positive])
        result[negative] = self._neg_sigmoid(z[negative])

        return result
    
    
    def derived_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def relu(self, z):
        s = np.maximum(0,z)
        if(math.isnan(s)):
            print("NAN" )
        return s
    
    def derived_relu(self, z):
        zprime = z
        zprime[zprime<=0] = 0
        zprime[zprime>0] = 1
        return zprime
    
    def softmax(self, z):
        s = (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
        return s

    def derived_softmax(self, z):
        s = self.softmax(z)
        return s * (np.eye(s.size) - s.reshape(-1, 1))
    
    def entropy_loss(self, y, y_pred):
        eps = np.finfo(float).eps
        return -np.sum(y * np.log(y_pred + eps))
    
    def forward(self, X):
        # Forward propagation through our network
        # input to hidden
        self.A0 = X
        #print(f"A0 = {self.A0.shape} x {self.W[0].shape} + {self.B[0].shape}")

        self.Z[0] = np.dot(self.A0, self.W[0]) + self.B[0]
        self.A[0] = self.relu(self.Z[0])
        
        # print(f"Layer0 = {self.A0.shape} x {self.Z[0].shape}")
        
        #traverse hidden
        for i in range(1, self.nb_hidden_layers):
            self.Z[i] = np.dot(self.A[i-1], self.W[i]) + self.B[i]
            self.A[i] = self.relu(self.Z[i])
            #print(f"Layer{i} = {self.A[i-1].shape} x {self.Z[i].shape}")
        
        # hidden to output
        self.Z[self.nb_hidden_layers] = np.dot(self.A[self.nb_hidden_layers - 1], self.W[self.nb_hidden_layers]) + self.B[self.nb_hidden_layers]
        self.A[self.nb_hidden_layers] = self.softmax(self.Z[self.nb_hidden_layers])
        #print(f"Layer{self.nb_hidden_layers} = {self.A[self.nb_hidden_layers-1].shape} x {self.Z[self.nb_hidden_layers].shape}")
        
        #print(self.A[self.nb_hidden_layers])
        return self.A[self.nb_hidden_layers]
    
    def backward(self, X, y):
        m = X.shape[0]  # number of examples
        
        dA = np.zeros(self.nb_hidden_layers + 2, dtype=object)
        dZ = np.zeros(self.nb_hidden_layers + 2, dtype=object)
        dW = np.zeros(self.nb_hidden_layers + 2, dtype=object)
        db = np.zeros(self.nb_hidden_layers + 2, dtype=object)
        
        #Output layers
        # Error in output
        dZ[self.nb_hidden_layers] = self.A[self.nb_hidden_layers]-y
        # Delta for the weights w2
        dW[self.nb_hidden_layers] = (1./m) * np.dot(self.A[self.nb_hidden_layers - 1].T, dZ[self.nb_hidden_layers])
        # Delta for the bias b2
        db[self.nb_hidden_layers] = np.sum(dZ[self.nb_hidden_layers], axis=0)  # sum across columns
        # Weights/bias update
        self.W[self.nb_hidden_layers] -= self.learning_rate * dW[self.nb_hidden_layers]
        self.B[self.nb_hidden_layers] -= self.learning_rate * db[self.nb_hidden_layers]
        
        #Hidden layers
        for i in range(self.nb_hidden_layers-1, 0, -1):
            dA[i] = np.dot(dZ[i+1], self.W[i+1].T)

            dZ[i] = self.A[i] * self.derived_relu(self.Z[i])
            # Delta for the weights wn
            dW[i] = (1./m) * np.dot(self.A[i].T, dZ[i])
            # Delta for the bias b2
            db[i] = np.sum(dZ[i], axis=0)  # sum across columns
            # Update weights/bias
            self.W[i] -= self.learning_rate * dW[i]
            self.B[i] -= self.learning_rate * db[i]

        #Input layers
        dA[0] = np.dot(dZ[1], self.W[1].T)
        dZ[0] = dA[0] * self.derived_relu(self.Z[0])
        # Delta for the weights w1
        dW[0] = (1./m) * np.dot(X.T, dZ[1])
        # Delta for the bias b1
        db[0] = (1./m) * np.sum(dZ[1], axis=0)  # sum across columns

        # Wights and biases update
        self.W[0] -= self.learning_rate * dW[0]
        self.B[0] -= self.learning_rate * db[0]


    def train(self, X, y, nb_iterations):
        for i in range(nb_iterations):
            y_pred = self.forward(X)
            loss = self.entropy_loss(y, y_pred)
            self.loss.append(loss)
            self.backward(X, y)

            # why we start i at 1 ?
            if i == 0 or i == nb_iterations-1:
                print(f"Iteration: {i+1}")
                print(tabulate(zip(X, y, [np.round(y_pred) for y_pred in self.A[self.nb_hidden_layers]] ), headers=["Input", "Actual", "Predicted"]))
                print(f"Loss: {loss}")                
                print("\n")

    def predict(self, X):
        return np.round(self.forward(X))