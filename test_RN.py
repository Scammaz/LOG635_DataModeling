from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import pickle

class NeuralNetwork2():
    def __init__(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_nodes_per_layer, learning_rate):
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.nb_hidden_layers = nb_hidden_layers
        self.nb_nodes_per_layer = nb_nodes_per_layer
        self.learning_rate = learning_rate
        self.loss = []
        self.A = [None] * (nb_hidden_layers + 1)
        self.Z = [None] * (nb_hidden_layers + 1)
        self.W = [None] * (nb_hidden_layers + 1)
        self.B = [None] * (nb_hidden_layers + 1)
        
        # Weight and Bias
        self.init_weights_bias()
        
    
    def init_weights_bias(self):
        # Weights
        self.W[0] = np.random.randn(self.nb_inputs, self.nb_nodes_per_layer)
        self.W[-1] = np.random.randn(self.nb_nodes_per_layer, self.nb_outputs)
            
        # Bias
        self.B[0] = np.ones((self.nb_nodes_per_layer))
        self.B[-1] = np.ones(self.nb_outputs)
        
        # Fill weight and bias for hidden layers
        for i in range(1, self.nb_hidden_layers):
            self.W[i] = np.random.randn(self.nb_nodes_per_layer, self.nb_nodes_per_layer)
            self.B[i] = np.ones((self.nb_nodes_per_layer))
         
        
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s
    
    def derived_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def relu(self, z):
        s = np.maximum(0,z)
        return s
    
    def derived_relu(self, z):
        zprime = z
        zprime[zprime<=0] = 0
        zprime[zprime>0] = 1
        return zprime
    
    def softmax(self, z):
        s = (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
        return s
    
    def softmaxStabilized(self, z):
        # Stabilize input for numerical stability
        e_x = np.exp(z - np.max(z))
        s = e_x / e_x.sum()
        
        # x = z - np.max(z, axis=-1, keepdims=True)
        # s = (np.exp(x.T) / np.sum(np.exp(x)))
        return s
    
    def entropy_loss(self, y, y_pred):
        eps = np.finfo(float).eps
        return -np.sum(y * np.log(y_pred + eps))
    
    def forward(self, X):
        # Forward propagation through our network
        # input to hidden
        self.x = X
        self.Z[0] = np.dot(self.x, self.W[0]) + self.B[0]
        self.A[0] = self.sigmoid(self.Z[0])
        
        #traverse hidden
        for i in range(1, self.nb_hidden_layers):
            self.Z[i] = np.dot(self.A[i-1], self.W[i]) + self.B[i]
            self.A[i] = self.sigmoid(self.Z[i])
        
        # Output layer
        self.Z[self.nb_hidden_layers] = np.dot(self.A[self.nb_hidden_layers-1], self.W[self.nb_hidden_layers]) + self.B[self.nb_hidden_layers]
        self.A[self.nb_hidden_layers] = self.sigmoid(self.Z[self.nb_hidden_layers])
        
        return self.A[self.nb_hidden_layers]
    
    def backward(self, X, y):
        m = X.shape[0]  # number of examples
        
        dA = [None] * (self.nb_hidden_layers + 1)
        dZ = [None] * (self.nb_hidden_layers + 1)
        dW = [None] * (self.nb_hidden_layers + 1)
        db = [None] * (self.nb_hidden_layers + 1)
        
        #Output to hidden layer
        # Error in output
        dZ[-1] = self.A[-1] - y
        # Delta for the weights w2
        dW[-1] = (1./m) * np.dot(self.A[-2].T, dZ[-1])
        # Delta for the bias b2
        db[-1] = np.sum(dZ[-1], axis=0)  # sum across columns
        # Weights/bias update
        self.W[-1] -= self.learning_rate * dW[-1]
        self.B[-1] -= self.learning_rate * db[-1]
        
        #Hidden layers
        for i in range(self.nb_hidden_layers- 1, 0, -1):
            dA[i] = np.dot(dZ[i+1], self.W[i+1].T)
            dZ[i] = dA[i] * self.derived_sigmoid(self.Z[i])
            # Delta for the weights wn
            dW[i] = (1./m) * np.dot(self.A[i-1].T, dZ[i])
            # Delta for the bias b2
            db[i] = np.sum(dZ[i], axis=0)  # sum across columns
            # Update weights/bias
            self.W[i] -= self.learning_rate * dW[i]
            self.B[i] -= self.learning_rate * db[i]

        #Hidden to Input layer
        dA[0] = np.dot(dZ[1], self.W[1].T)
        dZ[0] = dA[0] * self.derived_sigmoid(self.Z[0])
        # Delta for the weights w1
        dW[0] = (1./m) * np.dot(X.T, dZ[0])
        # Delta for the bias b1
        db[0] = np.sum(dZ[0], axis=0)  # sum across columns

        # Wights and biases update
        self.W[0] -= self.learning_rate * dW[0]
        self.B[0] -= self.learning_rate * db[0]


    def train(self, X, y, nb_iterations):
        for i in range(nb_iterations):
            y_pred = self.forward(X)
            loss = self.entropy_loss(y, y_pred)
            self.loss.append(loss)
            self.backward(X, y)

            if i == 0 or i == nb_iterations-1:
                print(f"Iteration: {i+1}")
                #print(tabulate(zip(X, y, [np.round(y_pred).tolist() for y_pred in self.A[self.nb_hidden_layers]] ), headers=["Input", "Actual", "Predicted"]))
                print(f"Loss: {loss}")                
                print("\n")

    def predict(self, X):
        var = np.round(self.forward(X))
        print("pred = " ,var)
        return var