from sklearn.model_selection import train_test_split
from tabulate import tabulate
import numpy as np

class NeuralNetwork():
    def __init__(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_nodes_per_layer, learning_rate,  validation=None, validation_arg=0.1, hidden_activation='sigmoid', out_activation='sigmoid', suppress_logging=False):
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
        
        self.tabulate_out = out_activation
        self.suppress_log = suppress_logging
        self.validation_type = validation
        self.validation_arg = validation_arg
        
        # Define activation functions
        self.init_activation_fn(hidden_activation, out_activation)
        
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
    
    def init_activation_fn(self, hidden, output):
        activation_fn = {
            'sigmoid': self.sigmoid,
            'ReLU': self.ReLU,
            'softmax': self.softmax
        }
        
        diff_act_fn = {
            "sigmoid": self.derived_sigmoid,
            "ReLU": self.derived_ReLU
        }
        
        self.hidden_activation = activation_fn.get(hidden, lambda: 'Invalid Function')
        self.hidden_diff = diff_act_fn.get(hidden, lambda: 'Invalid Function')
        self.out_activation = activation_fn.get(output, lambda: 'Invalid Function')
        self.out_diff = diff_act_fn.get(output, lambda: 'Invalid Function')
                                        

    # def _pos_sigmoid(self, z):
    #     return 1 / (1 + np.exp(-z))
    
    # def  _neg_sigmoid(self, z):
    #     exp = np.exp(z)
    #     return exp / (exp + 1)
         
    # def sigmoid(self, z):
    #     positive = z >= 0
    #     negative = ~positive

    #     result = np.empty_like(z, dtype=np.float)
    #     result[positive] = self._pos_sigmoid(z[positive])
    #     result[negative] = self._neg_sigmoid(z[negative])

    #     return result
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    
    def derived_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def ReLU(self, z):
        return np.maximum(0,z)
    
    def derived_ReLU(self, z):
        return np.where(z <= 0, 0, 1)
    
    def softmax(self, z):
        s = (np.exp(z - np.max(z)) / np.exp(z - np.max(z)).sum())
        return s
    
    def entropy_loss(self, y, y_pred):
        eps = np.finfo(float).eps
        return -np.sum(y * np.log(y_pred + eps))
    
    
    def forward(self, X):
        # Forward propagation through our network
        # input to hidden
        self.x = X
        self.Z[0] = np.dot(self.x, self.W[0]) + self.B[0]
        self.A[0] = self.hidden_activation(self.Z[0])
        
        #traverse hidden
        for i in range(1, self.nb_hidden_layers):
            self.Z[i] = np.dot(self.A[i-1], self.W[i]) + self.B[i]
            self.A[i] = self.hidden_activation(self.Z[i])
        
        # Output layer
        self.Z[self.nb_hidden_layers] = np.dot(self.A[self.nb_hidden_layers-1], self.W[self.nb_hidden_layers]) + self.B[self.nb_hidden_layers]
        self.A[self.nb_hidden_layers] = self.out_activation(self.Z[self.nb_hidden_layers])
        
        sum = np.sum(self.A[self.nb_hidden_layers], axis=0)
        
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
            dZ[i] = dA[i] * self.hidden_diff(self.Z[i])
            # Delta for the weights wn
            dW[i] = (1./m) * np.dot(self.A[i-1].T, dZ[i])
            # Delta for the bias b2
            db[i] = np.sum(dZ[i], axis=0)  # sum across columns
            # Update weights/bias
            self.W[i] -= self.learning_rate * dW[i]
            self.B[i] -= self.learning_rate * db[i]

        #Hidden to Input layer
        dA[0] = np.dot(dZ[1], self.W[1].T)
        dZ[0] = dA[0] * self.hidden_diff(self.Z[0])
        # Delta for the weights w1
        dW[0] = (1./m) * np.dot(X.T, dZ[0])
        # Delta for the bias b1
        db[0] = np.sum(dZ[0], axis=0)  # sum across columns

        # Wights and biases update
        self.W[0] -= self.learning_rate * dW[0]
        self.B[0] -= self.learning_rate * db[0]

    def train(self, X, y, epoch):
        if self.validation_type == 'hold_out':
            self.train_holdout_validation(X, y, epoch)
        elif self.validation_type == 'k_fold':
            self.train_kfold_validation(X, y, 5, epoch)
        elif self.validation_type == None:
            self.train_no_validation(X, y, epoch)
        else:
            print(f"No such validation method: {self.validation_type}\nTraining aborted!")
         
    def train_no_validation(self, X, y, epoch):
        for i in range(epoch):
            y_pred = self.forward(X)
            loss = self.entropy_loss(y, y_pred)
            self.loss.append(loss)
            self.backward(X, y)
            
            if not self.suppress_log and (i == 0 or i == epoch-1):
                self.print_training_step(X, y, y_pred, loss, i)    

    def train_holdout_validation(self, X, y, epoch):
        x_train, x_valid, y_train, y_valid = train_test_split( X, y, test_size=self.validation_arg, random_state=0, shuffle=True, stratify=y)
        self.loss_valid = []
        
        for i in range(epoch):
            y_valid_pred = self.forward(x_valid)
            y_pred = self.forward(x_train)
            loss = self.entropy_loss(y_train, y_pred)
            loss_valid = self.entropy_loss(y_valid, y_valid_pred)
            self.loss.append(loss)
            self.loss_valid.append(loss_valid)
            self.backward(x_train, y_train)
            
            self.print_training_step(x_train, y_train, y_pred, loss, i)       
            
    def train_kfold_validation(self, X, y, k, epoch):
        pass
    
    
    def prob_to_class(self, pred):
        return np.eye(pred.shape[1])[np.argmax(pred, axis=1)]
    
    def print_training_step(self, X, y, y_pred, loss, epoch):
        if not self.suppress_log and (epoch == 0 or epoch == epoch-1):
            print(f"Iteration: {epoch+1}")
            if self.tabulate_out == "sigmoid":
                print(tabulate(zip(X, y, [np.round(y_pred) for y_pred in self.A[self.nb_hidden_layers]]), headers=["Input", "Actual", "Predicted"]))
            elif self.tabulate_out == "softmax":
                print(tabulate(zip(X, y, self.prob_to_class(y_pred)), headers=["Input", "Actual", "Predicted"]))    
            print(f"Loss: {loss}")                
            print("\n")
        elif not self.suppress_log and epoch % 50 == 0:
            print(f"Iteration: {epoch+1}")
            print(f"\tLoss: {loss}")      
        
        

    def predict(self, X):
        if self.tabulate_out == "sigmoid":
            return np.round(self.forward(X))
        elif self.tabulate_out == "softmax":
            return self.prob_to_class(self.forward(X))
        else:
            print("No such output treatment")
            return None