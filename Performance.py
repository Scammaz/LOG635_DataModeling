from tabulate import tabulate
from sklearn.preprocessing import LabelBinarizer
from neural_network import NeuralNetwork
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import pandas as pd  # Import pandas library

def KNN():

    iris = load_iris()

    # le vecteur X contient tous les primitive (en total il y a 4 primitives) des exemples de données iris
    # sepal length(cm) | sepal width(cm) | petal length(cm) | petal width
    X = iris.data
    Y = iris.target

    #Change the label to one hot vector
    Y = label_binarize(Y, classes=[0,1,2])
    print(Y.shape)

    # Diviser les données en données d'entrainement et données de test 
    # Dans ce cas, j'ai utilisé 20% du données pour le test et 80% pour le données d'entrainement
    # le parametre responsable à régler la scalabilité de données est le 'test_size'
    x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2,random_state=4)

    # we create an instance of Neighbours Classifier and train with the training dataset.
    n_neighbors = 1
    weights = 'uniform'
    metric = 'euclidean'
    algorithm = 'brute'

    knn = KNeighborsClassifier(n_neighbors, weights=weights, algorithm=algorithm, metric=metric )
    knn = knn.fit(x_train, y_train)

    # Show all parameters of the model Normal model
    # You can change all these parameters
    # See the documentation
    # model

    # Predict the response for test dataset
    y_pred = knn.predict(x_test)
    
