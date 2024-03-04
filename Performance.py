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

def KNN(test_size, k):

    iris = load_iris()

    # le vecteur X contient tous les primitive (en total il y a 4 primitives) des exemples de données iris
    # sepal length(cm) | sepal width(cm) | petal length(cm) | petal width
    X = iris.data
    Y = iris.target

    #Change the label to one hot vector
    Y = label_binarize(Y, classes=[0,1,2,3,4,5,6,7])
    print(Y.shape)

    # Diviser les données en données d'entrainement et données de test 
    # Dans ce cas, j'ai utilisé 20% du données pour le test et 80% pour le données d'entrainement
    # le parametre responsable à régler la scalabilité de données est le 'test_size'
    x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = test_size, random_state = 4)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    
    # we create an instance of Neighbours Classifier and train with the training dataset.
    weights = 'uniform'
    metric = 'euclidean'
    algorithm = 'brute'

    knn = KNeighborsClassifier(k, weights=weights, algorithm=algorithm, metric=metric)
    knn = knn.fit(x_train, y_train)

    # Show all parameters of the model Normal model
    # You can change all these parameters
    # See the documentation
    # model

    # Use the model to predict the class of samples
    # Notice that we are testing the train dataset
    y_train_pred = knn.predict(x_train)
        
    # You can also predict the probability of each class
    # train dataset
    y_train_pred_prob = knn.predict_proba(x_train)

    acc_iris_data = accuracy_score(y_train, y_train_pred)
    print("Correct classification rate for the training dataset = "+str(acc_iris_data*100)+"%")

    target_names = ['0', '1', '2', '3', '4', '5', '6', '7'] # name of classes

    print(classification_report(y_train, y_train_pred, target_names=target_names))
    # This works, but we have labels with no predicted samples