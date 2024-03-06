from tabulate import tabulate
from sklearn.preprocessing import LabelBinarizer
from neural_network import NeuralNetwork
from sklearn.metrics import f1_score, precision_score, recall_score, auc, \
                            roc_curve, confusion_matrix, classification_report, accuracy_score, \
                            cohen_kappa_score, roc_auc_score, roc_curve
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
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

def KNN(k, x_train, y_train, x_test, y_test):
    
    # Begin KNN

    # A- Phase d'apprentissage

    # we create an instance of Neighbours Classifier and train with the training dataset.
    weights = 'uniform'
    metric = 'euclidean'
    algorithm = 'brute'

    # Dans ce cas, j'ai choisi aléatoirement le parametre 'n_neighbors'. Vous pouvez faire un 
    # boucle 'for' sur plusieurs valeurs de 'n_neighbors' et choisir celle qui donne la meilleure 
    # valeur de précision comme suivant:
    k_range = range(1, k)
    scores_list = []
    for ki in k_range:
        knn = KNeighborsClassifier(n_neighbors=ki, weights=weights, algorithm=algorithm, metric=metric)
        knn = knn.fit(x_train, y_train)
        y_pred=knn.predict(x_test)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        print("Accuracy is ", metrics.accuracy_score(y_test,y_pred), " for k-value:", ki)

    # B- Phase de prédiction ou de test
        
    # # Prédisez les étiquettes de classe pour les données fournies, dans ce cas, j'ai donné le x_test que j'ai déjà préparé au début.
    return knn.predict(x_train)

def SVM(x_train, y_train, x_test):
    
    # A- Phase d'apprentissage

    svm = SVC(kernel = 'linear', random_state = 0)
    
    #Entrainer le modèle pour les données
    svm.fit(x_train, y_train.argmax(axis=1))

    # B- Phase de prédiction ou de test
    return svm.predict(x_test)

def CNN(x_train, y_train, x_test):
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    print(x_train.shape, x_test.shape)

    # A- Créer le modèle du CNN

    cnn = Sequential()
    cnn.add(Conv1D(64, 2, activation="relu", input_shape=(x_train.shape[1],1)))
    cnn.add(Dense(16, activation="relu"))
    cnn.add(MaxPooling1D())
    cnn.add(Flatten())
    cnn.add(Dense(3, activation = 'softmax'))
    cnn.compile(loss = 'sparse_categorical_crossentropy', 
        optimizer = "sgd",    #adam           
                metrics = ['accuracy'])
    cnn.summary()

    # B- Phase d'apprentissage
    history=cnn.fit(x_train, y_train.argmax(axis=1),
                    validation_split= 0.2, batch_size=16,epochs=100, verbose=1)
    
    print('--------------------------------------------------------------------\n'
      'Evaluate the trained CNN ...')

    # plotting the metrics

    fig = plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='best')

    plt.show()

    # plotting the metrics

    fig = plt.figure()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='best')

    plt.show()

    # C- Phase de prédiction
    return cnn.predict(x_test)