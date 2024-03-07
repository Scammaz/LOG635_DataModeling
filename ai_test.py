from tabulate import tabulate
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
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
import knn 
import svm
import cnn
import Performance as p
from sklearn.metrics import precision_recall_fscore_support
from itertools import cycle

from dataset import B_LABELS
from neural_network import NeuralNetwork
from test_RN import NeuralNetwork2

label_encoder = LabelBinarizer()
test_res_path = lambda h, o : f'NN_test_results/{h}_{o}/'
    
def NN_test(nombre_de_couches_cachees=2, nombre_de_neurones_par_couche=512, taux_dapprentissage=0.2, nombre_diterations=500, hiddent_activation_func='sigmoid', output_activation_func='sigmoid'):
    
    with open("X.pkl", "rb") as db:
        features = pickle.load(db)
    with open("Y.pkl", "rb") as db:
        outputs = pickle.load(db)
        
    X_train, X_test = features['TRAIN'], features['TEST']
    outputs_train, outputs_test, classes = outputs['TRAIN'], outputs['TEST'], outputs['CLASSES']

    labels_train = [string_label for label in outputs_train for string_label in label.keys()]
    y_train = np.array([binary_label for label in outputs_train for binary_label in label.values()])

    labels_test = [string_label for label in outputs_test for string_label in label.keys()]
    y_test = np.array([binary_label for label in outputs_test for binary_label in label.values()])
    
    NN = NeuralNetwork(
        nb_inputs=1600,
        nb_outputs=8,
        nb_hidden_layers=nombre_de_couches_cachees, #2
        nb_nodes_per_layer=nombre_de_neurones_par_couche, #1024
        learning_rate=taux_dapprentissage, #0.01
        hidden_activation=hiddent_activation_func,
        out_activation=output_activation_func
    )
    
    epochs = nombre_diterations #300
    NN.train(X_train, y_train, epochs)
    
    name = f'{hiddent_activation_func}_{output_activation_func}_{nombre_de_couches_cachees}_{nombre_de_neurones_par_couche}_{taux_dapprentissage}_{nombre_diterations}'
    
    # Fonction de perte
    plt.plot(NN.loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss curve for training")
    plt.savefig(test_res_path(hiddent_activation_func, output_activation_func) + 'loss_' + name + '.png')
    # plt.show()
    plt.clf()
    
    # NN 
    # Tester le modele
    y_pred = NN.predict(X_test)
    print(tabulate(zip(X_test, labels_test, [classes.get(tuple(o), "--") for o in y_pred], y_pred), headers=["Input", "Actual", "Predicted", "Pred_Out"]))

    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    p.plot_confusion_matrix_Save(cm, classes= B_LABELS, normalize=True, title='Confusion matrix NN, with normalization', save=True, path=(test_res_path(hiddent_activation_func, output_activation_func) + 'conf-mat_' + name + '.png'))
    
    metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    return metrics

def main():
    
    hidden_fn = ["sigmoid", "ReLU"]
    out_fn = ["sigmoid", 'softmax']
    
    for i in range(2):
        test(hidden_fn[i], out_fn[i])


def test(hidden_fn, out_fn):
    path = f'{test_res_path(hidden_fn, out_fn)}resultsAiTraining.csv'
    
    list_couche_cachee = [1, 2, 3, 4, 5]
    list_neurones_par_couche = [512, 1024, 2048]
    list_taux_dapprentissage = [0.01, 0.02, 0.1]
    list_iterations = [500, 1000, 2000]
    percentageDone = len(list_couche_cachee) * len(list_neurones_par_couche) * len(list_taux_dapprentissage) * len(list_iterations)
    x = 0
    headerBool = True
    for couche_cachee in list_couche_cachee:
        for neurones_par_couche in list_neurones_par_couche:
            for taux_dapprentissage in list_taux_dapprentissage:
                for iterations in list_iterations:
                    print(f"couche_cachee = {couche_cachee}, neurones_par_couche = {neurones_par_couche}, taux_dapprentissage = {taux_dapprentissage}, iterations = {iterations}")
                    accuracy_score = NN_test(couche_cachee, neurones_par_couche, taux_dapprentissage, iterations, hidden_fn, out_fn)                   
                    precision = accuracy_score[0]
                    rappel = accuracy_score[1]  
                    f1 = accuracy_score[2]

                    result = [[couche_cachee, neurones_par_couche, taux_dapprentissage, iterations, precision, rappel, f1]]
                    df = pd.DataFrame(result, columns=['Couche Cachee', 'Neuronne pc', 'Taux Aprentissage', 'Iterations', 'Precision', 'Rappel', 'F1'])
                    df.to_csv(path, mode='a', header=headerBool, index=False)
                    headerBool = False
                    x += 1
                    print(f"Percentage done: {x/percentageDone*100} %")
    
def test_modele_apprentissage():

    with open("X.pkl", "rb") as db:
        features = pickle.load(db)
    with open("Y.pkl", "rb") as db:
        outputs = pickle.load(db)
        
    X_train, X_test = features['TRAIN'], features['TEST']
    outputs_train, outputs_test, classes = outputs['TRAIN'], outputs['TEST'], outputs['CLASSES']

    labels_train = [string_label for label in outputs_train for string_label in label.keys()]
    Y_train = np.array([binary_label for label in outputs_train for binary_label in label.values()])

    labels_test = [string_label for label in outputs_test for string_label in label.keys()]
    y_test = np.array([binary_label for label in outputs_test for binary_label in label.values()])

    # Diviser les données en données d'entrainement et données de test 
    # Dans ce cas, j'ai utilisé 20% du données pour le test et 80% pour le données d'entrainement
    # le parametre responsable à régler la scalabilité de données est le 'test_size'
    x_train,x_test,y_train,y_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 4)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # KNN

    # Nous utilisons un K de 40 pour trouver le meilleur K
    y_pred_KNN = knn.KNN(40, x_train, y_train, x_test, y_test)
    
    print(tabulate(zip(X_test, labels_test, [classes.get(tuple(o), "--") for o in y_pred_KNN], y_pred_KNN), headers=["Input", "Actual", "Predicted", "Pred_Out"]))

    p.Performance(name="KNN", y_test=y_test, y_pred=y_pred_KNN)

    # SVM
    y_pred_SVM = svm.SVM(x_train, y_train, x_test, y_test)
    print(tabulate(zip(X_test, labels_test, [classes.get(tuple(o), "--") for o in y_pred_SVM], y_pred_SVM), headers=["Input", "Actual", "Predicted", "Pred_Out"]))
    p.Performance(name="SVM", y_test=y_test, y_pred=y_pred_SVM)

    # CNN
    y_pred_CNN = cnn.CNN(x_train, y_train, x_test, y_test)
    print(tabulate(zip(X_test, labels_test, [classes.get(tuple(o), "--") for o in y_pred_CNN], y_pred_CNN), headers=["Input", "Actual", "Predicted", "Pred_Out"]))
    p.Performance(name="CNN", y_test=y_test, y_pred=y_pred_CNN)
    


main()