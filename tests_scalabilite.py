from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time

# Modeles
import models.knn as knn
import models.svm as svm
import models.random_forest_classifier as rf
import models.mlp_classifier as mlp
import models.cnn as cnn
from models.rn import NeuralNetwork

# Utility funcs
from DatasetPrepare.dataset import B_LABELS
import Performance as p

label_encoder = LabelBinarizer()
test_res_path = lambda h, o : f'NN_test_results/{h}_{o}/'

def main():
    test_modele_apprentissage()
    
def RN_test(X_train, Y_train, X_test):
    
    RN = NeuralNetwork(
        nb_inputs=1600,
        nb_outputs=8,
        nb_hidden_layers=1,
        nb_nodes_per_layer=1024,
        learning_rate=0.1,
        hidden_activation='sigmoid',
        out_activation='sigmoid',
        suppress_logging=True
    )
    
    epochs = 1000
    start_time = time.time()
    RN.train(X_train, Y_train, epochs)
    train_time = time.time() - start_time
    
    # RN 
    # Tester le modele
    start_time = time.time()
    y_pred_RN = RN.predict(X_test)
    predict_time = time.time() - start_time
    
    return {
        "y_pred": y_pred_RN,
        "train_time": train_time,
        "predict_time": predict_time,
    }

def test_modele_apprentissage():

    with open("DatasetPrepare/X.pickle", "rb") as db:
        features = pickle.load(db)
    with open("DatasetPrepare/Y.pickle", "rb") as db:
        outputs = pickle.load(db)
        
    X_train, X_test = features['TRAIN'], features['TEST']
    outputs_train, outputs_test, classes = outputs['TRAIN'], outputs['TEST'], outputs['CLASSES']

    Y_train = np.array([binary_label for label in outputs_train for binary_label in label.values()])
    Y_test = np.array([binary_label for label in outputs_test for binary_label in label.values()])

    # Diviser les données en données d'entrainement et données de test et validation
    # le parametre responsable à régler la scalabilité de données est le 'train_size'
    for ts in [0.8, 0.6, 0.4, 0.2]:
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, train_size=ts, random_state = 4)

        # Fonctionne
        # KNN
        metrics_KNN = knn.KNN(5, X_train, Y_train, X_test)
        p.Performance(test_size=ts, name="KNN", y_test=Y_test, metrics=metrics_KNN)

        # Fonctionne
        # Random Forest
        metrics_rf = rf.Random_Forest(X_train, Y_train, X_test)
        p.Performance(test_size=ts, name="Random Forest", y_test=Y_test, metrics=metrics_rf)

        # Fonctionne
        # MLP
        metrics_mlp = mlp.MlpClassifier(X_train, Y_train, X_test)
        p.Performance(test_size=ts, name="MLP", y_test=Y_test, metrics=metrics_mlp)

        # Fonctionne
        # SVM
        metrics_SVM = svm.SVM(X_train, Y_train, X_test)
        p.Performance(test_size=ts, name="SVM", y_test=Y_test, metrics=metrics_SVM)
        
        # Fonctionne
        # # CNN
        metrics_CNN = cnn.CNN(X_train, Y_train, X_test)
        p.Performance(test_size=ts, name="CNN", y_test=Y_test, metrics=metrics_CNN)

        # Fonctionne
        # RN
        metrics_RN = RN_test(X_train, Y_train, X_test)
        p.Performance(test_size=ts, name="RN", y_test=Y_test, metrics=metrics_RN)
    
main()

