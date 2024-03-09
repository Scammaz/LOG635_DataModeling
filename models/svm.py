import numpy as np
from sklearn.calibration import label_binarize
from sklearn.svm import SVC
import time

def SVM(x_train, y_train, x_test):
    
    # Convertir les données de sortie en 1D
    y_train_1d = np.argmax(y_train, axis=1)

    # A- Phase d'apprentissage
    svm = SVC(kernel='linear', degree=3, random_state=0)
    
    #Entrainer le modèle pour les données
    time_start = time.time()
    svm.fit(x_train, y_train_1d)
    train_time = time.time() - time_start

    # B- Phase de prédiction ou de test
    time_start = time.time()
    y_pred = svm.predict(x_test)
    predict_time = time.time() - time_start

    # Convertir les données de sortie en multilabel
    y_pred_multilabel = label_binarize(y_pred, classes=list(range(8)))

    return {
        "y_pred": y_pred_multilabel,
        "train_time": train_time,
        "predict_time": predict_time,
    }