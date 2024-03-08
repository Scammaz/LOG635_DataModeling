import numpy as np
from sklearn.calibration import label_binarize
from sklearn.svm import SVC
from sklearn import metrics

def SVM(x_train, y_train, x_test):
    
    # Convertir les données de sortie en 1D
    y_train_1d = np.argmax(y_train, axis=1)

    # A- Phase d'apprentissage
    svm = SVC(kernel='linear', degree=3, random_state=0)
    
    #Entrainer le modèle pour les données
    svm.fit(x_train, y_train_1d)

    # B- Phase de prédiction ou de test
    y_pred = svm.predict(x_test)

    # Convertir les données de sortie en multilabel
    y_pred_multilabel = label_binarize(y_pred, classes=list(range(8)))

    return y_pred_multilabel