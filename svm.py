from sklearn.svm import SVC
from sklearn import metrics

def SVM(x_train, y_train, x_test):
    
    # A- Phase d'apprentissage
    svm = SVC(kernel='linear', degree=3, random_state=0)
    
    #Entrainer le modèle pour les données
    svm.fit(x_train, y_train)

    # B- Phase de prédiction ou de test
    y_pred = svm.predict(x_test)

    return y_pred