from sklearn.svm import SVC

def SVM(x_train, y_train, x_test):
    
    # A- Phase d'apprentissage

    svm = SVC(kernel = 'linear', random_state = 0)
    
    #Entrainer le modèle pour les données
    svm.fit(x_train, y_train.argmax(axis=1))

    # B- Phase de prédiction ou de test
    return svm.predict(x_test)