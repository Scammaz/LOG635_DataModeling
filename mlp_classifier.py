from sklearn.neural_network import MLPClassifier

def MlpClassifier(X_train, Y_train, X_test):
    
        # A- Phase d'apprentissage
    mlp_classifier = MLPClassifier(random_state=0)

    # Entraîner le modèle sur les données d'entraînement
    mlp_classifier.fit(X_train, Y_train)

    # B- Phase de prédiction ou de test
    y_pred_mlp = mlp_classifier.predict(X_test)

    return y_pred_mlp