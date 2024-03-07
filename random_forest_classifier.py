from sklearn.ensemble import RandomForestClassifier

def Random_Forest(X_train, Y_train, X_test):
    # A- Phase d'apprentissage
    rf_classifier = RandomForestClassifier(random_state=0)

    # Entraîner le modèle sur les données d'entraînement
    rf_classifier.fit(X_train, Y_train)

    # B- Phase de prédiction ou de test
    y_pred_rf = rf_classifier.predict(X_test)

    return y_pred_rf