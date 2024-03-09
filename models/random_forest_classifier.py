from sklearn.ensemble import RandomForestClassifier
import time

def Random_Forest(X_train, Y_train, X_test):
    # A- Phase d'apprentissage
    rf_classifier = RandomForestClassifier(random_state=0)

    # Entraîner le modèle sur les données d'entraînement
    time_start = time.time()
    rf_classifier.fit(X_train, Y_train)
    train_time = time.time() - time_start
    
    # B- Phase de prédiction ou de test
    time_start = time.time()
    y_pred_rf = rf_classifier.predict(X_test)
    predict_time = time.time() - time_start

    return {
        "y_pred": y_pred_rf,
        "train_time": train_time,
        "predict_time": predict_time,
    }