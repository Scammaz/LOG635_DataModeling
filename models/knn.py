from sklearn.neighbors import KNeighborsClassifier
import time

def KNN(k, x_train, y_train, x_test):    
    # Begin KNN
    # A- Phase d'apprentissage
    # we create an instance of Neighbours Classifier and train with the training dataset.
    weights = 'uniform'
    metric = 'euclidean'
    algorithm = 'brute'
    
    # Utiliser le meilleur k pour entrainer le modèle
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, metric=metric)
    
    time_start = time.time()
    knn = knn.fit(x_train, y_train)
    train_time = time.time() - time_start

    # B- Phase de prédiction ou de test
    # # Prédisez les étiquettes de classe pour les données fournies, dans ce cas, j'ai donné le x_test que j'ai déjà préparé au début.
    time_start = time.time()
    y_pred = knn.predict(x_test)
    predict_time = time.time() - time_start

    return {
        "y_pred": y_pred,
        "train_time": train_time,
        "predict_time": predict_time,
    }