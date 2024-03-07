from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def KNN(k, x_train, y_train, x_test):
    
    # Begin KNN

    # A- Phase d'apprentissage

    # we create an instance of Neighbours Classifier and train with the training dataset.
    weights = 'uniform'
    metric = 'euclidean'
    algorithm = 'brute'
    
    # Utiliser le meilleur k pour entrainer le modèle
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, metric=metric)
    knn = knn.fit(x_train, y_train)

    # B- Phase de prédiction ou de test
    # # Prédisez les étiquettes de classe pour les données fournies, dans ce cas, j'ai donné le x_test que j'ai déjà préparé au début.
    y_pred = knn.predict(x_test)

    return y_pred