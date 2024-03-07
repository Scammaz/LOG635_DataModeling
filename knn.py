from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def KNN(k, x_train, y_train, x_test):
    
    # Begin KNN

    # A- Phase d'apprentissage

    # we create an instance of Neighbours Classifier and train with the training dataset.
    weights = 'uniform'
    metric = 'euclidean'
    algorithm = 'brute'

    # Dans ce cas, j'ai choisi aléatoirement le parametre 'n_neighbors'. Vous pouvez faire un 
    # boucle 'for' sur plusieurs valeurs de 'n_neighbors' et choisir celle qui donne la meilleure 
    # valeur de précision comme suivant:
    # k_range = range(1, k)
    # scores_list = []
    # for ki in k_range:

    #     if ki == 0:
    #         continue
    #     knn = KNeighborsClassifier(n_neighbors=ki, weights=weights, algorithm=algorithm, metric=metric)
    #     knn = knn.fit(x_train, y_train)
    #     y_pred=knn.predict(x_test)
    #     scores_list.append(metrics.accuracy_score(y_test,y_pred))
    #     print("Accuracy is ", metrics.accuracy_score(y_test,y_pred), " for k-value:", ki)

    
    # Utiliser le meilleur k pour entrainer le modèle
    #best_k = k_range[scores_list.index(max(scores_list))]
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, metric=metric)
    knn = knn.fit(x_train, y_train)

    # B- Phase de prédiction ou de test
        
    # # Prédisez les étiquettes de classe pour les données fournies, dans ce cas, j'ai donné le x_test que j'ai déjà préparé au début.
    y_pred = knn.predict(x_test)

    return y_pred