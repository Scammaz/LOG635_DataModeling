from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import label_binarize

def nn(x_train, y_train, x_test, y_test):
    
    # A- Créer le modèle du réseau de neurones

    # Initialiser un modèle séquentiel
    NN = Sequential()

    # Ajouter une couche Dense avec 1600 neurones, spécifier input_dim=1600 pour indiquer la forme de l'entrée,
    # et utiliser la fonction d'activation 'relu'.
    NN.add(Dense(1600, input_dim=1600, activation='relu'))

    # Ajouter une autre couche Dense avec 500 neurones et fonction d'activation 'relu'.
    NN.add(Dense(500, activation='relu'))

    # Ajouter une autre couche Dense avec 300 neurones et fonction d'activation 'relu'.
    NN.add(Dense(300, activation='relu'))

    # Ajouter une couche Dropout pour prévenir le surajustement en désactivant aléatoirement 20% des neurones.
    NN.add(Dropout(0.2))

    # Ajouter la dernière couche Dense avec 8 neurones (correspondant aux 8 classes) et fonction d'activation 'softmax'.
    NN.add(Dense(8, activation='softmax'))

    # Compiler le modèle en spécifiant la fonction de perte, l'optimiseur et les métriques à surveiller.
    NN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Afficher un résumé du modèle
    NN.summary()

    # B- Phase d'apprentissage

    # Entraîner le modèle en utilisant la méthode fit
    history = NN.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=20, epochs=10, verbose=1)

    print('--------------------------------------------------------------------\n'
      'Evaluate the trained CNN ...')

    # plotting the metrics

    fig = plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss NN ', )
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='best')

    plt.show()

    # plotting the metrics

    fig = plt.figure()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='best')

    plt.show()

    # C- Phase de prédiction ou de test
    y_pred_NN = NN.predict(x_test)

    # Convertir les données de sortie en multilabel
    # y_pred_NN_classes = label_binarize(y_pred_NN, classes=list(range(8)))

    return y_pred_NN.argmax(axis=1)