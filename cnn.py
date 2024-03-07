import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

def CNN(x_train, y_train, x_test):
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    print(x_train.shape, x_test.shape)

    # A- Créer le modèle du CNN

    cnn = Sequential()
    cnn.add(Conv1D(64, 2, activation="relu", input_shape=(x_train.shape[1],1)))
    cnn.add(Dense(16, activation="relu"))
    cnn.add(MaxPooling1D())
    cnn.add(Flatten())
    cnn.add(Dense(8, activation = 'softmax'))
    cnn.compile(loss = 'sparse_categorical_crossentropy', 
                optimizer = "sgd",    #adam           
                metrics = ['accuracy'])
    cnn.summary()

    # B- Phase d'apprentissage
    history=cnn.fit(x_train, y_train.argmax(axis=1),
                    validation_split= 0.2, batch_size=16,epochs=2, verbose=1)#100 epochs
    
    print('--------------------------------------------------------------------\n'
      'Evaluate the trained CNN ...')

    # plotting the metrics

    fig = plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
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

    # C- Phase de prédiction
    return cnn.predict(x_test)
