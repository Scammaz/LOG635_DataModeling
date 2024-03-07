from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from matplotlib import pyplot as plt

def nn(x_train, y_train, x_test, y_test):
    
    # A- Créer le modèle du réseau de neurone
    NN=Sequential()
    NN.add(Dense(1600,input_dim=4,activation='relu'))
    NN.add(Dense(500,activation='relu'))
    NN.add(Dense(300,activation='relu'))
    NN.add(Dropout(0.2))
    NN.add(Dense(8,activation='softmax'))
    NN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    NN.summary()

    # B- Phase d'apprentissage
    history= NN.fit(x_train,y_train,validation_data=(x_test,y_test), validation_split= 0.2, batch_size=20,epochs=10,verbose=1)

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

    # C- Phase de prédiction ou de test
    y_pred=NN.predict(x_test)

    return y_pred