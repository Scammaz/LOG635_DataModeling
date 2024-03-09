import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
import time

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
    time_start = time.time()
    history=cnn.fit(x_train, y_train.argmax(axis=1),
                    validation_split= 0.2, batch_size=16,epochs=30, verbose=1)
    train_time = time.time() - time_start

    print('--------------------------------------------------------------------\n'
      'Evaluate the trained CNN ...')


    # C- Phase de prédiction
    time_start = time.time()
    y_pred_cnn = cnn.predict(x_test)
    predict_time = time.time() - time_start
    
    return {
        "y_pred": y_pred_cnn,
        "train_time": train_time,
        "predict_time": predict_time,
    }
