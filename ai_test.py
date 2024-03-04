from tabulate import tabulate
from sklearn.preprocessing import LabelBinarizer
from neural_network import NeuralNetwork
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

from test_RN import NeuralNetwork2

label_encoder = LabelBinarizer()

def print_dataset():
    with open("X.pkl", "rb") as db:
        features = pickle.load(db)
    with open("Y.pkl", "rb") as db:
        labels = pickle.load(db) 
              
    print(tabulate(zip(features["TEST"], labels["TEST"]), headers=["Features", "Label"]))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect = 'auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    plt.show()
    
    
def NN_test():
    with open("X.pkl", "rb") as db:
        features = pickle.load(db)
    with open("Y.pkl", "rb") as db:
        outputs = pickle.load(db)
        
    X_train, X_test = features['TRAIN'], features['TEST']
    outputs_train, outputs_test, classes = outputs['TRAIN'], outputs['TEST'], outputs['CLASSES']

    labels_train = [string_label for label in outputs_train for string_label in label.keys()]
    y_train = np.array([binary_label for label in outputs_train for binary_label in label.values()])

    labels_test = [string_label for label in outputs_test for string_label in label.keys()]
    y_test = np.array([binary_label for label in outputs_test for binary_label in label.values()])
    
    NN = NeuralNetwork2(
        nb_inputs=1600,
        nb_outputs=8,
        nb_hidden_layers=2,
        nb_nodes_per_layer=1024,
        learning_rate=0.01
    )
    
    print(f"X_train =\n{X_train}")
    print(f"y_train =\n{y_train}")
    
    epochs = 500
    NN.train(X_train,y_train, epochs)
    
    # Fonction de perte
    plt.plot(NN.loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss curve for training")
    plt.show()
    
    # Tester le modele
    y_pred = NN.predict(X_test)
    print(tabulate(zip(X_test, labels_test, [classes.get(tuple(o), "--") for o in y_pred]), headers=["Input", "Actual", "Predicted"]))
    
    print(y_pred.shape)
    print(y_test.shape)
    
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    plot_confusion_matrix(cm, classes= ['0', '1', '2', '3', '4', '5', '6', '7'], title='Confusion matrix, without normalization')
    
    
def main():
    # print_dataset()
    NN_test()


main()