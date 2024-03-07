import copy
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import itertools
import time
from multiprocessing import Process, Value, Semaphore

from dataset import B_LABELS
from neural_network import NeuralNetwork
from test_RN import NeuralNetwork2

label_encoder = LabelBinarizer()
test_res_path = lambda h, o : f'test_results/{h}_{o}/'
test_len = 0
test_progress = 0

    
def testV2_init(X_data, Y_labels, hidden_fn='sigmoid', out_fn='sigmoid'):
    path = f'{test_res_path(hidden_fn, out_fn)}resultsAiTraining.csv'
    
    nb_hidden_layers = [1, 2, 3, 4]
    nb_node_per_layer = [256, 512, 1024, 2048]
    learning_rates = [0.01, 0.02, 0.05, 0.1]
    epochs = [500, 1000, 2500, 5000]
    
    csv_header = True
    start_time = time.time()
    
    global test_len
    test_len = len(nb_hidden_layers) * len(nb_node_per_layer) * len(learning_rates) * len(epochs)
    
    for hd in nb_hidden_layers:
        for npl in nb_node_per_layer:
            for lr in learning_rates:
                for e in epochs:
                    testV2(copy.deepcopy(X_data), copy.deepcopy(Y_labels),hd, npl, lr, e, hidden_fn, out_fn, path, csv_header)
                    if csv_header:
                        csv_header = False
        
    print(f"Tests finished in {time.time() - start_time} seconds")     

def testV2(X, Y, hidden_layers, node_per_layer, learn_rate, epochs, hidden_fn, out_fn, res_path, header_bool):
    print(f"hidden_layers = {hidden_layers}, node_per_layer = {node_per_layer}, learn_rate = {learn_rate}, epochs = {epochs}")
    name = f'{hidden_fn}_{out_fn}_{hidden_layers}_{node_per_layer}_{learn_rate}_{epochs}'
    
    start_time = time.time()
    metrics = NN_test(
        features=X, 
        outputs=Y, 
        hidden_layers=hidden_layers, 
        nodes_per_layer=node_per_layer, 
        learning_rate=learn_rate, 
        epochs=epochs, 
        hidden_activation_func=hidden_fn, 
        output_activation_func=out_fn
    )
    train_time = time.time() - start_time

    loss = metrics.get('loss')     
    cm = metrics.get('conf_matrix')            
    precision = metrics.get('precision')
    recall = metrics.get('recall')
    f1 = metrics.get('fscore')

    plot_loss_curve(loss, save=True, name=name, path=(test_res_path(hidden_fn, out_fn) + 'loss-curve_' + name + '.png'))
    plot_confusion_matrix(cm, classes= B_LABELS, normalize=True, title='Confusion matrix NN, with normalization', save=True, path=(test_res_path(hidden_fn, out_fn) + 'conf-mat_' + name + '.png'))
    result = [[hidden_layers, node_per_layer, learn_rate, epochs, loss[-1], precision, recall, f1, train_time]]
    df = pd.DataFrame(result, columns=['Hidden Layers', 'Nodes per Layer', 'Learning Rate', 'Epochs', 'Loss', 'Precision', 'Recall', 'F1', 'Training_time (sec)'])
    df.to_csv(res_path, mode='a', header=header_bool, index=False)
    
    global test_progress
    test_progress += 1
    print(f"Percentage done: {(test_progress/test_len)*100} %")


def NN_test(features, outputs, validation=None, validation_arg=0.1, hidden_layers=2, nodes_per_layer=512, learning_rate=0.2, epochs=500, hidden_activation_func='sigmoid', output_activation_func='sigmoid', suppress_log=True):
    X_train, X_test = features['TRAIN'], features['TEST']
    outputs_train, outputs_test = outputs['TRAIN'], outputs['TEST']

    y_train = np.array([binary_label for label in outputs_train for binary_label in label.values()])
    y_test = np.array([binary_label for label in outputs_test for binary_label in label.values()])
    
    NN = NeuralNetwork(
        nb_inputs=1600,
        nb_outputs=8,
        nb_hidden_layers=hidden_layers,
        nb_nodes_per_layer=nodes_per_layer,
        learning_rate=learning_rate,
        hidden_activation=hidden_activation_func,
        out_activation=output_activation_func,
        suppress_logging=suppress_log,
        validation=validation,
        validation_arg=validation_arg
    )

    # Train
    NN.train(X_train, y_train, epochs)
    
    # Test
    y_pred = NN.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    # Metrics
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    results = {
        'conf_matrix': cm,
        'loss': NN.loss,
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    }
    
    if validation == 'hold_out':
        results['loss_valid'] = NN.loss_valid
        
    
    return results
        
def plot_loss_curve(loss, loss_valid=None, save=False, name='', path=''):
    # Fonction de perte
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss training curve for [{name}]")
    
    if loss_valid != None:
        plt.plot(loss, color='blue', label='training')
        plt.plot(loss_valid, color='red', label='validation')
        plt.legend()
    else:
       plt.plot(loss) 
    
    if save:
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False,
                          path=''):
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
    
    if save:
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()
        
                
if __name__ == "__main__": 
    with open("X.pkl", "rb") as db:
        features = pickle.load(db)
    with open("Y.pkl", "rb") as db:
        outputs = pickle.load(db)
    
    testV2_init(features, outputs, hidden_fn='sigmoid', out_fn='sigmoid')
    
    # results = NN_test(
    #     features=features, 
    #     outputs=outputs, 
    #     hidden_layers=1, 
    #     nodes_per_layer=2048, 
    #     learning_rate=0.5, 
    #     epochs=1000, 
    #     hidden_activation_func='sigmoid', 
    #     output_activation_func='softmax', 
    #     suppress_log=False,
    #     validation=None,
    #     validation_arg=None
    #     )
    
    # loss_validation = results.get('loss_validation', None)
    
    # print(f"Loss: {results['loss'][-1]}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 score: {results['fscore']}")
    
    # plot_loss_curve(results['loss'], loss_validation, save=False)
    # plot_confusion_matrix(results['conf_matrix'], classes= B_LABELS, normalize=True, title='Confusion matrix NN, with normalization')
    
    
        
    