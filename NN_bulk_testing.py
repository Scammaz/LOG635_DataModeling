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
test_res_path = lambda h, o : f'NN_testV2_results/{h}_{o}/'
    
def testV2_init(X_data, Y_labels, nb_parallel_training, hidden_fn='sigmoid', out_fn='sigmoid'):
    path = f'{test_res_path(hidden_fn, out_fn)}resultsAiTraining.csv'
    
    nb_hidden_layers = [1, 2, 3]
    nb_node_per_layer = [512, 1024, 2048]
    learning_rates = [0.01, 0.02, 0.1]
    epochs = [500, 1000, 2000]
    
    procs = []
    csv_header = True
    start_time = time.time()
    
    test_len = len(nb_hidden_layers) * len(nb_node_per_layer) * len(learning_rates) * len(epochs)
    progress = Value('i', 0)
    semaphore = Semaphore(nb_parallel_training)
    
    for hd in nb_hidden_layers:
        for npl in nb_node_per_layer:
            for lr in learning_rates:
                for e in epochs:
                    semaphore.acquire()
                    p = Process(target=testV2, args=[X_data, Y_labels, hd, npl, lr, e, hidden_fn, out_fn, path, csv_header, test_len, progress, semaphore])
                    if csv_header:
                        csv_header = False
                    procs.append(p)
                    p.start()
    
    for p in procs:
        p.join()
        
    print(f"Tests finished in {time.time() - start_time} seconds")     

def proc_alive_limit(procs, limit):
    pass

def testV2(X, Y, hidden_layers, node_per_layer, learn_rate, epochs, hidden_fn, out_fn, res_path, header_bool, test_len, progress, semaphore):
    print(f"hidden_layers = {hidden_layers}, node_per_layer = {node_per_layer}, learn_rate = {learn_rate}, epochs = {epochs}")
    name = f'{hidden_fn}_{out_fn}_{hidden_layers}_{node_per_layer}_{learn_rate}_{epochs}'
    
    start_time = time.time()
    metrics = NN_test(X, Y, hidden_layers, node_per_layer, learn_rate, epochs, hidden_fn, out_fn)
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
    
    progress.value += 1
    print(f"Percentage done: {(progress.value/test_len)*100} %")
    semaphore.release()


def NN_test(features, outputs, validation=None, validation_arg=None, nombre_de_couches_cachees=2, nombre_de_neurones_par_couche=512, taux_dapprentissage=0.2, nombre_diterations=500, hiddent_activation_func='sigmoid', output_activation_func='sigmoid', suppress_log=True):
    X_train, X_test = features['TRAIN'], features['TEST']
    outputs_train, outputs_test, classes = outputs['TRAIN'], outputs['TEST'], outputs['CLASSES']

    labels_train = [string_label for label in outputs_train for string_label in label.keys()]
    y_train = np.array([binary_label for label in outputs_train for binary_label in label.values()])

    labels_test = [string_label for label in outputs_test for string_label in label.keys()]
    y_test = np.array([binary_label for label in outputs_test for binary_label in label.values()])
    
    NN = NeuralNetwork(
        nb_inputs=1600,
        nb_outputs=8,
        nb_hidden_layers=nombre_de_couches_cachees, #2
        nb_nodes_per_layer=nombre_de_neurones_par_couche, #1024
        learning_rate=taux_dapprentissage, #0.01
        hidden_activation=hiddent_activation_func,
        out_activation=output_activation_func,
        suppress_logging=suppress_log,
        validation=validation,
        validation_arg=validation_arg
    )

    # Train
    NN.train(X_train, y_train, nombre_diterations)
    
    # Tester
    y_pred = NN.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
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
    
    #testV2_init(features, outputs, nb_parallel_training=2, hidden_fn='sigmoid', out_fn='softmax')
    
    results = NN_test(features, outputs, 'hold_out', 0.1, 1, 2048, 0.1, 1000, 'sigmoid', 'sigmoid', False)
    
    print(f"Loss: {results['loss'][-1]}\nPrecision: {results['precision']}\nRecall: {results['recall']}\nF1 score: {results['fscore']}")
    
    plot_loss_curve(results['loss'], results['loss_valid'], save=False)
    plot_confusion_matrix(results['conf_matrix'], classes= B_LABELS, normalize=True, title='Confusion matrix NN, with normalization')
    
    
        
    