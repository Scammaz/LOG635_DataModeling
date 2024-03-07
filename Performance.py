from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from dataset import B_LABELS

class Performance():
    def __init__(self, name, y_test, y_pred):

        self.name = name
        self.y_test = y_test
        self.y_pred = y_pred

        if self.name != 'CNN':
            self.metrics = precision_recall_fscore_support(self.y_test, self.y_pred, average='weighted')
            print('Precision: ', self.metrics[0], '\n')
            print('Recall: ', self.metrics[1], '\n')
            print('F1 score: ', self.metrics[2], '\n')

        # Matrix de confusion
        self.cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), normalize='true')
        plot_confusion_matrix(self.cm, classes= B_LABELS, normalize=True, title='Confusion matrix %s, with normalization' % self.name)
        
        # Courbe ROC
        ROC(self.y_test, self.y_pred)
    

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes) #, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_Save(cm, classes,
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

def ROC(y_test, y_pred):
    """
    Trace la courbe ROC pour des problèmes de classification multiclasse.
    """
    
    n_classes = 8  # Nombre de classes

    fpr = dict()        # Taux de faux positifs
    tpr = dict()        # Taux de vrais positifs
    roc_auc = dict()    # Aire sous la courbe ROC
    lw = 2              # Largeur de ligne pour le tracé

    # Calcul des taux de faux positifs, vrais positifs et aire sous la courbe pour chaque classe
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Définition des couleurs pour les différentes classes
    colors = cycle(['blue', 'red', 'green', 'pink', 'yellow', 'purple', 'brown', 'grey'])

    # Tracé des courbes ROC pour chaque classe
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    # Tracé de la ligne en pointillés diagonale représentant le hasard
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    # Réglages de l'axe x et y
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])

    # Libellés des axes et titre
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-class Data')

    # Légende positionnée en bas à droite
    plt.legend(loc="lower right")

    # Affiche la courbe ROC
    plt.show()