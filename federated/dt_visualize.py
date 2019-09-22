import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,\
                            roc_curve, \
                            auc
import seaborn as sns
import matplotlib.pyplot as plt
import os

def dummy_labelize_swk(data, n_classes):

    """
        Make labels into dummy form

        (example)

        input : [0, 1, 2, 0, 0]
        output : [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0]
        ]
    
    Returns:
        [array] -- [dummy for of labels]
    """
    
    label = np.zeros((len(data), n_classes), dtype=int)
    
    for k, i in zip(data, label):
        i[int(k)] = 1
    
    return label

def save_roc_curve(y_test, y_pred_proba, roc_figure_save = True, n_classes = 2, save_path = './'):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # function edited by Sang Wook Kim, Korea University
    # test

    y_pred_proba_tr = np.amax(y_pred_proba, axis=1)

    # evaluation
    y_test_dummy = dummy_labelize_swk(y_test, n_classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummy[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10,10))

    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for mortality')
    plt.legend(loc="lower right")
    if roc_figure_save : plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    # plt.show()
    plt.close()

    # print('done')

    return

def save_confusion_matrix(y_true, y_pred_proba, save_path = './'):

    y_pred_proba_tr = np.argmax(y_pred_proba, axis=1)
    confusion = confusion_matrix(y_true, y_pred_proba_tr.astype(np.int))
    print (confusion)

    df_cm = pd.DataFrame(confusion, index = ['negative label', 'positive label'],
                    columns = ['negative prediction', 'positive prediction'])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    # plt.show()
    plt.close()

    return
