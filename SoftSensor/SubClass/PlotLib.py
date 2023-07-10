import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix


def Plot_ConfMat_EachCfy(y_true_all, y_pred_all, label, title):
    cm = confusion_matrix(y_true_all, y_pred_all)
    plot_confusion_matrix(cm,
                normalize    = True,
                digits       = 2,
                target_names = label,
                title        = title)    
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 6.5)
    fig.tight_layout()


def Plot_AccBox_AllCfy(dic_acc, acc_range=(0.7,1), title=""):
    plot_boxscatter(dic_acc)
    plt.title(title + "\n\n")
    plt.ylabel("Accuracy")
    plt.ylim(acc_range)
    plt.grid(axis="y", color='grey', linestyle='--', linewidth=1)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 6)
    fig.tight_layout()


def Plot_Bars_AllCLF_AllLBL(dic_y_true, dic_y_pred, LabelNames, f_metrics, YLBL_Name, title=""):

    metrics = {}
    for key in dic_y_true:
        metrics[key] = f_metrics(dic_y_true[key], dic_y_pred[key], average=None)

    plot_groupbars(metrics, LabelNames, ylabel=YLBL_Name)

    plt.grid(axis="y", color='grey', linestyle='--', linewidth=1)
    
    plt.title(title + "\n\n")

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 3)
    fig.tight_layout()
    #plt.show()







'''
----------------------------------------------------------------------------------------------------
'''    

def plot_groupbars(dict_data, member_labels, xlabel="", ylabel="", title="", width=0.8, digits=2):

    n_member = len(member_labels)
    n_group = len(dict_data.keys()) # # of Classifiers
    x = np.arange(n_member)  # the member_labels locations (Gesture 1~12)

    rects = []
    width_inc = width / n_group
    #hatch = ['/', '.', '\\', 'o', '-', '+', 'x', 'o', 'O',  '*', '|']
    cmap=plt.cm.get_cmap('viridis', n_group)

    fig, ax = plt.subplots()
    for i, key in enumerate(dict_data.keys()):
        rects.append(ax.bar(x - width/2 + width_inc*i, dict_data[key], width_inc, \
                    label=key, color=cmap(i/n_group)))
    
    # Add some text for labels, title and custom x-axis tick labels, etc.    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(member_labels)
    ax.legend(ncol=1, bbox_to_anchor=(1.03, 1.0), loc='upper right')


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate("{:0.{digits}f}".format(height, digits=digits),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # for re in rects:
    #     autolabel(re)

    fig.tight_layout()

    #plt.show()
    return fig


def plot_boxscatter(dict_data, 
                palette=['r', 'g', 'b', 'y', 'c', 'm']):
    
    fig, ax = plt.subplots()    

    cfyName = list(dict_data.keys())
    xs, data = [], []
    for i, key in enumerate(dict_data.keys()):

        data.append(dict_data[key])
        xs.append(np.random.normal(i+1, 0.05, len(dict_data[key])))

    ax.boxplot(data, labels=cfyName, showfliers=False)    
    
    for x, val, c in zip(xs, data, palette):
        ax.scatter(x, val, alpha=0.4, color=c)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          noShow=True,
                          digits=2):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.rcParams.update({'figure.max_open_warning': 0})
    fig = plt.figure(figsize=(8, 6.5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.{digits}f}".format(cm[i, j], digits=digits),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))    
    
    if not noShow:
        plt.show()

    return fig