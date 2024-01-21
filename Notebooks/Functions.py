import pandas                  as     pd
import numpy                   as     np
import seaborn                 as     sns
import matplotlib.pyplot       as     plt
from   sklearn.metrics         import confusion_matrix

def Confusion_Matrix_Func(y_test, y_pred, model_name):
    """   """
    fig, ax = plt.subplots(figsize=[10, 6])

# -------------------------------------------------------------------------------------------------
    cf_matrix     = confusion_matrix(y_test, y_pred)
    group_names   = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts  = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percent = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percent)]
    labels = np.asarray(labels).reshape(2, 2)

    # ----------------------------------------------------------------------------------------------
    sns.heatmap(cf_matrix,
                annot     = labels,
                ax        = ax,
                annot_kws = {'size': 13},
                cmap      = 'Blues',
                fmt       = ''
                )

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Truth')
    ax.xaxis.set_ticklabels(['Benign', 'Malignant'])
    ax.yaxis.set_ticklabels(['Benign', 'Malignant'])
    ax.set_title(f'Confusion Matrix of {model_name}\n')
    
    return cf_matrix, fig