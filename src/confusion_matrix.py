from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion(y_teste, assuntos, result):
    print('Relatório de Classificação\n', result[-1]['report'])
    y_pred = result[-1]['predictions']
    conf_matrix = confusion_matrix(y_teste, y_pred)
    _, test_counts = np.unique(y_teste, return_counts = True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize = (9, 8), dpi = 120)
    plt.title(result[-1]['model'].upper() + ' resultados')
    plt.xlabel('Valor Real')
    plt.ylabel('Previsão do Modelo')
    ticklabels = [f'r/{sub}' for sub in assuntos]
    sns.heatmap(data = conf_matrix_percent, xticklabels = ticklabels, yticklabels = ticklabels, annot = True, fmt = '.2f')
    plt.show()