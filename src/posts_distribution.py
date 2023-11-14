import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(assuntos, labels):
    _, counts = np.unique(labels, return_counts = True)
    sns.set_theme(style = 'whitegrid')
    plt.figure(figsize = (15, 6), dpi = 120)
    plt.title(('Número de Posts Por Assunto'))
    sns.barplot(x = assuntos, y = counts)
    plt.legend([' '.join([f.title(), f'- {c} posts']) for f, c in zip(assuntos, counts)])
    plt.show()