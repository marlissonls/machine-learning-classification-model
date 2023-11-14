""" Text classification with supervised learning """

from src.load_data import carrega_dados
from src.split_data import split_data
from src.preprocessing_pipeline import preprocessing_pipeline
from src.create_models import criar_modelos
from src.train_evaluate import treina_avalia
from src.posts_distribution import plot_distribution
from src.confusion_matrix import plot_confusion


if __name__ == '__main__':

    assuntos = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']

    data, labels = carrega_dados(assuntos)

    X_treino, X_teste, y_treino, y_teste = split_data(data, labels)

    pipeline = preprocessing_pipeline()

    all_models = criar_modelos()

    resultados = treina_avalia(all_models, pipeline, X_treino, X_teste, y_treino, y_teste)

print('Concluído com sucesso!')


## Viewing results

# evaluation
plot_distribution(assuntos, labels)

# KNN results
plot_confusion(y_teste, assuntos, resultados[0])

# RandomForest results
plot_confusion(y_teste, assuntos, resultados[1])

# Logistic Regression results
plot_confusion(y_teste, assuntos, resultados[2])