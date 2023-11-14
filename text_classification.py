""" Text classification with supervised learning """

from os import getenv
import dotenv
import re
import praw
import config
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


dotenv.load_dotenv()

## Load data
assuntos = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy', ]

def carrega_dados():
    
    api_reddit = praw.Reddit(client_id = getenv('client_id'),
                             client_secret = getenv('client_secret'),
                             password = getenv('password'),
                             user_agent = getenv('user_agent'),
                             username = getenv('user'))

    char_count = lambda post: len(re.sub('\W|\d', '', post.selftext))

    mask = lambda post: char_count(post) >= 100

    data = []
    labels = []

    for i, assunto in enumerate(assuntos):

        subreddit_data = api_reddit.subreddit(assunto).new(limit=1000)

        posts = [post.selftext for post in filter(mask, subreddit_data)]

        data.extend(posts)
        labels.extend([i] * len(posts))

        print(f'Número de posts do assunto r/{assunto}: {len(posts)}',
              f'\nUm dos posts extráidos: {posts[0][:600]}...\n',
              "_" * 80 + '\n')
    
    return data, labels


## Split data into train and test

TEST_SIZE = .2
RANDOM_STATE = 0

def split_data(data, labels):

    print(f'Split {100 * TEST_SIZE}% dos dados para teste e avaliação do modelo...')

    X_treino, X_teste, y_treino, y_teste = train_test_split(data, labels, test_size = TEST_SIZE, random_state = RANDOM_STATE)

    print(f'{len(y_teste)} amostras de teste')

    return X_treino, X_teste, y_treino, y_teste


## Data Preprocessing and Attribute Extraction

MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30

def preprocessing_pipeline():

    pattern = r'\W|\d|http.*\s+|www.*\s+'
    preprocessor = lambda text: re.sub(pattern, ' ', text)

    vectorizer = TfidfVectorizer(preprocessor = preprocessor, stop_words = 'english', min_df = MIN_DOC_FREQ)

    decomposition = TruncatedSVD(n_components = N_COMPONENTS, n_iter = N_ITER)

    pipeline = [('tfidf', vectorizer), ('svd', decomposition)]

    return pipeline


## MODEL SELECTION

N_NEIGHBORS = 4
CV = 3

def criar_modelos():

    modelo_1 = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
    modelo_2 = RandomForestClassifier(random_state = RANDOM_STATE)
    modelo_3 = LogisticRegressionCV(cv = CV, random_state = RANDOM_STATE)

    modelos = [('KNN', modelo_1), ('RandomForest', modelo_2), ('LogReg', modelo_3)]

    return modelos


## Models training and evaluation

def treina_avalia(modelos, pipeline, X_treino, X_teste, y_treino, y_teste):

    resultados = []

    for name, modelo in modelos:

        pipe = Pipeline(pipeline + [(name, modelo)])

        print(f'Treinando o modelo {name} com dados do treino...')
        pipe.fit(X_treino, y_treino)

        y_pred = pipe.predict(X_teste)

        report = classification_report(y_teste, y_pred)
        print('Relatório de Classificação\n', report)

        resultados.append([modelo, {'model': name, 'predictions': y_pred, 'report': report}])
    
    return resultados


## Pipeline execution for all models

if __name__ == '__main__':

    data, labels = carrega_dados()

    X_treino, X_teste, y_treino, y_teste = split_data(data, labels)

    pipeline = preprocessing_pipeline()

    all_models = criar_modelos()

    resultados = treina_avalia(all_models, pipeline, X_treino, X_teste, y_treino, y_teste)

print('Concluído com sucesso!')


## Viewing results

def plot_distribution():
    _, counts = np.unique(labels, return_counts = True)
    sns.set_theme(style = 'whitegrid')
    plt.figure(figsize = (15, 6), dpi = 120)
    plt.title(('Número de Posts Por Assunto'))
    sns.barplot(x = assuntos, y = counts)
    plt.legend([' '.join([f.title(), f'- {c} posts']) for f, c in zip(assuntos, counts)])
    plt.show()

def plot_confusion(result):
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


# evaluation
plot_distribution()

# KNN results
plot_confusion(resultados[0])

# RandomForest results
plot_confusion(resultados[1])

# Logistic Regression results
plot_confusion(resultados[2])