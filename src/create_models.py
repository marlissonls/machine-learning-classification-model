from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

RANDOM_STATE = 0
N_NEIGHBORS = 4
CV = 3

def criar_modelos():

    modelo_1 = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
    modelo_2 = RandomForestClassifier(random_state = RANDOM_STATE)
    modelo_3 = LogisticRegressionCV(cv = CV, random_state = RANDOM_STATE)

    modelos = [('KNN', modelo_1), ('RandomForest', modelo_2), ('LogReg', modelo_3)]

    return modelos