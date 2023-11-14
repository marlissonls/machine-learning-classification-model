from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


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