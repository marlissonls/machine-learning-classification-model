from sklearn.model_selection import train_test_split


TEST_SIZE = .2
RANDOM_STATE = 0

def split_data(data, labels):

    print(f'Split {100 * TEST_SIZE}% dos dados para teste e avaliação do modelo...')

    X_treino, X_teste, y_treino, y_teste = train_test_split(data, labels, test_size = TEST_SIZE, random_state = RANDOM_STATE)

    print(f'{len(y_teste)} amostras de teste')

    return X_treino, X_teste, y_treino, y_teste