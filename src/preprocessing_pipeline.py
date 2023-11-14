import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


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