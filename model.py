import pandas as pd
import joblib
import nltk
import re
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

nltk.download('stopwords')

if not os.path.exists('models'):
    os.makedirs('models')

df = pd.read_csv('IMDB_Dataset.csv')

def limpar_texto(texto):
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = texto.lower()
    return texto

df['review_limpa'] = df['review'].apply(limpar_texto)

X = df['review_limpa']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

stop_words_custom = list(stopwords.words('english'))

negacoes = ['not', 'no', 'nor', 'neither', 'ain\'t', 'aren\'t', 'couldn\'t', 'didn\'t', 'doesn\'t', 'hadn\'t', 'hasn\'t', 'haven\'t', 'isn\'t', 'wasn\'t', 'weren\'t', 'won\'t', 'wouldn\'t']
stop_words_custom = [word for word in stop_words_custom if word not in negacoes]

vetorizador = TfidfVectorizer(
    max_features=12500,
    stop_words=stop_words_custom,
    ngram_range=(1, 2)
)

X_train_vec = vetorizador.fit_transform(X_train)
X_test_vec = vetorizador.transform(X_test)

modelo = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

modelo.fit(X_train_vec, y_train)

y_pred = modelo.predict(X_test_vec)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

joblib.dump(modelo, 'models/sentiment_model.pkl')
joblib.dump(vetorizador, 'models/vectorizer.pkl')