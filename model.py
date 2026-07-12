import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# 1. Carregar os dados
print("Carregando dados...")
df = pd.read_csv('IMDB_Dataset.csv')

# 2. Verificar balanceamento
print(f"Distribuição das classes:\n{df['sentiment'].value_counts()}")

# 3. Função para limpar texto
def limpar_texto(texto):
    texto = re.sub(r'<.*?>', '', texto)          # Remove tags HTML
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)     # Remove números e pontuação
    texto = texto.lower()                         # Converte para minúsculas
    return texto

print("Limpando textos...")
df['review_limpa'] = df['review'].apply(limpar_texto)

# 4. Separar features e target
X = df['review_limpa']
y = df['sentiment']  # 'positive' ou 'negative'

# 5. Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Vetorizar o texto com TF-IDF
print("Vetorizando textos...")
vetorizador = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vetorizador.fit_transform(X_train)
X_test_vec = vetorizador.transform(X_test)

# 7. Treinar o modelo MultinomialNB
print("Treinando modelo Naive Bayes...")
modelo = MultinomialNB(alpha=1.0)
modelo.fit(X_train_vec, y_train)

# 8. Avaliar o modelo
print("\n--- Avaliação do Modelo ---")
y_pred = modelo.predict(X_test_vec)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 9. Salvar o modelo e o vetorizador
print("\nSalvando modelo e vetorizador...")
joblib.dump(modelo, 'models/sentiment_model.pkl')
joblib.dump(vetorizador, 'models/vectorizer.pkl')
print("Arquivos 'models/sentiment_model.pkl' e 'models/vectorizer.pkl' salvos com sucesso!")