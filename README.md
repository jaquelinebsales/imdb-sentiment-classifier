# 🎬 IMDb Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A Machine Learning web app to classify movie reviews as **positive** or **negative**.

---

## 📊 Dataset

Este projeto utiliza o [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

---

## ⚙️ Como Executar

### 🔹 Pré-requisitos
- Python 3.8 ou superior
- Pip (gerenciador de pacotes)

### 🔹 Passo a Passo

1. **Clone o repositório**
```bash
git clone https://github.com/seu-usuario/imdb-sentiment-classifier.git
cd imdb-sentiment-classifier
```

2. **Crie e ative um ambiente virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Treine o modelo**
```bash
python model.py
```
Este comando vai limpar os dados, treinar o modelo e salvar os arquivos .pkl na pasta models/.

5. **Execute a aplicação web**

```bash
python -m streamlit run app.py
```
A aplicação irá rodar na porta 8501. Acesse http://localhost:8501 no navegador.
