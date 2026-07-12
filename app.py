import streamlit as st
import joblib
import re
import time

# 1. Configurar a página
st.set_page_config(
    page_title="Classificador de Sentimentos - IMDb",
    page_icon="🎬",
    layout="centered"
)

# 2. Título e descrição
st.title("🎬 Análise de Sentimentos de Filmes")
st.markdown("Digite ou cole a avaliação de um filme para saber se ela é **Positiva** ou **Negativa**.")

# 3. Função para limpar texto (deve ser a mesma usada no treino)
def limpar_texto(texto):
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = texto.lower()
    return texto

# 4. Carregar o modelo e o vetorizador (com cache para performance)
@st.cache_resource
def carregar_modelo():
    try:
        modelo = joblib.load('models/sentiment_model.pkl')
        vetorizador = joblib.load('models/vectorizer.pkl')
        return modelo, vetorizador
    except FileNotFoundError:
        st.error("Arquivos do modelo não encontrados. Execute o 'model.py' primeiro!")
        return None, None

modelo, vetorizador = carregar_modelo()

# 5. Interface do usuário
if modelo is not None:
    # Campo de texto para a crítica
    review_input = st.text_area(
        "Escreva sua avaliação aqui:",
        height=150,
        placeholder="Ex: Este filme é incrível, adorei a atuação dos atores!"
    )
    
    # Botão para classificar
    if st.button("🔍 Classificar Sentimento", type="primary"):
        if review_input.strip():
            # a) Limpar o texto
            review_limpo = limpar_texto(review_input)
            
            # b) Transformar em vetor numérico
            review_vetor = vetorizador.transform([review_limpo])
            
            # c) Fazer a previsão
            with st.spinner("Classificando..."):
                time.sleep(0.5)  # Para dar um efeito de processamento
                predicao = modelo.predict(review_vetor)[0]
                probabilidades = modelo.predict_proba(review_vetor)[0]
                confianca = max(probabilidades) * 100
                
            # d) Exibir o resultado com base na classe
            if predicao == 'positive':
                st.success(f"😊 **Positivo** (Confiança: {confianca:.2f}%)")
                st.balloons()
            else:
                st.error(f"😞 **Negativo** (Confiança: {confianca:.2f}%)")
                
        else:
            st.warning("Por favor, escreva uma avaliação para classificar.")

    # 6. Exemplo rápido para testar
    with st.expander("💡 Exemplos para testar"):
        st.write("**Clique no botão abaixo para preencher o campo com um exemplo:**")
        if st.button("Crítica Positiva"):
            st.session_state.review = "This movie was absolutely fantastic! The acting was superb and the story kept me on the edge of my seat."
            st.rerun()
        if st.button("Crítica Negativa"):
            st.session_state.review = "I hated this film. The plot made no sense and the characters were completely unlikable."
            st.rerun()

    # 7. Rodapé
    st.markdown("---")
    st.caption("Desenvolvido para o Trabalho Final de Probabilidade e Estatística - IFCE")

# 8. Tratar o estado da sessão para os exemplos
if 'review' in st.session_state:
    review_input = st.text_area(
        "Escreva sua avaliação aqui:",
        value=st.session_state.review,
        height=150
    )
    # Limpa o estado depois de usar, para não travar
    del st.session_state.review