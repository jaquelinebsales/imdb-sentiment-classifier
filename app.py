import json
import streamlit as st
import joblib
import re
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÕES INICIAIS ---
st.set_page_config(
    page_title="Classificador de Sentimentos - IMDb",
    page_icon="🎬",
    layout="wide"
)

# --- 2. FUNÇÃO DE LIMPEZA DE TEXTO ---
def limpar_texto(texto):
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = texto.lower()
    return texto

# --- 3. FUNÇÃO DE CLASSIFICAÇÃO COM THRESHOLD ---
def classificar_com_threshold(texto, vetorizador, modelo, 
                              threshold_inferior=0.50, 
                              threshold_superior=0.50):
    """
    Classifica uma crítica com zona neutra (experiência média).
    
    threshold_inferior: Probabilidades abaixo disso são NEGATIVAS.
    threshold_superior: Probabilidades acima disso são POSITIVAS.
    O que ficar no meio é classificado como NEUTRO.
    """
    texto_limpo = limpar_texto(texto)
    texto_vetorizado = vetorizador.transform([texto_limpo])
    
    probabilidades = modelo.predict_proba(texto_vetorizado)[0]
    prob_negativo = probabilidades[0]
    prob_positivo = probabilidades[1]
    
    if prob_positivo >= threshold_superior:
        sentimento = "POSITIVO"
        classe = "positive"
        confianca = prob_positivo * 100
        cor = "#16a34a"
        bg = "#dcfce7"
        emoji = "😊"
    elif prob_positivo <= threshold_inferior:
        sentimento = "NEGATIVO"
        classe = "negative"
        confianca = prob_negativo * 100
        cor = "#dc2626"
        bg = "#fee2e2"
        emoji = "😞"
    else:
        sentimento = "NEUTRO (Experiência Média)"
        classe = "neutral"
        confianca = (1.0 - abs(prob_positivo - 0.5) * 2) * 100
        cor = "#f59e0b"
        bg = "#fef3c7"
        emoji = "😐"
    
    return {
        'sentimento': sentimento,
        'classe': classe,
        'confianca': confianca,
        'prob_positivo': prob_positivo,
        'prob_negativo': prob_negativo,
        'cor': cor,
        'bg': bg,
        'emoji': emoji
    }

# --- 4. CARREGAMENTO DO MODELO ---
@st.cache_resource
def carregar_modelo():
    try:
        modelo = joblib.load('models/sentiment_model.pkl')
        vetorizador = joblib.load('models/vectorizer.pkl')
        return modelo, vetorizador
    except FileNotFoundError:
        st.error("🚨 Arquivos do modelo não encontrados. Execute o 'model.py' primeiro!")
        return None, None

# --- 5. CARREGAMENTO DAS MÉTRICAS ---
@st.cache_data
def carregar_metricas():
    try:
        with open('models/metricas.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("🚨 Arquivo de métricas não encontrado. Execute o 'model.py' primeiro!")
        return None

modelo, vetorizador = carregar_modelo()
metricas = carregar_metricas()

# --- 6. INTERFACE PRINCIPAL ---
st.title("🎬 Análise de Sentimentos de Filmes")
st.markdown("Classifique avaliações de filmes como **Positivas**, **Negativas** ou **Neutras** usando Machine Learning.")

# Barra lateral com informações
with st.sidebar:
    st.header("📊 Sobre o Modelo")
    if metricas:
        st.markdown(f"""
        - **Modelo:** {metricas['params']['modelo']}
        - **Acurácia:** {metricas['acuracia']:.2%}
        - **Dataset:** 50.000 críticas do IMDb
        - **Vetorização:** TF-IDF + N-gramas (1,2)
        """)
    else:
        st.markdown("""
        - **Modelo:** Regressão Logística
        - **Acurácia:** ~89,6%
        - **Dataset:** 50.000 críticas do IMDb
        - **Vetorização:** TF-IDF + N-gramas (1,2)
        """)
    
    st.header("📝 Explicação")
    st.markdown("""
    O modelo analisa as palavras da sua crítica e calcula a probabilidade dela ser positiva ou negativa.
    
    **Exemplos de palavras positivas:**
    - excellent, amazing, wonderful, fantastic
    
    **Exemplos de palavras negativas:**
    - terrible, awful, boring, disappointing
    """)
    
    st.header("🔍 Dica")
    st.markdown("""
    Para melhores resultados, escreva críticas em **inglês** (o modelo foi treinado com dados em inglês).
    """)
    
    st.header("⚙️ Ajuste de Sensibilidade")
    st.markdown("""
    Ajuste os limiares para controlar a zona neutra:
    - **Limiar Inferior:** abaixo disso = Negativo
    - **Limiar Superior:** acima disso = Positivo
    - **Entre os dois:** Neutro
    """)
    
    threshold_inferior = st.slider(
        "Limiar Inferior (Negativo)", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.50, 
        step=0.05
    )
    
    threshold_superior = st.slider(
        "Limiar Superior (Positivo)", 
        min_value=0.5, 
        max_value=0.9, 
        value=0.50, 
        step=0.05
    )

# Criação das Abas
tab1, tab2, tab3 = st.tabs(["⌨️ Análise", "📈 Estatísticas do Modelo", "🔍 Palavras Mais Importantes"])

# --- ABA 1: ANÁLISE INDIVIDUAL ---
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        review_input = st.text_area(
            "Escreva sua avaliação aqui:",
            height=150,
            placeholder="Ex: This movie was absolutely fantastic! The acting was superb..."
        )
    
    with col2:
        st.markdown("### 💡 Exemplos")
        if st.button("🎬 Crítica Positiva", width='stretch'):
            st.session_state.review = "This movie was absolutely fantastic! The acting was superb and the story kept me on the edge of my seat."
            st.rerun()
        if st.button("💀 Crítica Negativa", width='stretch'):
            st.session_state.review = "I hated this film. The plot made no sense and the characters were completely unlikable."
            st.rerun()
        if st.button("⚖️ Crítica Neutra", width='stretch'):
            st.session_state.review = "The movie was okay, not great but not terrible either. Average experience."
            st.rerun()
    
    if st.button("🔍 Classificar Sentimento", type="primary", width='stretch'):
        if review_input and review_input.strip():
            with st.spinner("🔄 Classificando..."):
                time.sleep(0.5)
                resultado = classificar_com_threshold(
                    review_input, 
                    vetorizador, 
                    modelo,
                    threshold_inferior,
                    threshold_superior
                )
            
            st.markdown(f"""
            <div style="background-color: {resultado['bg']}; padding: 25px; border-radius: 12px; border-left: 6px solid {resultado['cor']};">
                <h2 style="color: {resultado['cor']}; margin: 0;">{resultado['emoji']} {resultado['sentimento']}</h2>
                <p style="color: {resultado['cor']}; margin: 5px 0 0 0; font-size: 18px;">
                    Confiança: {resultado['confianca']:.2f}%
                </p>
                <p style="color: {resultado['cor']}; margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">
                    {review_input[:200]}{'...' if len(review_input) > 200 else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if resultado['classe'] == 'positive':
                st.balloons()

            st.write("\n \n")

            st.write("### 📊 Distribuição de Probabilidades")
            
            prob_df = pd.DataFrame({
                'Sentimento': ['Negativo', 'Positivo'],
                'Probabilidade': [resultado['prob_negativo'] * 100, 
                                 resultado['prob_positivo'] * 100]
            })
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#dc2626', '#16a34a']
            bars = ax.bar(prob_df['Sentimento'], prob_df['Probabilidade'], color=colors)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Probabilidade (%)')
            ax.set_title('Probabilidade por Sentimento')
            
            ax.axhline(y=threshold_inferior * 100, color='orange', linestyle='--', 
                       label=f'Limiar Inferior ({threshold_inferior*100:.0f}%)')
            ax.axhline(y=threshold_superior * 100, color='purple', linestyle='--', 
                       label=f'Limiar Superior ({threshold_superior*100:.0f}%)')
            ax.legend()
            
            for bar, prob in zip(bars, prob_df['Probabilidade']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{prob:.1f}%', ha='center', va='bottom', fontsize=12)
            
            st.pyplot(fig)
            
            st.caption(f"🔹 Zona Negativa: < {threshold_inferior*100:.0f}% | 🔸 Zona Neutra: {threshold_inferior*100:.0f}% - {threshold_superior*100:.0f}% | 🔹 Zona Positiva: > {threshold_superior*100:.0f}%")
            
        else:
            st.warning("⚠️ Por favor, escreva uma avaliação para classificar.")

    if 'review' in st.session_state:
        review_input = st.text_area(
            "Escreva sua avaliação aqui:",
            value=st.session_state.review,
            height=150
        )
        del st.session_state.review

# --- ABA 2: ESTATÍSTICAS DO MODELO ---
with tab2:
    st.subheader("📈 Estatísticas do Modelo")
    
    if metricas:
        acuracia = metricas['acuracia']
        matriz = metricas['matriz_confusao']
        vn, fp, fn, vp = matriz[0][0], matriz[0][1], matriz[1][0], matriz[1][1]
        
        neg_precisao = metricas['relatorio']['negativo']['precisao']
        neg_recall = metricas['relatorio']['negativo']['recall']
        neg_f1 = metricas['relatorio']['negativo']['f1']
        
        pos_precisao = metricas['relatorio']['positivo']['precisao']
        pos_recall = metricas['relatorio']['positivo']['recall']
        pos_f1 = metricas['relatorio']['positivo']['f1']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Acurácia", f"{acuracia:.2%}")
        col2.metric("📊 Precisão Média", f"{(neg_precisao + pos_precisao)/2:.2%}", "✅")
        col3.metric("🔄 Recall Médio", f"{(neg_recall + pos_recall)/2:.2%}", "✅")
        
        st.write("### 📊 Métricas por Classe")
        df_metricas = pd.DataFrame({
            'Classe': ['Negativo', 'Positivo'],
            'Precisão': [f"{neg_precisao:.2%}", f"{pos_precisao:.2%}"],
            'Recall': [f"{neg_recall:.2%}", f"{pos_recall:.2%}"],
            'F1-Score': [f"{neg_f1:.2%}", f"{pos_f1:.2%}"],
            'Suporte': [metricas['relatorio']['negativo']['suporte'], 
                        metricas['relatorio']['positivo']['suporte']]
        })
        st.dataframe(df_metricas, width='stretch', hide_index=True)
        
        st.write("### 📊 Matriz de Confusão")
        st.code(f"""
        [[{vn:>4}  {fp:>4}]   ← Negativos Reais: {vn} certos, {fp} errados
         [{fn:>4}  {vp:>4}]]  ← Positivos Reais: {fn} errados, {vp} certos
        """)
        
        st.write("### 📝 Interpretação")
        st.markdown(f"""
        - **Verdadeiros Negativos:** {vn} críticas negativas foram classificadas corretamente.
        - **Falsos Positivos:** {fp} críticas negativas foram classificadas como positivas (erro).
        - **Falsos Negativos:** {fn} críticas positivas foram classificadas como negativas (erro).
        - **Verdadeiros Positivos:** {vp} críticas positivas foram classificadas corretamente.
        
        O modelo está **equilibrado** e tem desempenho consistente em ambas as classes.
        """)
        
        st.write("### 🛠️ Parâmetros do Modelo")
        st.json(metricas['params'])
        
    else:
        st.warning("⚠️ Métricas não disponíveis. Execute o 'model.py' primeiro!")

# --- ABA 3: PALAVRAS MAIS IMPORTANTES ---
with tab3:
    st.subheader("🔍 Palavras Mais Importantes")
    st.markdown("Estas são as palavras que **mais influenciam** a decisão do modelo.")
    
    if metricas and 'palavras_importantes' in metricas:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 😊 Palavras que indicam **POSITIVO**")
            st.markdown("*(Quanto maior o coeficiente, mais forte a associação)*")
            for item in metricas['palavras_importantes']['positivas']:
                st.markdown(f"- `{item['palavra']}`: **{item['coeficiente']:.4f}**")
        
        with col2:
            st.markdown("### 😞 Palavras que indicam **NEGATIVO**")
            st.markdown("*(Quanto menor o coeficiente, mais forte a associação)*")
            for item in metricas['palavras_importantes']['negativas']:
                st.markdown(f"- `{item['palavra']}`: **{item['coeficiente']:.4f}**")
        
        st.divider()
        
        # Gráfico de barras para visualizar as palavras mais importantes
        st.write("### 📊 Visualização das Palavras Mais Importantes")
        
        # Preparar dados para o gráfico
        palavras_pos = [item['palavra'] for item in metricas['palavras_importantes']['positivas']]
        coef_pos = [item['coeficiente'] for item in metricas['palavras_importantes']['positivas']]
        
        palavras_neg = [item['palavra'] for item in metricas['palavras_importantes']['negativas']]
        coef_neg = [item['coeficiente'] for item in metricas['palavras_importantes']['negativas']]
        
        # Juntar todas as palavras e coeficientes
        todas_palavras = palavras_pos + palavras_neg
        todos_coeficientes = coef_pos + coef_neg
        cores = ['#16a34a'] * len(palavras_pos) + ['#dc2626'] * len(palavras_neg)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(todas_palavras, todos_coeficientes, color=cores)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Coeficiente')
        ax.set_title('Coeficientes das Palavras Mais Importantes')
        
        # Adicionar valores nas barras
        for bar, coef in zip(bars, todos_coeficientes):
            ax.text(bar.get_width() + 0.02 if coef > 0 else bar.get_width() - 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f'{coef:.2f}',
                    va='center',
                    ha='left' if coef > 0 else 'right',
                    fontsize=9)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretação
        st.divider()
        st.write("### 💡 Interpretação")
        st.markdown(f"""
        - A palavra mais associada a **POSITIVO** é **'{metricas['palavras_importantes']['positivas'][0]['palavra']}'** com coeficiente {metricas['palavras_importantes']['positivas'][0]['coeficiente']:.2f}.
        - A palavra mais associada a **NEGATIVO** é **'{metricas['palavras_importantes']['negativas'][0]['palavra']}'** com coeficiente {metricas['palavras_importantes']['negativas'][0]['coeficiente']:.2f}.
        - Esses coeficientes mostram o quanto cada palavra contribui para a decisão do modelo.
        """)
        
    else:
        st.warning("⚠️ Informações de palavras importantes não disponíveis. Execute o 'model.py' primeiro!")

# --- RODAPÉ ---
st.markdown("---")
st.caption("🎓 Desenvolvido para o Trabalho Final de Probabilidade e Estatística - IFCE")
st.caption("👥 Jaqueline Sales | Camila Azevedo | Isaac Cunha | Diego Inácio")