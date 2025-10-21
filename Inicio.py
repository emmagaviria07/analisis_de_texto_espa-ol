import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ===============================
# 🌅 INTERFAZ VISUAL RENOVADA
# ===============================

st.set_page_config(page_title="Analizador de Textos", page_icon="🧠", layout="wide")

# Imagen de cabecera (opcional, comenta si no tienes imagen)
# st.image("encabezado.png", use_container_width=True)

st.title("🧠 Analizador Semántico en Español")
st.markdown("""
Explora cómo la **inteligencia artificial** analiza similitudes entre textos usando TF-IDF.  
Ingresa tus documentos y una pregunta para descubrir cuál tiene la respuesta más cercana. 🌟
""")

# Documentos de ejemplo — CAMBIADOS VISUALMENTE
default_docs = """El sol brilla sobre las montañas al amanecer.
El río fluye lentamente hacia el mar.
Los árboles crecen altos en el bosque verde.
Las flores se abren cuando llega la primavera.
El viento sopla entre las hojas del parque.
Los pájaros vuelan en grupos buscando el cielo despejado."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# ===============================
# 💡 INTERFAZ DE USUARIO
# ===============================

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📜 Ingresa tus textos (uno por línea):", default_docs, height=180)
    question = st.text_input("❓ Escribe tu consulta:", "¿Dónde sopla el viento?")

with col2:
    st.markdown("### 🌼 Preguntas sugeridas:")
    
    if st.button("¿Dónde sopla el viento?", use_container_width=True):
        st.session_state.question = "¿Dónde sopla el viento?"
        st.rerun()
        
    if st.button("¿Qué sucede con las flores en primavera?", use_container_width=True):
        st.session_state.question = "¿Qué sucede con las flores en primavera?"
        st.rerun()
        
    if st.button("¿Qué hacen los pájaros en el cielo?", use_container_width=True):
        st.session_state.question = "¿Qué hacen los pájaros en el cielo?"
        st.rerun()
        
    if st.button("¿Cómo se comporta el río?", use_container_width=True):
        st.session_state.question = "¿Cómo se comporta el río?"
        st.rerun()
        
    if st.button("¿Qué elementos hay en el bosque?", use_container_width=True):
        st.session_state.question = "¿Qué elementos hay en el bosque?"
        st.rerun()

# Actualizar la pregunta seleccionada
if 'question' in st.session_state:
    question = st.session_state.question

# ===============================
# 🧮 ANÁLISIS TF-IDF
# ===============================

if st.button("🚀 Analizar texto", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Por favor, ingresa al menos un texto para analizar.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta antes de continuar.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )
        
        X = vectorizer.fit_transform(documents)
        
        st.markdown("### 📊 Resultados TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Texto {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # Calcular similitud
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        # ===============================
        # 🎯 RESULTADOS VISUALES
        # ===============================
        st.markdown("### 🎯 Resultado del análisis")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:
            st.success(f"**Respuesta más similar:** {best_doc}")
            st.info(f"📈 Nivel de similitud: {best_score:.3f}")
        else:
            st.warning(f"**Respuesta con baja coincidencia:** {best_doc}")
            st.info(f"📉 Nivel de similitud: {best_score:.3f}")

# ===============================
# 🎨 PIE DE PÁGINA
# ===============================
st.markdown("""
---
🌿 *Versión visual renovada del demo TF-IDF.*  
Creado para mostrar cómo pequeñas variaciones visuales pueden transformar la experiencia del usuario ✨
""")
