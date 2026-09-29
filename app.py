import streamlit as st
import chromadb
from openai import OpenAI
import re
import os
from db_builder import build_database, DB_PATH, COLLECTION_NAME

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(
    page_title="Поиск по СНиПам",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== КАСТОМНЫЙ CSS ====================
st.markdown("""
<style>
    /* Основной фон */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    /* Заголовок */
    .main-header {
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-subheader {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    }
    
    /* Поле ввода */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #cbd5e1;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Карточки примеров */
    .example-btn {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .example-btn:hover {
        border-color: #2563eb;
        background: #eff6ff;
    }
    
    /* Сайдбар */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: white;
    }
    
    /* Карточки документов */
    .doc-card {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    
    /* Ответ */
    .answer-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== ИНИЦИАЛИЗАЦИЯ ====================
@st.cache_resource
def load_client():
    api_key = st.secrets["CLOUD_API_KEY"]
    return OpenAI(
        api_key=api_key,
        base_url="https://foundation-models.api.cloud.ru/v1",
        timeout=300.0,
        max_retries=2
    )


client = load_client()


@st.cache_resource(show_spinner=False)
def load_collection():
    need_build = not os.path.exists(DB_PATH)

    if not need_build:
        try:
            chroma_client = chromadb.PersistentClient(path=DB_PATH)
            chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception:
            need_build = True

    if need_build:
        with st.spinner("🔨 Первый запуск: собираю векторную базу. Это займёт 2-5 минут..."):
            progress_placeholder = st.empty()

            def show_progress(msg):
                progress_placeholder.info(msg)

            build_database(progress_callback=show_progress)
            progress_placeholder.success("✅ База собрана!")

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return chroma_client.get_collection(name=COLLECTION_NAME)


collection = load_collection()


@st.cache_data
def count_sources():
    try:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        sources = set(m["source"] for m in all_meta if m and "source" in m)
        return sorted(sources)
    except Exception:
        return []


sources_list = count_sources()

# ==================== ИСТОРИЯ ВОПРОСОВ ====================
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== САЙДБАР ====================
with st.sidebar:
    st.markdown("## 📚 База знаний")
    
    # Счётчик документов
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"### {len(sources_list)}")
    with col2:
        st.markdown("**документов**")
        st.markdown("*загружено в базу*")
    
    st.markdown("---")
    
    # Список документов в карточках
    st.markdown("### 📄 Документы")
    for src in sources_list:
        # Определяем иконку по типу
        if "ГОСТ" in src:
            icon = "📘"
        elif "СП" in src:
            icon = "📗"
        else:
            icon = "📄"
        
        # Убираем .txt из имени
        display_name = src.replace(".txt", "")
        
        st.markdown(
            f'<div class="doc-card">{icon} <b>{display_name}</b></div>',
            unsafe_allow_html=True
        )
    
    # История
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕐 История")
        for q in st.session_state.history[-5:]:
            st.markdown(f"• {q}")


# ==================== ОСНОВНОЙ КОНТЕНТ ====================
st.markdown('<h1 class="main-header">🏗️ Поиск по строительным нормам</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subheader">Задайте вопрос — программа найдёт ответ в СП, СНиП и ГОСТ с указанием источника.</p>', unsafe_allow_html=True)

# Примеры вопросов
st.markdown("**💡 Примеры вопросов:**")
example_cols = st.columns(4)
examples = [
    "толщина слоя асфальта",
    "допуски по кернам",
    "поперечный уклон дороги",
    "требования к мостам"
]

if "selected_example" not in st.session_state:
    st.session_state.selected_example = ""

for i, ex in enumerate(examples):
    with example_cols[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.selected_example = ex

# Поле ввода
question = st.text_input(
    "Ваш вопрос:",
    value=st.session_state.selected_example,
    placeholder="Например: допуски по асфальту",
    key="question_input"
)

# Кнопка
col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("🔍 Найти ответ", type="primary", use_container_width=True)

# Обработка Enter (кнопка или Enter)
if ask_button or (question and question != st.session_state.selected_example and st.session_state.get("last_question") != question):
    if not question.strip():
        st.warning("Пожалуйста, введите вопрос.")
    else:
        # Сохраняем в историю
        if question not in st.session_state.history:
            st.session_state.history.append(question)
        st.session_state.last_question = question
        st.session_state.selected_example = ""
        
        with st.spinner("Ищу ответ в документах..."):
            candidates = []

            vector_results = collection.query(
                query_texts=[question],
                n_results=30
            )
            if vector_results['documents'] and vector_results['documents'][0]:
                for i, doc in enumerate(vector_results['documents'][0]):
                    meta = vector_results['metadatas'][0][i]
                    candidates.append({
                        'text': doc,
                        'source': meta['source'],
                        'section': meta.get('section', ''),
                        'tables': meta.get('tables', ''),
                        'type': 'векторный'
                    })

            table_matches = re.findall(
                r'таблиц[аы]?\s*([А-ЯA-Z]?\.?\d+(?:\.\d+)?)',
                question,
                re.IGNORECASE
            )
            if table_matches:
                for table_num in table_matches:
                    try:
                        table_results = collection.get(
                            where_document={"$contains": f"Таблица {table_num}"},
                            limit=10
                        )
                        if table_results['documents']:
                            for i, doc in enumerate(table_results['documents']):
                                meta = table_results['metadatas'][i]
                                candidates.append({
                                    'text': doc,
                                    'source': meta['source'],
                                    'section': meta.get('section', ''),
                                    'tables': meta.get('tables', ''),
                                    'type': f'таблица {table_num}'
                                })
                    except Exception:
                        pass

            if re.search(r'допуск|отклонени', question, re.IGNORECASE):
                keyword_results = collection.query(
                    query_texts=["допуск отклонение не более мм"],
                    n_results=20
                )
                if keyword_results['documents'] and keyword_results['documents'][0]:
                    for i, doc in enumerate(keyword_results['documents'][0]):
                        meta = keyword_results['metadatas'][0][i]
                        candidates.append({
                            'text': doc,
                            'source': meta['source'],
                            'section': meta.get('section', ''),
                            'tables': meta.get('tables', ''),
                            'type': 'ключевые слова'
                        })

            filtered = []
            for c in candidates:
                text_lower = c['text'].lower()
                has_number = bool(re.search(r'\d+', c['text']))
                has_keyword = any(
                    word in text_lower
                    for word in ['допуск', 'отклонен', 'мм', 'таблиц', 'не более']
                )
                if has_number and has_keyword:
                    filtered.append(c)

            if not filtered:
                filtered = candidates

            seen = set()
            unique_filtered = []
            for c in filtered:
                if c['text'] not in seen:
                    seen.add(c['text'])
                    unique_filtered.append(c)

            unique_filtered = unique_filtered[:25]

            context = ""
            sources = []
            for c in unique_filtered:
                ref = c['source']
                if c['section']:
                    ref += f", раздел {c['section']}"
                if c['tables']:
                    ref += f", таблица {c['tables']}"
                context += f"\n\n--- Источник: {ref} ---\n{c['text']}"
                if ref not in sources:
                    sources.append(ref)

            # Красивая карточка с источниками
            st.markdown(f"""
            <div style="background: #eff6ff; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                <b>📖 Отобрано фрагментов:</b> {len(unique_filtered)}
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(f"📚 Показать источники ({len(sources)})", expanded=False):
                for src in sources[:15]:
                    st.markdown(f"• {src}")

            prompt = f"""Не размышляй. Сразу давай ответ.
Ты — эксперт по строительным нормам и правилам.
Отвечай подробно. Приведи ВСЕ найденные допуски и отклонения из фрагментов.
Структурируй ответ: раздели на пункты, для каждого укажи значение и источник.
Если в фрагментах нет ответа — честно скажи об этом.
Обязательно укажи, из какого документа, раздела и пункта взята информация.

ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}

ВОПРОС:
{question}
"""

            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen3-30B-A3B",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    extra_body={"enable_thinking": False}
                )

                answer = response.choices[0].message.content

                st.success("✅ Ответ найден!")
                st.markdown("### 📖 Ответ:")
                st.markdown(answer)

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")