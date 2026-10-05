import streamlit as st
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import re
import os
from datetime import datetime
from db_builder import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# PDF
try:
    import ironpress
    IRONPRESS_OK = True
except ImportError:
    IRONPRESS_OK = False

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(
    page_title="Поиск по СНиПам",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="auto"
)

# ==================== КАСТОМНЫЙ CSS ====================
st.markdown("""
<style>
    .stApp { background: var(--background-color); }
    .main-header {
        color: var(--primary-color);
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .main-subheader {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .sidebar-header {
        color: var(--primary-color);
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 300px !important;
    }
    .doc-card {
        background: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        border-radius: 6px;
        font-size: 0.8rem;
        line-height: 1.3;
    }
    .fragments-info {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        color: var(--primary-color);
        font-weight: 600;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    @media (max-width: 768px) {
        .main-header { font-size: 1.3rem !important; line-height: 1.2; margin-bottom: 0.3rem; }
        .main-subheader { font-size: 0.85rem; margin-bottom: 1rem; line-height: 1.3; }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 3rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        [data-testid="stSidebar"] { min-width: 0 !important; max-width: 100% !important; }
        [data-testid="stSidebar"] .doc-card { font-size: 0.75rem; padding: 0.4rem 0.6rem; }
        .stButton > button, .stFormSubmitButton > button {
            font-size: 0.9rem !important;
            padding: 0.6rem 0.8rem !important;
            min-height: 2.6rem;
        }
        .stFormSubmitButton > button {
            font-size: 1rem !important;
            padding: 0.75rem 1.5rem !important;
            min-height: 3rem !important;
        }
        .stTextInput > div > div > input {
            font-size: 1rem !important;
            padding: 0.75rem !important;
            min-height: 2.75rem;
        }
        .fragments-info { font-size: 0.85rem; padding: 0.6rem 0.8rem; margin: 0.75rem 0; }
        .stExpander { margin-bottom: 0.5rem !important; }
        .stExpander summary { font-size: 0.9rem !important; }
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


@st.cache_resource(show_spinner="Загружаю модель эмбеддингов…")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


embedder = load_embedder()


@st.cache_resource(show_spinner=False)
def load_collection():
    if not os.path.isdir(DB_PATH) or not os.listdir(DB_PATH):
        st.error(
            "❌ База `chroma_db` не найдена в репозитории.\n\n"
            "Соберите её локально: `python db_builder.py`, затем закоммитьте "
            "папку `chroma_db/` и запушьте в `main`."
        )
        st.stop()

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(
            f"❌ Не удалось открыть коллекцию `{COLLECTION_NAME}` в `{DB_PATH}`: {e}"
        )
        st.stop()


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


# ==================== АУДИТ БАЗЫ ====================
def run_audit():
    lines = []
    try:
        all_meta = collection.get(include=["metadatas", "documents"])
    except Exception as e:
        return [f"❌ Ошибка доступа к базе: {e}"]

    metas = all_meta.get("metadatas") or []
    docs = all_meta.get("documents") or []
    sources = sorted(set(m["source"] for m in metas if m and m.get("source")))

    lines.append(f"**Всего чанков:** {len(docs)}")
    lines.append(f"**Всего документов:** {len(sources)}")
    lines.append("")
    lines.append("---")

    for src in sources:
        try:
            r = collection.get(where={"source": src}, include=["metadatas", "documents"])
        except Exception as e:
            lines.append(f"**{src}**: ошибка — {e}")
            continue

        m_list = r.get("metadatas") or []
        d_list = r.get("documents") or []
        n = len(d_list)
        if n == 0:
            lines.append(f"**{src}** — пусто")
            continue

        chapters = sorted(
            set(m.get("chapter", "") for m in m_list if m.get("chapter")),
            key=lambda x: int(x) if x.isdigit() else 999,
        )
        points = sorted(set(m.get("point", "") for m in m_list if m.get("point")))
        tables = sorted(set(m.get("table_number", "") for m in m_list if m.get("table_number")))

        no_chapter = sum(1 for m in m_list if not m.get("chapter"))
        no_point = sum(1 for m in m_list if not m.get("point"))
        short = sum(1 for d in d_list if len(d) < 200)
        no_digits = sum(1 for d in d_list if not re.search(r"\d", d))
        avg_len = sum(len(d) for d in d_list) / n if n else 0

        lines.append(f"### {src}")
        lines.append(f"- Чанков: **{n}**, средняя длина: **{avg_len:.0f}** символов")
        lines.append(f"- Главы ({len(chapters)}): `{chapters}`")
        lines.append(f"- Пунктов: **{len(points)}** — первые: `{points[:20]}`")
        lines.append(f"- Таблицы ({len(tables)}): `{tables}`")
        lines.append(f"- Пустой chapter: **{no_chapter}**, пустой point: **{no_point}**")
        lines.append(f"- Чанков <200 символов: **{short}**, без цифр: **{no_digits}**")
        lines.append("")

    return lines


# ==================== SESSION STATE ====================
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "current_fragments" not in st.session_state:
    st.session_state.current_fragments = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# ==================== САЙДБАР ====================
with st.sidebar:
    st.markdown('<div class="sidebar-header">📚 База знаний</div>', unsafe_allow_html=True)
    st.markdown(f"**{len(sources_list)}** документов загружено")
    st.markdown("---")

    st.markdown("### 🎯 Фильтр по документам")
    selected_sources = st.multiselect(
        "Искать только в:",
        options=sources_list,
        default=[],
        placeholder="Выберите документы...",
        format_func=lambda x: x.replace(".txt", "")[:40] + "...",
        key="source_filter"
    )

    if selected_sources:
        st.caption(f"🔍 Поиск в **{len(selected_sources)}** документ(ах)")
    else:
        st.caption("🔍 Поиск во **всех** документах")

    st.markdown("---")
    st.markdown("### 📄 Документы")
    for src in sources_list:
        if "ГОСТ" in src:
            icon = "📘"
        elif "СП" in src:
            icon = "📗"
        else:
            icon = "📄"

        clean_name = src.replace(".txt", "")
        if len(clean_name) > 40:
            truncated = clean_name[:40]
            if ' ' in truncated:
                truncated = truncated.rsplit(' ', 1)[0]
            display_name = truncated + "..."
        else:
            display_name = clean_name

        st.markdown(f'<div class="doc-card">{icon} {display_name}</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕐 История")
        st.caption("Нажми на вопрос, чтобы повторить")
        for i, q in enumerate(reversed(st.session_state.history[-5:])):
            if st.button(f"↻ {q[:50]}", key=f"hist_{i}", use_container_width=True):
                st.session_state.current_question = q
                st.rerun()

    if st.session_state.feedback:
        st.markdown("---")
        st.markdown("### 📊 Оценки")
        ups = sum(1 for v in st.session_state.feedback.values() if v == 1)
        downs = sum(1 for v in st.session_state.feedback.values() if v == 0)
        st.markdown(f"👍 **{ups}** · 👎 **{downs}**")

    st.markdown("---")
    st.markdown("### 🔎 Аудит базы")
    if st.button("Запустить аудит", use_container_width=True, key="audit_btn"):
        with st.spinner("Считаю статистику…"):
            st.session_state["audit_report"] = run_audit()

    if st.session_state.get("audit_report"):
        with st.expander("Показать отчёт", expanded=True):
            for line in st.session_state["audit_report"]:
                st.markdown(line)


# ==================== ОСНОВНОЙ КОНТЕНТ ====================
st.markdown('<h1 class="main-header">🏗️ Поиск по СНиПам</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subheader">Задайте вопрос — программа найдёт ответ в СП, СНиП и ГОСТ с указанием источника.</p>', unsafe_allow_html=True)


# ==================== ФОРМА ====================
with st.form("search_form", clear_on_submit=False):
    question = st.text_input(
        "Ваш вопрос:",
        placeholder="Например: допуски по асфальту",
        key="question_field"
    )
    ask_button = st.form_submit_button("🔍 Найти ответ", type="primary", use_container_width=False)

# ==================== ОБРАБОТКА ====================
if ask_button:
    if not question.strip():
        st.warning("Пожалуйста, введите вопрос.")
    else:
        if question not in st.session_state.history:
            st.session_state.history.append(question)

        status_placeholder = st.empty()
        status_placeholder.info("⏳ Ищу ответ в документах…")

        candidates = []

        # === Векторизация вопроса ТОЙ ЖЕ моделью, что база ===
        try:
            q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
        except Exception as e:
            st.error(f"Ошибка векторизации вопроса: {e}")
            q_emb = None

        where_filter = None
        if selected_sources:
            where_filter = {"source": {"$in": selected_sources}}

        # === Векторный поиск через query_embeddings ===
        if q_emb is not None:
            try:
                vector_results = collection.query(
                    query_embeddings=q_emb,
                    n_results=30,
                    where=where_filter,
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
            except Exception as e:
                st.warning(f"Ошибка векторного поиска: {e}")
                if where_filter:
                    try:
                        vector_results = collection.query(
                            query_embeddings=q_emb,
                            n_results=30,
                        )
                        if vector_results['documents'] and vector_results['documents'][0]:
                            for i, doc in enumerate(vector_results['documents'][0]):
                                meta = vector_results['metadatas'][0][i]
                                candidates.append({
                                    'text': doc,
                                    'source': meta['source'],
                                    'section': meta.get('section', ''),
                                    'tables': meta.get('tables', ''),
                                    'type': 'векторный (без фильтра)'
                                })
                    except Exception:
                        pass

        # === Поиск по упомянутым таблицам ===
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
                            if selected_sources and meta['source'] not in selected_sources:
                                continue
                            candidates.append({
                                'text': doc,
                                'source': meta['source'],
                                'section': meta.get('section', ''),
                                'tables': meta.get('tables', ''),
                                'type': f'таблица {table_num}'
                            })
                except Exception:
                    pass

        # === Дополнительный поиск по ключевым словам ===
        if re.search(r'допуск|отклонени', question, re.IGNORECASE) and q_emb is not None:
            try:
                kw_emb = embedder.encode(
                    ["допуск отклонение не более мм"],
                    normalize_embeddings=True
                ).tolist()
                keyword_results = collection.query(
                    query_embeddings=kw_emb,
                    n_results=20,
                    where=where_filter
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
            except Exception:
                pass

        # === Смягчённый фильтр: достаточно цифры ИЛИ ключевого слова ===
        filtered = []
        for c in candidates:
            text_lower = c['text'].lower()
            has_number = bool(re.search(r'\d+', c['text']))
            has_keyword = any(
                word in text_lower
                for word in ['допуск', 'отклонен', 'мм', 'таблиц', 'не более', 'отметк']
            )
            if has_number or has_keyword:
                filtered.append(c)

        if not filtered:
            filtered = candidates

        # === Уникализация ===
        seen = set()
        unique_filtered = []
        for c in filtered:
            key = c['text'][:200]
            if key not in seen:
                seen.add(key)
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

        status_placeholder.empty()

        st.session_state.current_question = question
        st.session_state.current_sources = sources
        st.session_state.current_fragments = unique_filtered

        st.markdown(
            f'<div class="fragments-info">📖 Отобрано фрагментов: {len(unique_filtered)}</div>',
            unsafe_allow_html=True
        )

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
            with st.spinner("🤖 ИИ формулирует ответ…"):
                response = client.chat.completions.create(
                    model="Qwen/Qwen3-30B-A3B",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    extra_body={"enable_thinking": False}
                )
            answer = response.choices[0].message.content
            st.session_state.current_answer = answer
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
            st.session_state.current_answer = None

# ==================== РЕЗУЛЬТАТ ====================
if st.session_state.current_answer:
    question = st.session_state.current_question
    answer = st.session_state.current_answer
    sources = st.session_state.current_sources
    unique_filtered = st.session_state.current_fragments

    st.success("✅ Ответ найден!")
    st.markdown("### 📖 Ответ:")
    st.markdown(answer)

    action_cols = st.columns([1, 1, 1, 2])

    with action_cols[0]:
        if IRONPRESS_OK:
            try:
                pdf_bytes = ironpress.markdown_to_pdf(
                    f"# {question}\n\n{answer}\n\n---\n\n## Источники\n\n" +
                    "\n".join(f"- {s}" for s in sources)
                )
                st.download_button(
                    "💾 Скачать PDF",
                    data=pdf_bytes,
                    file_name=f"snip_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="dl_pdf"
                )
            except Exception as e:
                st.caption(f"PDF недоступен: {e}")
        else:
            st.caption("PDF: установите ironpress")

    with action_cols[1]:
        with st.popover("📋 Копировать"):
            st.code(answer, language="markdown")

    with action_cols[2]:
        fb = st.feedback("thumbs", key=f"fb_{hash(question)}")
        if fb is not None:
            st.session_state.feedback[question] = fb
            if fb == 1:
                st.toast("👍 Спасибо за оценку!")
            else:
                st.toast("👎 Спасибо, мы учтём это")

    with st.expander(f"📚 Показать источники ({len(sources)})", expanded=False):
        for src in sources[:15]:
            st.markdown(f"• {src}")

    with st.expander(f"🔍 Показать фрагменты ({len(unique_filtered)})", expanded=False):
        for i, c in enumerate(unique_filtered, 1):
            ref = c['source']
            if c['section']:
                ref += f", раздел {c['section']}"
            if c['tables']:
                ref += f", таблица {c['tables']}"
            st.markdown(f"**Фрагмент {i}** · *{ref}* · тип: `{c['type']}`")
            st.markdown(f"> {c['text']}")
            st.markdown("---")