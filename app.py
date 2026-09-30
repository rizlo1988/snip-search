import streamlit as st
import chromadb
from openai import OpenAI
import re
import os
from datetime import datetime
from db_builder import build_database, DB_PATH, COLLECTION_NAME

try:
    import ironpress
    IRONPRESS_OK = True
except ImportError:
    IRONPRESS_OK = False

st.set_page_config(
    page_title="Поиск по СНиПам",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    .stApp { background: var(--background-color); }
    .main-header { color: var(--primary-color); font-size: 1.7rem; font-weight: 700; margin-bottom: 0.3rem; line-height: 1.2; }
    .main-subheader { color: var(--text-color); opacity: 0.7; font-size: 1rem; margin-bottom: 1rem; }
    .sidebar-header { color: var(--primary-color); font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem; }
    .stButton > button, .stFormSubmitButton > button { border-radius: 8px; font-weight: 600; transition: all 0.3s; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { transform: translateY(-2px); }
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] { padding-left: 0 !important; padding-right: 0 !important; justify-content: center !important; }
    .stTextInput > div > div > input { border-radius: 8px; padding: 0.75rem; font-size: 1rem; }
    [data-testid="stSidebar"] { min-width: 260px !important; max-width: 300px !important; }
    .doc-card {
        background: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; border-radius: 6px;
        font-size: 0.8rem; line-height: 1.3;
    }
    .fragments-info {
        background: var(--secondary-background-color); border-radius: 8px;
        padding: 0.75rem 1rem; margin: 1rem 0;
        color: var(--primary-color); font-weight: 600;
    }
    .source-ref { font-size: 0.85rem; color: var(--primary-color); margin-bottom: 0.4rem; font-weight: 600; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .db-status {
        background: var(--secondary-background-color);
        border-left: 4px solid #22c55e;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 0.75rem;
    }
    [data-testid="stForm"] { border: none; padding: 0; }
    @media (max-width: 768px) {
        .main-header { font-size: 1.3rem !important; line-height: 1.2; margin-bottom: 0.3rem; }
        .main-subheader { font-size: 0.85rem; margin-bottom: 0.75rem; line-height: 1.3; }
        .block-container { padding-top: 0.75rem !important; padding-bottom: 3rem !important; padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
        [data-testid="stSidebar"] { min-width: 0 !important; max-width: 100% !important; }
        [data-testid="stSidebar"] .doc-card { font-size: 0.75rem; padding: 0.4rem 0.6rem; }
        .stButton > button, .stFormSubmitButton > button { font-size: 0.9rem !important; padding: 0.6rem 0.8rem !important; min-height: 2.6rem; }
        .stFormSubmitButton > button { font-size: 1rem !important; padding: 0.75rem 1.5rem !important; min-height: 3rem !important; }
        .stTextInput > div > div > input { font-size: 1rem !important; padding: 0.75rem !important; min-height: 2.75rem; }
        .fragments-info { font-size: 0.85rem; padding: 0.6rem 0.8rem; margin: 0.75rem 0; }
        .stExpander { margin-bottom: 0.5rem !important; }
        .stExpander summary { font-size: 0.9rem !important; }
        .db-status { font-size: 0.8rem; padding: 0.4rem 0.6rem; }
    }
</style>
""", unsafe_allow_html=True)


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
            progress_placeholder.empty()

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


TRASH_CHAPTERS = {'1', '2'}
TRASH_TITLE_PATTERNS = [
    re.compile(r'Нормативные ссылки', re.IGNORECASE),
    re.compile(r'Область применения', re.IGNORECASE),
]


def is_trash_fragment(c, is_definition_question=False):
    ch = c.get('chapter', '')
    ch_title = c.get('chapter_title', '')

    if not is_definition_question and ch == '3':
        return True

    if ch in TRASH_CHAPTERS:
        return True

    for p in TRASH_TITLE_PATTERNS:
        if p.search(ch_title):
            return True

    if ch.isdigit() and int(ch) > 30:
        return True

    return False


def search_and_answer(question, selected_sources):
    candidates = []

    where_filter = None
    if selected_sources:
        where_filter = {"source": {"$in": selected_sources}}

    is_definition_question = bool(re.search(
        r'что такое|определени|термин|называется',
        question, re.IGNORECASE
    ))

    try:
        vector_results = collection.query(
            query_texts=[question],
            n_results=40,
            where=where_filter
        )
        if vector_results['documents'] and vector_results['documents'][0]:
            for i, doc in enumerate(vector_results['documents'][0]):
                meta = vector_results['metadatas'][0][i]
                candidates.append({
                    'text': doc,
                    'source': meta.get('source', ''),
                    'chapter': meta.get('chapter', ''),
                    'chapter_title': meta.get('chapter_title', ''),
                    'section': meta.get('section', ''),
                    'section_title': meta.get('section_title', ''),
                    'point': meta.get('point', ''),
                    'is_table': meta.get('is_table', False),
                    'table_number': meta.get('table_number', ''),
                    'type': 'векторный'
                })
    except Exception:
        pass

    table_matches = re.findall(
        r'таблиц[аы]?\s*([А-ЯA-Z]?\.?\d+(?:\.\d+)?)',
        question, re.IGNORECASE
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
                        if selected_sources and meta.get('source') not in selected_sources:
                            continue
                        candidates.append({
                            'text': doc,
                            'source': meta.get('source', ''),
                            'chapter': meta.get('chapter', ''),
                            'chapter_title': meta.get('chapter_title', ''),
                            'section': meta.get('section', ''),
                            'section_title': meta.get('section_title', ''),
                            'point': meta.get('point', ''),
                            'is_table': meta.get('is_table', False),
                            'table_number': meta.get('table_number', ''),
                            'type': f'таблица {table_num}'
                        })
            except Exception:
                pass

    if re.search(r'допуск|отклонени|отметк|ширин|уклон', question, re.IGNORECASE):
        try:
            table_query = collection.get(
                where_document={"$contains": "Таблица"},
                limit=50
            )
            if table_query['documents']:
                for i, doc in enumerate(table_query['documents']):
                    if not re.search(r'допуск|отклонени|отметк|ширин|уклон', doc, re.IGNORECASE):
                        continue
                    meta = table_query['metadatas'][i]
                    if selected_sources and meta.get('source') not in selected_sources:
                        continue
                    candidates.append({
                        'text': doc,
                        'source': meta.get('source', ''),
                        'chapter': meta.get('chapter', ''),
                        'chapter_title': meta.get('chapter_title', ''),
                        'section': meta.get('section', ''),
                        'section_title': meta.get('section_title', ''),
                        'point': meta.get('point', ''),
                        'is_table': True,
                        'table_number': meta.get('table_number', ''),
                        'type': 'таблица допусков'
                    })
        except Exception:
            pass

    seen = set()
    unique_candidates = []
    for c in candidates:
        key = c['text'][:200]
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    filtered = [c for c in unique_candidates if not is_trash_fragment(c, is_definition_question)]

    if not filtered:
        filtered = unique_candidates

    filtered.sort(key=lambda c: (
        0 if c.get('is_table') else 1,
        1 if c.get('chapter') not in ('', '3', '1', '2') else 0,
        len(c['text'])
    ), reverse=True)

    unique_filtered = filtered[:25]

    context_parts = []
    sources_set = []
    for c in unique_filtered:
        ref_parts = [c['source'].replace('.txt', '')]
        if c.get('chapter'):
            ch_title = c.get('chapter_title', '')
            if ch_title:
                ref_parts.append(f"раздел {c['chapter']} «{ch_title}»")
            else:
                ref_parts.append(f"раздел {c['chapter']}")
        if c.get('section'):
            sec_title = c.get('section_title', '')
            if sec_title:
                ref_parts.append(f"подраздел {c['section']} «{sec_title}»")
            else:
                ref_parts.append(f"подраздел {c['section']}")
        if c.get('point'):
            ref_parts.append(f"пункт {c['point']}")
        if c.get('table_number'):
            ref_parts.append(f"Таблица {c['table_number']}")

        ref = " · ".join(ref_parts)
        context_parts.append(f"\n\n--- Источник: {ref} ---\n{c['text']}")
        if ref not in sources_set:
            sources_set.append(ref)

    context = "".join(context_parts)

    prompt = f"""Ты — эксперт по строительным нормам и правилам (СП, СНиП, ГОСТ).

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе фрагментов ниже. Не выдумывай.
2. Если во фрагментах нет ответа — честно скажи: «В найденных фрагментах нет полного ответа».
3. Отвечай структурированно, по пунктам.

ОСОБОЕ ВНИМАНИЕ (если вопрос про допуски/отклонения):
- Ищи ВСЕ виды допусков, а не только первый попавшийся:
  * допуски на высотные отметки
  * допуски на ширину (покрытия, слоя, конструкции)
  * допуски на уклоны (продольные, поперечные)
  * допуски на ровность
  * допуски на толщину слоёв
  * допуски на прямолинейность
- В таблицах обычно перечислены ВСЕ допуски — проверь их.
- Если в найденных фрагментах нет какого-то вида допусков — честно скажи,
  каких именно допусков нет.

ФОРМАТ ОТВЕТА (для каждого требования):
- **Документ:** полное название (например, «СП 78.13330.2012 Автомобильные дороги»)
- **Раздел:** номер и название (например, «8 Дорожные одежды»)
- **Подраздел:** номер и название (если есть)
- **Пункт:** номер (например, 8.10)
- **Таблица:** номер (если есть)
- **Текст требования:** точная цитата или близкий пересказ

ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}

ВОПРОС:
{question}

ОТВЕТ:"""

    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        extra_body={"enable_thinking": False}
    )

    answer = response.choices[0].message.content
    return answer, sources_set, unique_filtered


# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
# ✅ Счётчик для смены key поля ввода (чтобы очищать без rerun)
if "input_version" not in st.session_state:
    st.session_state.input_version = 0


# ==================== САЙДБАР ====================
with st.sidebar:
    st.markdown(
        f'<div class="db-status">✅ База собрана · {len(sources_list)} документов</div>',
        unsafe_allow_html=True
    )

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
        icon = "📘" if "ГОСТ" in src else "📗"
        clean_name = src.replace(".txt", "")
        if len(clean_name) > 40:
            truncated = clean_name[:40]
            if ' ' in truncated:
                truncated = truncated.rsplit(' ', 1)[0]
            display_name = truncated + "..."
        else:
            display_name = clean_name
        st.markdown(
            f'<div class="doc-card">{icon} {display_name}</div>',
            unsafe_allow_html=True
        )

    st.markdown("")
    if st.button("🗑️ Очистить историю", key="clear_history_btn", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.feedback = {}
        st.session_state.input_version += 1
        st.rerun()

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕐 История")
        st.caption("Нажми — вопрос вставится в поле ввода")
        for i, q in enumerate(reversed(st.session_state.history[-10:])):
            if st.button(f"↻ {q[:50]}", key=f"hist_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    if st.session_state.feedback:
        st.markdown("---")
        st.markdown("### 📊 Оценки")
        ups = sum(1 for v in st.session_state.feedback.values() if v == 1)
        downs = sum(1 for v in st.session_state.feedback.values() if v == 0)
        st.markdown(f"👍 **{ups}** · 👎 **{downs}**")


# ==================== ЗАГОЛОВОК + ФОРМА ====================
st.markdown('<h1 class="main-header">🏗️ Поиск по СНиПам</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subheader">Задайте вопрос — программа найдёт ответ в СП, СНиП и ГОСТ с указанием источника.</p>', unsafe_allow_html=True)

# ✅ Ключ поля зависит от input_version — при инкременте поле очищается
input_key = f"question_input_{st.session_state.input_version}"
prefill_value = st.session_state.pending_question if st.session_state.pending_question else ""
st.session_state.pending_question = ""

with st.form("question_form", clear_on_submit=False):
    input_cols = st.columns([5, 1])
    with input_cols[0]:
        user_input_text = st.text_input(
            "Вопрос",
            value=prefill_value,
            placeholder="Задайте вопрос по строительным нормам...",
            label_visibility="collapsed",
            key=input_key
        )
    with input_cols[1]:
        ask_clicked = st.form_submit_button("🔍", use_container_width=True, type="primary")

user_input = None
if ask_clicked and user_input_text.strip():
    user_input = user_input_text.strip()


# ==================== ЧАТ ====================
chat_container = st.container()

if user_input:
    if re.search(r'допуск|отклонени', user_input, re.IGNORECASE) and len(user_input.split()) < 4:
        st.info(
            "💡 Уточните: допуски на что?\n\n"
            "Например:\n"
            "- «допуски на высотные отметки»\n"
            "- «допуски на ширину покрытия»\n"
            "- «допуски на поперечный уклон»\n"
            "- «допуски на ровность»"
        )

    st.session_state.messages.append({"role": "user", "content": user_input})
    if user_input not in st.session_state.history:
        st.session_state.history.append(user_input)

    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🏗️"):
            with st.spinner("⏳ Ищу ответ в документах…"):
                try:
                    answer, sources, fragments = search_and_answer(user_input, selected_sources)
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "question": user_input,
                        "sources": sources,
                        "fragments": fragments
                    })

                    # ✅ Меняем версию ключа → поле очистится БЕЗ rerun
                    st.session_state.input_version += 1

                except Exception as e:
                    error_msg = f"Произошла ошибка: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "question": user_input,
                        "sources": [],
                        "fragments": []
                    })

with chat_container:
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🏗️"):
                st.markdown(msg["content"])

                q = msg.get("question", "")
                sources = msg.get("sources", [])
                fragments = msg.get("fragments", [])

                action_cols = st.columns([1, 1, 1, 2])

                with action_cols[0]:
                    if IRONPRESS_OK:
                        try:
                            pdf_bytes = ironpress.markdown_to_pdf(
                                f"# {q}\n\n{msg['content']}\n\n---\n\n## Источники\n\n" +
                                "\n".join(f"- {s}" for s in sources)
                            )
                            st.download_button(
                                "💾 PDF", data=pdf_bytes,
                                file_name=f"snip_answer_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf", key=f"dl_pdf_{idx}"
                            )
                        except Exception as e:
                            st.caption(f"PDF: {e}")

                with action_cols[1]:
                    with st.popover("📋 Копировать", use_container_width=True):
                        st.code(msg["content"], language="markdown")

                with action_cols[2]:
                    fb = st.feedback("thumbs", key=f"fb_{idx}")
                    if fb is not None:
                        st.session_state.feedback[q] = fb
                        if fb == 1:
                            st.toast("👍 Спасибо!")
                        else:
                            st.toast("👎 Учтём")

                if sources:
                    with st.expander(f"📚 Источники ({len(sources)})", expanded=False):
                        for s in sources[:15]:
                            st.markdown(f"• {s}")

                if fragments:
                    with st.expander(f"🔍 Фрагменты ({len(fragments)})", expanded=False):
                        for i, c in enumerate(fragments, 1):
                            ref_parts = [c.get('source', '').replace('.txt', '')]
                            if c.get('chapter'):
                                ch_title = c.get('chapter_title', '')
                                if ch_title:
                                    ref_parts.append(f"раздел {c['chapter']} «{ch_title}»")
                                else:
                                    ref_parts.append(f"раздел {c['chapter']}")
                            if c.get('section'):
                                sec_title = c.get('section_title', '')
                                if sec_title:
                                    ref_parts.append(f"подраздел {c['section']} «{sec_title}»")
                                else:
                                    ref_parts.append(f"подраздел {c['section']}")
                            if c.get('point'):
                                ref_parts.append(f"пункт {c['point']}")
                            if c.get('table_number'):
                                ref_parts.append(f"Таблица {c['table_number']}")
                            ref = " · ".join(ref_parts)

                            st.markdown(f"**Фрагмент {i}** · тип: `{c.get('type', '')}`")
                            st.markdown(f'<div class="source-ref">📄 {ref}</div>', unsafe_allow_html=True)
                            st.markdown(f"> {c['text'][:1500]}")
                            st.markdown("---")