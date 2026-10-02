import streamlit as st
import chromadb
from openai import OpenAI
import re
import os
import time
from datetime import datetime
from sentence_transformers import CrossEncoder
from db_builder import build_database, DB_PATH, COLLECTION_NAME

try:
    import ironpress
    IRONPRESS_OK = True
except ImportError:
    IRONPRESS_OK = False


st.set_page_config(
    page_title="Поиск по СНиПам",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="auto"
)


# ==================== ЛОГИРОВАНИЕ ОЦЕНОК ====================

FEEDBACK_WEBHOOK_URL = ""


def log_feedback(question, answer, rating, fragments_count=0):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": (answer or "")[:500],
        "rating": int(rating),
        "fragments_count": fragments_count,
    }
    print(f"[FEEDBACK] {payload}")

    if FEEDBACK_WEBHOOK_URL:
        try:
            import urllib.request
            import json as _json
            req = urllib.request.Request(
                FEEDBACK_WEBHOOK_URL,
                data=_json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[FEEDBACK] webhook error: {e}")


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

    .top-clock {
        position: fixed;
        top: 12px;
        left: 20px;
        z-index: 999999;
        background: var(--secondary-background-color);
        color: var(--text-color);
        padding: 10px 18px;
        border-radius: 10px;
        font-family: 'SF Mono', 'Consolas', 'Menlo', monospace;
        box-shadow: 0 2px 10px rgba(0,0,0,0.18);
        border: 1px solid rgba(128,128,128,0.2);
        pointer-events: none;
        white-space: nowrap;
        display: flex;
        flex-direction: column;
        align-items: center;
        line-height: 1.2;
    }
    .top-clock .clock-date {
        font-size: 0.95rem;
        font-weight: 600;
        opacity: 0.85;
        color: var(--text-color);
    }
    .top-clock .clock-time {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--primary-color);
        letter-spacing: 1px;
    }

    @media (max-width: 768px) {
        .main-header { font-size: 1.3rem !important; line-height: 1.2; margin-bottom: 0.3rem; }
        .main-subheader { font-size: 0.85rem; margin-bottom: 0.75rem; line-height: 1.3; }
        .block-container { padding-top: 4.5rem !important; padding-bottom: 3rem !important; padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
        [data-testid="stSidebar"] { min-width: 0 !important; max-width: 100% !important; }
        [data-testid="stSidebar"] .doc-card { font-size: 0.75rem; padding: 0.4rem 0.6rem; }
        .stButton > button, .stFormSubmitButton > button { font-size: 0.9rem !important; padding: 0.6rem 0.8rem !important; min-height: 2.6rem; }
        .stFormSubmitButton > button { font-size: 1rem !important; padding: 0.75rem 1.5rem !important; min-height: 3rem !important; }
        .stTextInput > div > div > input { font-size: 1rem !important; padding: 0.75rem !important; min-height: 2.75rem; }
        .fragments-info { font-size: 0.85rem; padding: 0.6rem 0.8rem; margin: 0.75rem 0; }
        .stExpander { margin-bottom: 0.5rem !important; }
        .stExpander summary { font-size: 0.9rem !important; }
        .db-status { font-size: 0.8rem; padding: 0.4rem 0.6rem; }

        .top-clock {
            top: 6px;
            left: 8px;
            padding: 6px 10px;
            border-radius: 8px;
        }
        .top-clock .clock-date { font-size: 0.7rem; }
        .top-clock .clock-time { font-size: 1rem; letter-spacing: 0.5px; }
    }
</style>
""", unsafe_allow_html=True)


# ✅ Часы в левом верхнем углу
import streamlit.components.v1 as components

components.html("""
<script>
(function() {
    const parentDoc = window.parent.document;

    const old = parentDoc.querySelector('.top-clock');
    if (old) old.remove();

    const clock = parentDoc.createElement('div');
    clock.className = 'top-clock';
    clock.innerHTML = '<span class="clock-date"></span><span class="clock-time"></span>';
    parentDoc.body.appendChild(clock);

    const dateEl = clock.querySelector('.clock-date');
    const timeEl = clock.querySelector('.clock-time');

    function pad(n) { return n < 10 ? '0' + n : '' + n; }

    function tick() {
        const d = new Date();
        const dateStr = pad(d.getDate()) + '.' + pad(d.getMonth() + 1) + '.' + d.getFullYear();
        const timeStr = pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
        dateEl.textContent = dateStr;
        timeEl.textContent = timeStr;
    }

    tick();
    setInterval(tick, 1000);
})();
</script>
""", height=0)


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


# ✅ Этап 3: reranker
@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-base", max_length=512)


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


# ==================== RERANKER ====================

def rerank_candidates(question, candidates, top_k=10):
    """Прогоняет кандидатов через cross-encoder, возвращает топ-K."""
    if not candidates:
        return []

    try:
        reranker = load_reranker()
    except Exception as e:
        print(f"[RERANK] не удалось загрузить reranker: {e}")
        return candidates[:top_k]

    # Ограничиваем длину текста — экономия памяти и скорости
    pairs = [(question, (c.get("text") or "")[:1500]) for c in candidates]

    try:
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    except Exception as e:
        print(f"[RERANK] ошибка predict: {e}")
        return candidates[:top_k]

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


# ==================== ПОИСК ====================

def gather_candidates(question, selected_sources, limit_vector=40):
    """Собирает кандидатов из векторного поиска и поиска по таблицам."""
    candidates = []

    where_filter = None
    if selected_sources:
        where_filter = {"source": {"$in": selected_sources}}

    # 1) Векторный поиск
    try:
        vector_results = collection.query(
            query_texts=[question],
            n_results=limit_vector,
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

    # 2) Явное упоминание таблицы в вопросе
    table_matches = re.findall(
        r'таблиц[аы]?\s*([А-ЯA-Z]?\.?\d+(?:\.\d+)?)',
        question, re.IGNORECASE
    )
    for table_num in table_matches:
        try:
            r = collection.get(
                where={"table_number": table_num},
                limit=10
            )
            if r['documents']:
                for i, doc in enumerate(r['documents']):
                    meta = r['metadatas'][i]
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
                        'is_table': meta.get('is_table', True),
                        'table_number': meta.get('table_number', ''),
                        'type': f'таблица {table_num}'
                    })
        except Exception:
            pass

    # Дедупликация по тексту
    seen = set()
    unique = []
    for c in candidates:
        key = c['text'][:500]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def search_and_answer(question, selected_sources):
    # 1. Собрать кандидатов
    candidates = gather_candidates(question, selected_sources)

    if not candidates:
        return (
            "В найденных фрагментах нет полного ответа по этому вопросу.",
            [],
            []
        )

    # 2. Reranker: топ-50 → топ-10
    pre_top = candidates[:50]
    top = rerank_candidates(question, pre_top, top_k=10)

    # 3. Формируем контекст
    context_parts = []
    sources_set = []
    for c in top:
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

        chunk_text = c['text'][:1500]
        ref = " · ".join(ref_parts)
        context_parts.append(f"\n\n--- Источник: {ref} ---\n{chunk_text}")
        if ref not in sources_set:
            sources_set.append(ref)

    context = "".join(context_parts)

    prompt = f"""Ты — эксперт по строительным нормам и правилам (СП, СНиП, ГОСТ).

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе фрагментов ниже. Не выдумывай.
2. КРИТИЧНО: Если во фрагментах НЕТ информации, напрямую отвечающей
   на вопрос — ОБЯЗАТЕЛЬНО скажи:
   «В найденных фрагментах нет полного ответа по теме "<вопрос>".
    Найдены только косвенные упоминания.»
   НЕ пересказывай нерелевантные фрагменты как ответ.
3. Отвечай структурированно, по пунктам.
4. ВАЖНО: Отвечай МАКСИМУМ 7 пунктами. Выбери ТОЛЬКО самые
   релевантные фрагменты. Не перечисляй всё подряд.
5. Не дублируй один и тот же фрагмент дважды.
6. Если фрагмент — это строка таблицы, а не раздел документа — НЕ оформляй
   его как «Раздел».
7. КРИТИЧНО: Если фрагмент содержит ТОЛЬКО общую фразу без конкретных
   чисел, допусков (±, мм, %) или названий — НЕ цитируй его вообще.
   Примеры мусора: «Основные положения, допуски, отклонения и посадки»,
   «Технические требования. Контроль. Способ контроля» — такие фразы
   НЕ являются ответом.

ОСОБОЕ ВНИМАНИЕ (если вопрос про допуски/отклонения):
- Ищи ВСЕ виды допусков, а не только первый попавшийся:
  * допуски на высотные отметки
  * допуски на ширину (покрытия, слоя, конструкции)
  * допуски на уклоны (продольные, поперечные)
  * допуски на ровность
  * допуски на толщину слоёв
  * допуски на прямолинейность
- В таблицах обычно перечислены ВСЕ допуски — проверь их.

ФОРМАТ ОТВЕТА (для каждого требования):
- **Документ:** полное название
- **Раздел:** номер и название
- **Подраздел:** номер и название (если есть)
- **Пункт:** номер
- **Таблица:** номер (если есть)
- **Текст требования:** точная цитата или близкий пересказ

ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}

ВОПРОС:
{question}

ОТВЕТ:"""

    max_retries = 3
    response = None
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-30B-A3B",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                extra_body={"enable_thinking": False}
            )
            break
        except Exception as e:
            last_error = e
            err_str = str(e)
            if '429' in err_str or 'TooManyRequests' in err_str or 'rate limit' in err_str:
                wait_time = 15 * (attempt + 1)
                time.sleep(wait_time)
            else:
                raise
    if response is None and last_error:
        raise last_error

    answer = response.choices[0].message.content
    return answer, sources_set, top


# ==================== СЕССИЯ ====================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "input_version" not in st.session_state:
    st.session_state.input_version = 0


# ==================== САЙДБАР ====================

with st.sidebar:
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
        st.session_state.pending_question = ""
        st.session_state.input_version += 1
        st.rerun()

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕐 История")
        st.caption("Нажми — вопрос вставится в поле ввода")
        for i, q in enumerate(reversed(st.session_state.history[-10:])):
            if st.button(f"↻ {q[:50]}", key=f"hist_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.session_state.input_version += 1
                st.rerun()

    if st.session_state.feedback:
        st.markdown("---")
        st.markdown("### 📊 Оценки")
        ups = sum(1 for v in st.session_state.feedback.values() if v == 1)
        downs = sum(1 for v in st.session_state.feedback.values() if v == 0)
        total = ups + downs

        st.markdown(
            f'<div style="font-size:1.2rem; margin-bottom:0.4rem;">'
            f'<span style="color:#22c55e; font-weight:700;">👍 {ups}</span>'
            f'<span style="color:#888;"> &nbsp;·&nbsp; </span>'
            f'<span style="color:#ef4444; font-weight:700;">👎 {downs}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        if total > 0:
            quality = round(ups / total * 100)
            if quality >= 70:
                q_color = "#22c55e"
            elif quality >= 40:
                q_color = "#f59e0b"
            else:
                q_color = "#ef4444"
            st.markdown(
                f'<div style="font-size:0.85rem; opacity:0.8;">'
                f'Качество: <span style="color:{q_color}; font-weight:700;">{quality}%</span> '
                f'из {total} оценок'
                f'</div>',
                unsafe_allow_html=True
            )

        if st.button("♻️ Сбросить оценки", key="reset_feedback_btn", use_container_width=True):
            st.session_state.feedback = {}
            st.rerun()


# ==================== ЗАГОЛОВОК ====================

st.markdown('<h1 class="main-header" style="margin-top: 4.5rem;">📐 Поиск по СНиПам</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subheader">Задайте вопрос — программа найдёт ответ в СП, СНиП и ГОСТ с указанием источника.</p>', unsafe_allow_html=True)


# ==================== ВВОД ВОПРОСА ====================

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


# ==================== ОБРАБОТКА ВОПРОСА ====================

user_input = None
if ask_clicked and user_input_text.strip():
    user_input = user_input_text.strip()

if user_input:
    st.session_state.messages = []

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

    with st.spinner("⏳ Ищу ответ в документах…"):
        try:
            answer, sources, fragments = search_and_answer(user_input, selected_sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "question": user_input,
                "sources": sources,
                "fragments": fragments
            })

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

    st.session_state.input_version += 1
    st.rerun()


# ==================== ЧАТ ====================

chat_container = st.container()

with chat_container:
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="📐"):
                st.markdown(msg["content"])

                q = msg.get("question", "")
                sources = msg.get("sources", [])
                fragments = msg.get("fragments", [])

                action_cols = st.columns([1, 1, 1, 1, 2])

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
                    current_fb = st.session_state.feedback.get(q)
                    up_label = "👍" if current_fb != 1 else "✅👍"
                    if st.button(
                        up_label,
                        key=f"fb_up_{idx}_{hash(q)}",
                        use_container_width=True,
                        help="Ответ полезен",
                    ):
                        st.session_state.feedback[q] = 1
                        log_feedback(q, msg["content"], 1, len(fragments))
                        st.toast("👍 Спасибо! Учли.")
                        st.rerun()

                with action_cols[3]:
                    down_label = "👎" if current_fb != 0 else "✅👎"
                    if st.button(
                        down_label,
                        key=f"fb_down_{idx}_{hash(q)}",
                        use_container_width=True,
                        help="Ответ неточен",
                    ):
                        st.session_state.feedback[q] = 0
                        log_feedback(q, msg["content"], 0, len(fragments))
                        st.toast("👎 Учтём.")
                        st.rerun()

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