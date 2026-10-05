# app.py — RAG-поиск по строительным нормам (СП, СНиП, ГОСТ)
# Стек: Streamlit + ChromaDB + sentence-transformers + Cloud.ru (Qwen3-30B-A3B)
#
# Требования:
#   1. Папка chroma_db/ в корне репозитория (собирается локально: python db_builder.py)
#   2. Secrets (Streamlit → Settings → Secrets):
#        CLOUDRU_API_KEY  = "..."
#        CLOUDRU_BASE_URL = "https://foundation-models.api.cloud.ru/v1"
#        CLOUDRU_MODEL    = "Qwen/Qwen3-30B-A3B"
#
# Запуск: streamlit run app.py

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ------------------------------------------------------------------ config
DB_PATH = "chroma_db"
COLLECTION_NAME = "snip_norms"
EMBEDDING_MODEL = "sergeyzh/rubert-mini-frida"
DEFAULT_TOP_K = 6
MAX_CONTEXT_CHARS = 12000

st.set_page_config(
    page_title="Поиск по строительным нормам",
    page_icon="🏗️",
    layout="wide",
)

# ------------------------------------------------------- проверка базы
if not os.path.isdir(DB_PATH) or not os.listdir(DB_PATH):
    st.error(
        "❌ База `chroma_db` не найдена в репозитории.\n\n"
        "Соберите её локально: `python db_builder.py`, затем закоммитьте "
        "папку `chroma_db/` и запушьте в `main`."
    )
    st.stop()

# ------------------------------------------------------- кешированные ресурсы
@st.cache_resource(show_spinner="Загружаю модель эмбеддингов…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Открываю базу…")
def load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)


@st.cache_resource
def load_llm_client() -> OpenAI:
    api_key = st.secrets.get("CLOUDRU_API_KEY") or os.environ.get("CLOUDRU_API_KEY")
    base_url = (
        st.secrets.get("CLOUDRU_BASE_URL")
        or os.environ.get("CLOUDRU_BASE_URL")
        or "https://foundation-models.api.cloud.ru/v1"
    )
    if not api_key:
        st.error(
            "❌ Не задан `CLOUDRU_API_KEY`. "
            "Добавьте его в Streamlit → Settings → Secrets."
        )
        st.stop()
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model_name() -> str:
    return (
        st.secrets.get("CLOUDRU_MODEL")
        or os.environ.get("CLOUDRU_MODEL")
        or "Qwen/Qwen3-30B-A3B"
    )


# ------------------------------------------------------- утилиты
def format_source(meta: dict) -> str:
    """Собирает человекочитаемую ссылку на источник."""
    parts = [meta.get("source", "?")]
    if meta.get("chapter"):
        ch = meta["chapter"]
        title = meta.get("chapter_title", "")
        parts.append(f"глава {ch}" + (f" «{title}»" if title else ""))
    if meta.get("section"):
        parts.append(f"раздел {meta['section']}")
    if meta.get("point"):
        parts.append(f"п. {meta['point']}")
    if meta.get("table_number"):
        parts.append(f"таблица {meta['table_number']}")
    if meta.get("appendix"):
        parts.append(f"приложение {meta['appendix']}")
    return " → ".join(parts)


def build_context(hits: list) -> str:
    """Склеивает top-K чанков в один контекст, не превышая лимит."""
    blocks = []
    total = 0
    for i, (doc, meta, _dist) in enumerate(hits, 1):
        header = f"[{i}] {format_source(meta)}"
        block = f"{header}\n{doc}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block)
    return "\n---\n".join(blocks)


SYSTEM_PROMPT = (
    "Ты — инженер-эксперт по строительным нормам РФ (СП, СНиП, ГОСТ). "
    "Отвечай ТОЛЬКО на основе приведённых ниже фрагментов документов. "
    "В конце ответа обязательно перечисли использованные источники в формате "
    "«источник → глава → раздел → пункт/таблица». "
    "Если ответа в контексте нет — прямо скажи: "
    "«В предоставленных фрагментах ответа нет» и не выдумывай."
)


# ------------------------------------------------------- UI
st.title("🏗️ Поиск по строительным нормам")
st.caption(
    "СП, СНиП, ГОСТ. Ответ со ссылками на документ / главу / раздел / пункт / таблицу."
)

with st.sidebar:
    st.header("Настройки")
    top_k = st.slider("Сколько фрагментов искать", 3, 12, DEFAULT_TOP_K)
    show_context = st.checkbox("Показать найденный контекст", value=False)
    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Рендер прошлых сообщений
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Источники"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Ввод
query = st.chat_input("Спросите про норму, пункт, таблицу…")
if not query:
    st.stop()

st.session_state.messages.append({"role": "user", "content": query})
with st.chat_message("user"):
    st.markdown(query)

# Загрузка ресурсов (после UI, чтобы Streamlit показал спиннеры)
model = load_model()
collection = load_collection()
llm = load_llm_client()
model_name = get_model_name()

with st.chat_message("assistant"):
    with st.spinner("Ищу в нормах…"):
        q_emb = model.encode([query], normalize_embeddings=True).tolist()[0]
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

    hits = list(zip(docs, metas, dists))
    context = build_context(hits)
    sources = [format_source(m) for _, m, _ in hits]

    if show_context:
        with st.expander("Найденный контекст (top-K)", expanded=False):
            st.text(context)

    user_msg = (
        f"Вопрос: {query}\n\n"
        f"Фрагменты нормативных документов:\n{context}\n\n"
        f"Дай ответ и перечисли источники."
    )

    with st.spinner("Формулирую ответ…"):
        try:
            resp = llm.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ Ошибка обращения к LLM: `{e}`"

    st.markdown(answer)
    if sources:
        with st.expander("Источники"):
            for s in sources:
                st.markdown(f"- {s}")

st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "sources": sources,
})