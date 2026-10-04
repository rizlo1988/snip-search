# db_builder.py — ФИНАЛЬНАЯ версия (согласована с app.py)
# Фиксы A (pending_table_refs), B (обрыв таблиц по ^N Название),
# C (не наследовать chapter в табличных чанках),
# D (поддержка СП 317 и др.),
# E (совместимость с app.py: build_database, DB_PATH, progress_callback,
#    chapter_title / section_title / is_table в метаданных),
# F (ужесточённый RE_CHAPTER, слабый is_real_chapter_start,
#    отключение распознавания раздела/пункта внутри таблиц).

import os
import re
import hashlib
from typing import List, Dict, Optional, Callable

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sergeyzh/rubert-mini-frida"
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 1500
MERGE_MIN_SIZE = 300

# ---- имена, которые ждёт app.py ----
DB_PATH = "chroma_db"
COLLECTION_NAME = "snip_norms"
# ------------------------------------

VALID_CHAPTER_RANGES = {
    "ГОСТ Р 51872-2024": (1, 5),
    "СП 126.13330.2017": (1, 10),
    "СП 34.13330.2021": (1, 7),
    "СП 46.13330.2012": (1, 14),
    "СП 70.13330.2012": (1, 18),
    "СП 78.13330.2012": (1, 7),
    "СП 317.1325800.2017": (1, 8),
}

# Раздел: "5 Состав инженерно-геодезических изысканий. Общие технические требования"
# ВАЖНО: после номера — слово с заглавной буквы, затем строчная.
# Это отсекает предисловия "1 РАЗРАБОТАН", "2 ИСПОЛНИТЕЛИ",
# "3 ПОДГОТОВЛЕН", "4 УТВЕРЖДЕН", "5 ЗАРЕГИСТРИРОВАН" и т.п.
RE_CHAPTER = re.compile(r"^(\d{1,2})\s+([А-ЯЁ][а-яё][^\n]{1,})$")

# Пункт: до 4 уровней вложенности ("5.3.1.4", "5.7.1.13")
RE_POINT = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,4})\s+(.*)$")

RE_SUBITEM = re.compile(r"^[а-яё]\)\s|^\d\)\s")

# Таблица: "Таблица 5.1" или "Таблица 5.1 - Основные требования..."
RE_TABLE = re.compile(r"^Таблица\s+(\d+(?:\.\d+)?)\s*(?:-.*)?$")

RE_APPENDIX = re.compile(r"^Приложение\s+([А-ЯЁ])\s*$")

# Маркеры конца приложения
RE_APPENDIX_END = re.compile(r"^(Библиография|УДК\s)")


def _strip_title(rest: str) -> str:
    """Заголовок раздела/подраздела без номера, в одну строку, без завершающих точек."""
    return rest.strip().rstrip(".").strip()


def is_real_chapter_start(lines: List[str], idx: int) -> bool:
    """После ужесточения RE_CHAPTER (первые две буквы — Заглавная+строчная)
    дополнительная проверка по следующей строке не нужна.
    Оставлена для совместимости — всегда True."""
    return True


def extract_chapter_number(line: str) -> Optional[int]:
    m = RE_CHAPTER.match(line.strip())
    return int(m.group(1)) if m else None


def extract_table_number(line: str) -> Optional[str]:
    m = RE_TABLE.match(line.strip())
    return m.group(1) if m else None


def make_chunk_id(source: str, idx: int, text: str) -> str:
    h = hashlib.md5(f"{source}|{idx}|{text[:80]}".encode("utf-8")).hexdigest()
    return f"{source}_{idx}_{h[:8]}"


def parse_document(path: str) -> List[Dict]:
    source = os.path.basename(path).replace(".txt", "").strip()
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().split("\n")]

    pending_table_refs: Dict[str, Dict[str, str]] = {}

    chunks: List[Dict] = []
    cur_chapter: Optional[int] = None
    cur_chapter_title: str = ""
    cur_section: str = ""
    cur_section_title: str = ""
    cur_point: str = ""
    cur_table: str = ""
    cur_appendix: str = ""
    in_table: bool = False
    in_appendix_section: bool = False

    buffer: List[str] = []
    buffer_meta: Dict[str, str] = {}

    def flush_buffer():
        nonlocal buffer, buffer_meta
        text = "\n".join(buffer).strip()
        if len(text) >= MIN_CHUNK_SIZE:
            chunks.append({
                "text": text[:MAX_CHUNK_SIZE],
                "source": source,
                "chapter": str(buffer_meta.get("chapter", "")),
                "chapter_title": buffer_meta.get("chapter_title", ""),
                "section": buffer_meta.get("section", ""),
                "section_title": buffer_meta.get("section_title", ""),
                "point": buffer_meta.get("point", ""),
                "table_number": buffer_meta.get("table_number", ""),
                "is_table": "1" if buffer_meta.get("table_number", "") else "0",
                "appendix": buffer_meta.get("appendix", ""),
            })
        buffer = []
        buffer_meta = {}

    def start_new_buffer(meta: Dict[str, str]):
        nonlocal buffer, buffer_meta
        flush_buffer()
        buffer_meta = dict(meta)

    def default_meta():
        return {
            "chapter": str(cur_chapter or ""),
            "chapter_title": cur_chapter_title,
            "section": cur_section,
            "section_title": cur_section_title,
            "point": cur_point,
            "table_number": "",
            "appendix": cur_appendix,
        }

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Конец приложения — по "Библиография" или "УДК"
        if in_appendix_section and RE_APPENDIX_END.match(stripped):
            in_appendix_section = False

        if stripped == "":
            if buffer:
                buffer.append("")
            i += 1
            continue

        # Приложение
        m_app = RE_APPENDIX.match(stripped)
        if m_app:
            cur_appendix = m_app.group(1)
            cur_table = ""
            in_table = False
            in_appendix_section = True
            start_new_buffer(default_meta())
            buffer.append(stripped)
            i += 1
            continue

        # Внутри приложения НЕ распознаём пункты и разделы
        if in_appendix_section:
            if not buffer:
                start_new_buffer(default_meta())
            buffer.append(stripped)
            i += 1
            continue

        # Таблица N (или "Таблица N - Заголовок")
        m_tbl = RE_TABLE.match(stripped)
        if m_tbl:
            tbl_num = m_tbl.group(1)
            cur_table = tbl_num
            in_table = True

            if tbl_num in pending_table_refs:
                ref = pending_table_refs[tbl_num]
                meta = {
                    "chapter": ref.get("chapter", str(cur_chapter or "")),
                    "chapter_title": ref.get("chapter_title", cur_chapter_title),
                    "section": ref.get("section", cur_section),
                    "section_title": ref.get("section_title", cur_section_title),
                    "point": ref.get("point", cur_point),
                    "table_number": tbl_num,
                    "appendix": cur_appendix,
                }
            else:
                meta = default_meta()
                meta["table_number"] = tbl_num

            start_new_buffer(meta)
            buffer.append(stripped)
            i += 1
            continue

        # ВНУТРИ ТАБЛИЦЫ — не распознаём RE_POINT / RE_CHAPTER.
        # Копим всё в текущий буфер таблицы.
        if in_table:
            # Признак конца таблицы: пустая строка ПЕРЕД этим местом +
            # текущая строка НЕ похожа на строку таблицы (нет цифр и спецзнаков).
            # Упрощённо: если после пустой строки идёт строка, матчащая RE_POINT,
            # начинается новый пункт → таблица закончилась.
            prev_is_blank = (i > 0 and lines[i - 1].strip() == "")
            m_pt = RE_POINT.match(stripped)
            if prev_is_blank and m_pt and not RE_SUBITEM.match(stripped):
                # конец таблицы: сбрасываем флаг и обрабатываем как пункт
                in_table = False
                cur_table = ""
                # (не continue — проваливаемся в блок обработки пункта ниже)
            else:
                if not buffer:
                    start_new_buffer(default_meta())
                buffer.append(stripped)
                i += 1
                continue

        # Пункт N.M[.K[.L]]
        m_pt = RE_POINT.match(stripped)
        if m_pt and not RE_SUBITEM.match(stripped):
            point_num = m_pt.group(1)
            parts = point_num.split(".")
            cur_section = ".".join(parts[:2]) if len(parts) >= 2 else ""
            cur_point = point_num

            start_new_buffer({
                "chapter": str(cur_chapter or ""),
                "chapter_title": cur_chapter_title,
                "section": cur_section,
                "section_title": cur_section_title,
                "point": cur_point,
                "table_number": "",
                "appendix": cur_appendix,
            })
            buffer.append(stripped)
            i += 1
            continue

        # Раздел ^N Название
        m_ch = RE_CHAPTER.match(stripped)
        if m_ch:
            ch_num = int(m_ch.group(1))

            valid_range = VALID_CHAPTER_RANGES.get(source)
            if valid_range and not (valid_range[0] <= ch_num <= valid_range[1]):
                buffer.append(stripped)
                i += 1
                continue

            # Настоящий раздел
            cur_chapter = ch_num
            cur_chapter_title = _strip_title(m_ch.group(2))
            cur_section = ""
            cur_section_title = ""
            cur_point = ""
            cur_table = ""
            in_table = False

            start_new_buffer({
                "chapter": str(cur_chapter),
                "chapter_title": cur_chapter_title,
                "section": "",
                "section_title": "",
                "point": "",
                "table_number": "",
                "appendix": cur_appendix,
            })
            buffer.append(stripped)
            i += 1
            continue

        # Ссылка на таблицу в тексте
        ref_match = re.search(
            r"таблиц[аеыо][й]?\s+(\d+(?:\.\d+)?)", stripped, re.IGNORECASE
        )
        if ref_match and not in_table:
            ref_num = ref_match.group(1)
            pending_table_refs[ref_num] = {
                "chapter": str(cur_chapter or ""),
                "chapter_title": cur_chapter_title,
                "section": cur_section,
                "section_title": cur_section_title,
                "point": cur_point,
            }

        if not buffer:
            start_new_buffer(default_meta())
        buffer.append(stripped)
        i += 1

    flush_buffer()
    return chunks


def merge_short_chunks(chunks: List[Dict]) -> List[Dict]:
    if not chunks:
        return chunks
    merged: List[Dict] = []
    cur = dict(chunks[0])
    for nxt in chunks[1:]:
        same_ctx = (
            cur["source"] == nxt["source"]
            and cur["chapter"] == nxt["chapter"]
            and cur["section"] == nxt["section"]
            and cur["appendix"] == nxt["appendix"]
        )
        cur_short = len(cur["text"]) < MERGE_MIN_SIZE
        nxt_short = len(nxt["text"]) < MERGE_MIN_SIZE
        if same_ctx and (cur_short or nxt_short):
            combined = cur["text"] + "\n" + nxt["text"]
            if len(combined) <= MAX_CHUNK_SIZE:
                cur["text"] = combined
                if not cur["point"] and nxt["point"]:
                    cur["point"] = nxt["point"]
                if not cur["table_number"] and nxt["table_number"]:
                    cur["table_number"] = nxt["table_number"]
                    cur["is_table"] = nxt.get("is_table", "0")
                if not cur.get("chapter_title") and nxt.get("chapter_title"):
                    cur["chapter_title"] = nxt["chapter_title"]
                if not cur.get("section_title") and nxt.get("section_title"):
                    cur["section_title"] = nxt["section_title"]
                continue
        merged.append(cur)
        cur = dict(nxt)
    merged.append(cur)
    return merged


def build_database(
    documents_dir: str = "documents",
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """Собирает ChromaDB-коллекцию из .txt в documents_dir.

    Совместимо с app.py: принимает kwarg progress_callback(msg).
    """
    def report(msg: str):
        print(msg)
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception as e:
                print(f"[progress_callback] error: {e}")

    report(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: List[Dict] = []
    for fname in sorted(os.listdir(documents_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(documents_dir, fname)
        report(f"Parsing {fname} ...")
        doc_chunks = parse_document(path)
        doc_chunks = merge_short_chunks(doc_chunks)
        report(f"  -> {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    report(f"Total chunks: {len(all_chunks)}")
    BATCH = 64
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start:start + BATCH]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        ids = [make_chunk_id(c["source"], start + j, c["text"])
               for j, c in enumerate(batch)]
        metadatas = [{
            "source": c["source"],
            "chapter": c["chapter"],
            "chapter_title": c.get("chapter_title", ""),
            "section": c["section"],
            "section_title": c.get("section_title", ""),
            "point": c["point"],
            "table_number": c["table_number"],
            "is_table": c.get("is_table", "0"),
            "appendix": c["appendix"],
        } for c in batch]
        collection.add(
            ids=ids, documents=texts,
            embeddings=embeddings, metadatas=metadatas,
        )
        report(f"  indexed {start + len(batch)}/{len(all_chunks)}")
    report("Done.")


# Обратная совместимость (если где-то остался старый вызов build_db)
build_db = build_database


if __name__ == "__main__":
    build_database()