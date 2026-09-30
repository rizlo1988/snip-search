import os
import re
import chromadb
from chromadb.utils import embedding_functions

DOCS_FOLDER = "documents"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "snip_docs"

# Модель эмбеддингов (B: ru-en-RoSBERTa)
EMBEDDING_MODEL = "ai-forever/ru-en-RoSBERTa"

# Максимальный размер чанка (C: гибрид)
MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 100


# ==================== ПАРСИНГ СТРУКТУРЫ ДОКУМЕНТА ====================

def parse_document(text, filename):
    """
    Универсальный парсер СП и ГОСТ с сохранением структуры.
    Возвращает список чанков с метаданными.
    """
    chunks = []

    # Извлекаем тип и номер документа из имени файла
    doc_type = "ГОСТ" if "ГОСТ" in filename.upper() else "СП"
    doc_number_match = re.search(r'(\d+(?:\.\d+)*)', filename)
    doc_number = doc_number_match.group(1) if doc_number_match else ""

    lines = text.split('\n')
    total_lines = len(lines)

    # Регулярные выражения для структуры
    # Раздел: "1 Область применения", "12 Устройство асфальтобетонных покрытий"
    chapter_re = re.compile(r'^(\d{1,2})\s+([А-ЯЁ][А-Яа-яЁё\s,\-\.\(\)]{3,80})$')
    # Подраздел: "7.1 Общие положения", "12.3 Укладка асфальтобетонных смесей"
    section_re = re.compile(r'^(\d{1,2}\.\d{1,2})\s+([А-ЯЁ][А-Яа-яЁё\s,\-\.\(\)]{3,80})$')
    # Пункт: "7.1.1 ...", "12.2.4 ...", "8.28 ..."
    point_re = re.compile(r'^(\d{1,2}(?:\.\d{1,2}){1,3})\s+')
    # Таблица: "Таблица 1", "Таблица 9", "Таблица 5.1", "Таблица 11а"
    table_re = re.compile(r'^Таблица\s+([А-ЯA-Z]?\.?\d+(?:\.\d+)?[а-яa-z]?)')
    # Приложение: "Приложение А", "Приложение Б (обязательное)"
    appendix_re = re.compile(r'^Приложение\s+([А-ЯA-Z])')

    # Текущее состояние парсинга
    current_chapter = ""
    current_chapter_title = ""
    current_section = ""
    current_section_title = ""
    current_point = ""
    current_buffer = []
    current_start_line = 0

    def flush_buffer(end_line, is_table=False, table_num=""):
        """Сохраняет накопленный буфер как чанк."""
        nonlocal current_buffer, current_start_line
        if not current_buffer:
            return
        chunk_text = '\n'.join(current_buffer).strip()
        if len(chunk_text) < MIN_CHUNK_SIZE:
            current_buffer = []
            current_start_line = end_line
            return

        # Если чанк слишком большой — режем на части
        if len(chunk_text) > MAX_CHUNK_SIZE:
            for sub_chunk in split_large_chunk(chunk_text, MAX_CHUNK_SIZE):
                add_chunk(sub_chunk, is_table, table_num)
        else:
            add_chunk(chunk_text, is_table, table_num)

        current_buffer = []
        current_start_line = end_line

    def add_chunk(chunk_text, is_table=False, table_num=""):
        """Добавляет чанк в список."""
        chunks.append({
            "text": chunk_text,
            "source": filename,
            "doc_type": doc_type,
            "doc_number": doc_number,
            "chapter": current_chapter,
            "chapter_title": current_chapter_title,
            "section": current_section,
            "section_title": current_section_title,
            "point": current_point,
            "is_table": is_table,
            "table_number": table_num,
        })

    i = 0
    while i < total_lines:
        line = lines[i]
        stripped = line.strip()

        # Проверяем — не исключён ли пункт
        is_excluded = bool(re.search(r'\(Исключен[а]?,?\s', stripped) or
                          re.search(r'\(Исключен[а]?,?\s', stripped))

        # 1. Новая таблица?
        table_match = table_re.match(stripped)
        if table_match:
            # Сохраняем накопленное
            flush_buffer(i)
            # Собираем таблицу целиком
            table_lines = [line]
            table_num = table_match.group(1)
            j = i + 1
            empty_count = 0
            while j < total_lines:
                next_line = lines[j]
                next_stripped = next_line.strip()
                # Таблица заканчивается на пустой строке или заголовке
                if not next_stripped:
                    empty_count += 1
                    if empty_count >= 2:
                        break
                    table_lines.append(next_line)
                elif chapter_re.match(next_stripped) or \
                     section_re.match(next_stripped) or \
                     table_re.match(next_stripped):
                    break
                else:
                    empty_count = 0
                    table_lines.append(next_line)
                j += 1

            table_text = '\n'.join(table_lines).strip()
            add_chunk(table_text, is_table=True, table_num=table_num)
            i = j
            current_start_line = i
            continue

        # 2. Новый раздел?
        chapter_match = chapter_re.match(stripped)
        if chapter_match:
            flush_buffer(i)
            current_chapter = chapter_match.group(1)
            current_chapter_title = chapter_match.group(2).strip()
            current_section = ""
            current_section_title = ""
            current_point = ""
            i += 1
            continue

        # 3. Новый подраздел?
        section_match = section_re.match(stripped)
        if section_match:
            flush_buffer(i)
            current_section = section_match.group(1)
            current_section_title = section_match.group(2).strip()
            current_point = ""
            i += 1
            continue

        # 4. Новый пункт?
        point_match = point_re.match(stripped)
        if point_match and not is_excluded:
            flush_buffer(i)
            current_point = point_match.group(1)
            current_buffer.append(line)
            i += 1
            continue

        # 5. Приложение?
        appendix_match = appendix_re.match(stripped)
        if appendix_match:
            flush_buffer(i)
            current_chapter = f"Приложение {appendix_match.group(1)}"
            current_chapter_title = stripped
            current_section = ""
            current_section_title = ""
            current_point = ""
            i += 1
            continue

        # 6. Продолжение текущего пункта
        if current_buffer or current_point:
            current_buffer.append(line)
        elif stripped:
            # Начало документа без структуры (введение, предисловие)
            if not current_buffer:
                current_start_line = i
            current_buffer.append(line)

        i += 1

    # Сохраняем последний буфер
    flush_buffer(total_lines)

    return chunks


def split_large_chunk(text, max_size):
    """Режет большой чанк на части по абзацам."""
    parts = []
    paragraphs = text.split('\n\n')
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_size:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                parts.append(current)
            if len(p) > max_size:
                # Режем абзац по предложениям
                sentences = re.split(r'(?<=[.!?])\s+', p)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) + 1 <= max_size:
                        sub += (" " if sub else "") + s
                    else:
                        if sub:
                            parts.append(sub)
                        sub = s
                if sub:
                    parts.append(sub)
            else:
                current = p
    if current:
        parts.append(current)
    return parts


# ==================== ПОСТРОЕНИЕ БАЗЫ ====================

def build_database(progress_callback=None):
    client = chromadb.PersistentClient(path=DB_PATH)

    # Новая модель эмбеддингов
    ru_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection(name=COLLECTION_NAME)
        if progress_callback:
            progress_callback("Старая база удалена. Создаю новую...")
    except Exception:
        if progress_callback:
            progress_callback("Создаю новую базу...")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ru_ef
    )

    if not os.path.exists(DOCS_FOLDER):
        raise FileNotFoundError(f"Папка '{DOCS_FOLDER}' не найдена!")

    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"В папке '{DOCS_FOLDER}' нет .txt файлов!")

    if progress_callback:
        progress_callback(f"Найдено документов: {len(files)}")

    total_chunks = 0
    for filename in files:
        filepath = os.path.join(DOCS_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = parse_document(content, filename)

        if progress_callback:
            progress_callback(f"  {filename}: разбит на {len(chunks)} чанков")

        # Батчами по 100 для скорости
        BATCH_SIZE = 100
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start:batch_start + BATCH_SIZE]
            documents = [c["text"] for c in batch]
            metadatas = [{
                "source": c["source"],
                "doc_type": c["doc_type"],
                "doc_number": c["doc_number"],
                "chapter": c["chapter"],
                "chapter_title": c["chapter_title"],
                "section": c["section"],
                "section_title": c["section_title"],
                "point": c["point"],
                "is_table": c["is_table"],
                "table_number": c["table_number"],
            } for c in batch]
            ids = [f"{filename}_{batch_start + i}" for i in range(len(batch))]

            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            total_chunks += len(batch)

    if progress_callback:
        progress_callback(f"✅ База данных создана! Всего чанков: {total_chunks}")

    return collection