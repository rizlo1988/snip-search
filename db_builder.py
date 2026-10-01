import os
import re
import chromadb
from chromadb.utils import embedding_functions

DOCS_FOLDER = "documents"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "snip_docs"

EMBEDDING_MODEL = "sergeyzh/rubert-tiny-turbo"

MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 100


# ==================== ЧИСТКА КОЛОНТИТУЛОВ ====================

HEADER_PATTERNS = [
    re.compile(r'СП\s+\d+\.\d+\.\d+.*Актуализированная редакция'),
    re.compile(r'Свод правил от \d+\.\d+\.\d+ N \d+'),
    re.compile(r'^Страница \d+$'),
    re.compile(r'Внимание! Документ включен'),
    re.compile(r'ИС «Кодекс: \d поколение»'),
    re.compile(r'Внимание! О порядке применения документа'),
    re.compile(r'Внимание! Дополнительную информацию см\.'),
    re.compile(r'Документ предоставлен КонсультантПлюс'),
    re.compile(r'^\s*КонсультантПлюс:'),
    re.compile(r'^СП \d+\.\d+\.\d+\.\d+$'),
    re.compile(r'^ГОСТ Р \d+-\d+$'),
    re.compile(r'^ГОСТ \d+-\d+$'),
    re.compile(r'^Применяется с \d+\.\d+\.\d+'),
    re.compile(r'^\d+$'),
    re.compile(r'^ГОСТ Р 51872-2024'),
    re.compile(r'^СП 34\.13330\.2021'),
    re.compile(r'^СП 70\.13330\.2012 Несущие'),
    re.compile(r'^СП 126\.13330\.2017'),
    re.compile(r'ИС «Техэксперт'),
    re.compile(r'^КонсультантПлюс: примечание'),
    re.compile(r'^\s*СП \d+\.\d+\.\d+\.\d+\.\d+$'),
]


def is_header_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in HEADER_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def clean_text(text):
    lines = text.split('\n')
    cleaned = [line for line in lines if not is_header_line(line)]
    return '\n'.join(cleaned)


# ==================== КОНТЕКСТ ТАБЛИЦЫ ====================

def looks_like_table_context(lines, i, window=6):
    start = max(0, i - window)
    end = min(len(lines), i + window)
    context = ' '.join(lines[start:end])

    number_lines = 0
    total_nonempty = 0
    for j in range(start, end):
        stripped = lines[j].strip()
        if not stripped:
            continue
        total_nonempty += 1
        if re.search(r'\d', stripped):
            number_lines += 1

    if total_nonempty == 0:
        return False

    number_ratio = number_lines / total_nonempty
    has_units = bool(re.search(r'\b(мм|см|кг|м|м/с|м/сут|‰|%|МПа|см²|м²|м³)\b', context))
    has_columns = bool(re.search(r'\s{3,}', context))

    return number_ratio > 0.5 or (has_units and has_columns)


# ==================== ШАПКА ТАБЛИЦЫ ====================

# ✅ ФИКС: шапки таблиц, которые не несут данных
TABLE_HEADER_PATTERNS = [
    re.compile(r'^Технические требования\s+Контроль\s+Способ контроля', re.IGNORECASE),
    re.compile(r'^Технические требования\s+Контроль\s+Метод', re.IGNORECASE),
    re.compile(r'^Допускаемые отклонения\s+Контроль\s+Способ', re.IGNORECASE),
    re.compile(r'^Наименование\s+Контроль', re.IGNORECASE),
    re.compile(r'^\s*Технические требования\s*$', re.IGNORECASE),
    re.compile(r'^\s*Контроль\s+Способ\s+контроля\s*$', re.IGNORECASE),
    re.compile(r'^\s*Контроль\s+Метод или способ\s*$', re.IGNORECASE),
    re.compile(r'^\s*Значения технических требований', re.IGNORECASE),
]


def is_table_header_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in TABLE_HEADER_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


# ==================== ПРОВЕРКА "НАСТОЯЩИЙ ЛИ ЭТО ЗАГОЛОВОК" ====================

TRASH_TITLE_STARTS = (
    'Отклонение', 'Разность', 'Измерительный', 'То же',
    'Допускаемые', 'Предельные', 'Наименьшие', 'Наибольшие',
    'Не более', 'Не менее', 'Св.', 'Св ', 'Примечание',
    'Значения', 'Величина', 'Параметр', 'Показатель',
    'Первая', 'Вторая', 'Третья', 'Первый', 'Второй', 'Третий',
    'До ', 'От ', 'Свыше', 'Менее', 'Более',
)

BAD_CHAPTER_TITLE_PREFIXES = (
    'Допускаемое соединение',
    'Допускаемые соединения',
    'Устройство асфальтобетонного покрытия',
    'Инъецирование закрытых каналов',
    'Допускаемые отклонения',
    'Допускаемые значения',
    'Предельные отклонения',
    'Предельные значения',
    'Нормальные прохождения',
    'Нормальное прохождение',
    'Операции по выпуску',
    'Операцию по выпуску',
    'Технические требования',
    'Наименование отклонения',
    'Номинальный размер',
)


def is_plausible_chapter_title(title):
    title = title.strip()

    if len(title) < 5 or len(title) > 120:
        return False

    for trash in TRASH_TITLE_STARTS:
        if title.startswith(trash):
            return False

    for prefix in BAD_CHAPTER_TITLE_PREFIXES:
        if title.startswith(prefix):
            return False

    if re.match(r'^\d+\s*(мм|см|м|кг|%|‰|МПа|м/с)', title):
        return False

    if not re.match(r'^[А-ЯЁ]', title):
        return False

    digits = sum(c.isdigit() for c in title)
    if digits > len(title) * 0.3:
        return False

    letters = sum(c.isalpha() for c in title)
    if letters < len(title) * 0.5:
        return False

    return True


def is_plausible_section_title(title):
    return is_plausible_chapter_title(title)


# ==================== СПЕЦИФИЧНЫЕ МУСОРНЫЕ РАЗДЕЛЫ ДЛЯ ДОКУМЕНТОВ ====================

BAD_CHAPTERS_BY_DOC = {
    'НЕСУЩИЕ И ОГРАЖДАЮЩИЕ': {
        '4', '5', '6', '7', '8', '13', '14',
        '20', '26', '30', '35', '37',
    },
    'МОСТЫ И ТРУБЫ': {
        '4', '6', '9', '13', '20', '26', '30', '35', '37',
    },
}


def is_bad_chapter_for_doc(chapter_num, title, source):
    title_stripped = title.strip()
    for doc_key, bad_chapters in BAD_CHAPTERS_BY_DOC.items():
        if doc_key in source and chapter_num in bad_chapters:
            for prefix in BAD_CHAPTER_TITLE_PREFIXES:
                if title_stripped.startswith(prefix):
                    return True
            if re.search(r'\bмм\b', title_stripped) and len(title_stripped) < 80:
                return True
    return False


# ==================== ПАРСИНГ ====================

def parse_document(text, filename):
    chunks = []

    doc_type = "ГОСТ" if "ГОСТ" in filename.upper() else "СП"
    doc_number_match = re.search(r'(\d+(?:\.\d+)*)', filename)
    doc_number = doc_number_match.group(1) if doc_number_match else ""

    lines = text.split('\n')
    total_lines = len(lines)

    chapter_full_re = re.compile(r'^(\d{1,2})\s+([А-ЯЁ][А-Яа-яЁё\s,\-\.\(\)]{4,100})$')
    chapter_number_only_re = re.compile(r'^(\d{1,2})$')
    section_re = re.compile(r'^(\d{1,2}\.\d{1,2})\s+([А-ЯЁ][А-Яа-яЁё\s,\-\.\(\)]{4,100})$')
    point_re = re.compile(r'^(\d{1,2}(?:\.\d{1,2}){1,3})[\s\.]+')
    table_re = re.compile(r'^Таблица\s+([А-ЯA-Z]?\.?\d+(?:\.\d+)?[а-яa-z]?)')
    appendix_re = re.compile(r'^Приложение\s+([А-ЯA-Z])')
    title_re = re.compile(r'^[А-ЯЁ][А-Яа-яЁё\s,\-\.\(\)]{4,100}$')

    current_chapter = ""
    current_chapter_title = ""
    current_section = ""
    current_section_title = ""
    current_point = ""
    current_buffer = []
    current_start_line = 0
    in_appendix = False

    def flush_buffer(end_line, is_table=False, table_num=""):
        nonlocal current_buffer, current_start_line
        if not current_buffer:
            return
        chunk_text = '\n'.join(current_buffer).strip()
        if len(chunk_text) < MIN_CHUNK_SIZE:
            current_buffer = []
            current_start_line = end_line
            return

        if len(chunk_text) > MAX_CHUNK_SIZE:
            for sub_chunk in split_large_chunk(chunk_text, MAX_CHUNK_SIZE):
                add_chunk(sub_chunk, is_table, table_num)
        else:
            add_chunk(chunk_text, is_table, table_num)

        current_buffer = []
        current_start_line = end_line

    def add_chunk(chunk_text, is_table=False, table_num=""):
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

        is_excluded = bool(re.search(r'\(Исключен[а]?,?\s', stripped))
        in_table_context = looks_like_table_context(lines, i)

        # 1. Новая таблица
        table_match = table_re.match(stripped)
        if table_match:
            flush_buffer(i)
            table_lines = [line]
            table_num = table_match.group(1)
            # ✅ Сохраняем контекст раздела/пункта для таблицы
            saved_chapter = current_chapter
            saved_chapter_title = current_chapter_title
            saved_section = current_section
            saved_section_title = current_section_title
            saved_point = current_point

            j = i + 1
            empty_count = 0
            header_skipped = 0  # ✅ сколько шапок пропущено
            while j < total_lines:
                next_line = lines[j]
                next_stripped = next_line.strip()

                # Новая таблица — стоп
                if table_re.match(next_stripped):
                    break

                # ✅ ФИКС: НЕ прерываем по chapter_full_re — таблица
                # не должна обрываться на строке, похожей на заголовок.
                # Прерываем только по section_re / point_re, которые
                # гарантированно начинают новый пункт ДОКУМЕНТА, а не
                # строку таблицы.
                # Проверяем только если следующая строка явно похожа на
                # заголовок раздела И не содержит разделителей-колонок.

                if (re.match(r'^\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\s+[А-ЯЁ]', next_stripped)
                        and not re.search(r'\s{3,}', next_stripped)  # не таблица
                        and not in_appendix):
                    break

                if (chapter_full_re.match(next_stripped)
                        and not re.search(r'\s{3,}', next_stripped)  # не таблица
                        and not in_appendix):
                    break

                if not next_stripped:
                    empty_count += 1
                    if empty_count >= 5:  # ✅ ФИКС: было 3
                        break
                    table_lines.append(next_line)
                else:
                    empty_count = 0
                    # ✅ ФИКС: пропускаем шапки таблиц
                    if is_table_header_line(next_stripped):
                        header_skipped += 1
                        # Не добавляем в table_lines — это шапка
                        j += 1
                        continue
                    table_lines.append(next_line)
                    # ✅ ФИКС: лимит поднят с 200 до 400
                    if len(table_lines) > 400:
                        break
                j += 1

            table_text = '\n'.join(table_lines).strip()
            # Восстанавливаем контекст
            current_chapter = saved_chapter
            current_chapter_title = saved_chapter_title
            current_section = saved_section
            current_section_title = saved_section_title
            current_point = saved_point
            add_chunk(table_text, is_table=True, table_num=table_num)
            i = j
            current_start_line = i
            continue

        # 2. Двухстрочный раздел
        chapter_num_match = chapter_number_only_re.match(stripped)
        if chapter_num_match and not in_table_context and not in_appendix:
            j = i + 1
            while j < total_lines and not lines[j].strip():
                j += 1
            if j < total_lines:
                next_stripped = lines[j].strip()
                if (title_re.match(next_stripped)
                        and is_plausible_chapter_title(next_stripped)
                        and not is_bad_chapter_for_doc(
                            chapter_num_match.group(1), next_stripped, filename)):
                    flush_buffer(i)
                    current_chapter = chapter_num_match.group(1)
                    current_chapter_title = next_stripped
                    current_section = ""
                    current_section_title = ""
                    current_point = ""
                    i = j + 1
                    continue

        # 3. Однострочный раздел
        chapter_match = chapter_full_re.match(stripped)
        if chapter_match and not in_table_context and not in_appendix:
            title = chapter_match.group(2).strip()
            if (is_plausible_chapter_title(title)
                    and not is_bad_chapter_for_doc(
                        chapter_match.group(1), title, filename)):
                flush_buffer(i)
                current_chapter = chapter_match.group(1)
                current_chapter_title = title
                current_section = ""
                current_section_title = ""
                current_point = ""
                i += 1
                continue

        # 4. Подраздел
        section_match = section_re.match(stripped)
        if section_match and not in_table_context and not in_appendix:
            title = section_match.group(2).strip()
            if is_plausible_section_title(title):
                section_num = section_match.group(1)
                section_chapter = section_num.split('.')[0]
                if current_chapter != section_chapter:
                    current_chapter = section_chapter
                    current_chapter_title = ""
                flush_buffer(i)
                current_section = section_num
                current_section_title = title
                current_point = ""
                i += 1
                continue

        # 5. Пункт
        point_match = point_re.match(stripped)
        if point_match and not is_excluded and not in_table_context and not in_appendix:
            new_point = point_match.group(1)
            point_chapter = new_point.split('.')[0]
            # ✅ Защита от фантомных пунктов
            if current_chapter and current_chapter.isdigit():
                if point_chapter != current_chapter:
                    if current_buffer or current_point:
                        current_buffer.append(line)
                    else:
                        current_buffer.append(line)
                    i += 1
                    continue
            flush_buffer(i)
            current_point = new_point
            current_buffer.append(line)
            i += 1
            continue

        # 6. Приложение
        appendix_match = appendix_re.match(stripped)
        if appendix_match and not in_table_context:
            flush_buffer(i)
            current_chapter = f"Приложение {appendix_match.group(1)}"
            current_chapter_title = stripped
            current_section = ""
            current_section_title = ""
            current_point = ""
            in_appendix = True
            i += 1
            continue

        if in_appendix and chapter_full_re.match(stripped) and not in_table_context:
            in_appendix = False

        # 7. Продолжение
        if current_buffer or current_point:
            current_buffer.append(line)
        elif stripped:
            if not current_buffer:
                current_start_line = i
            current_buffer.append(line)

        i += 1

    flush_buffer(total_lines)
    return chunks


def split_large_chunk(text, max_size):
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


# ==================== СБОРКА ====================

def build_database(progress_callback=None):
    client = chromadb.PersistentClient(path=DB_PATH)

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

        content = clean_text(content)
        chunks = parse_document(content, filename)

        if progress_callback:
            progress_callback(f"  {filename}: разбит на {len(chunks)} чанков")

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