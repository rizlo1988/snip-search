import os
import re
import chromadb
from chromadb.utils import embedding_functions


DOCS_FOLDER = "documents"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "snip_docs"

EMBEDDING_MODEL = "sergeyzh/rubert-mini-frida"

MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 200


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
    # ✅ Фикс СП 126: служебные скобки после заголовков
    re.compile(r'^\((?:раздел|пункт|таблица|приложение)\s+\d', re.IGNORECASE),
    re.compile(r'^\(в ред\.', re.IGNORECASE),
    re.compile(r'^\(введен', re.IGNORECASE),
    re.compile(r'^\(Измененная редакция', re.IGNORECASE),
    re.compile(r'^\(Изменение', re.IGNORECASE),
    re.compile(r'^\(Введен', re.IGNORECASE),
    re.compile(r'^\(Исключен', re.IGNORECASE),
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


# ==================== БЕЛЫЙ СПИСОК ГЛАВ ====================

VALID_CHAPTER_RANGES = {
    "ГОСТ Р 51872-2024": (1, 5),
    "СП 126.13330.2017": (1, 20),
    "СП 34.13330.2021": (1, 18),
    "СП 46.13330.2012": (1, 14),
    "СП 70.13330.2012": (1, 22),
    "СП 78.13330.2012": (1, 16),
}


def get_valid_chapter_range(filename):
    for key, rng in VALID_CHAPTER_RANGES.items():
        if key in filename:
            return rng
    return (1, 30)


def is_valid_chapter_num(chapter_num, filename):
    try:
        n = int(chapter_num)
    except (ValueError, TypeError):
        return False
    lo, hi = get_valid_chapter_range(filename)
    return lo <= n <= hi


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
    has_units = bool(re.search(
        r'\b(мм|см|кг|м|м/с|м/сут|‰|%|МПа|см²|м²|м³)\b', context
    ))
    has_columns = bool(re.search(r'\s{3,}', context))

    return number_ratio > 0.5 or (has_units and has_columns)


def is_inside_table_block(lines, i, window=5):
    """✅ Фикс A: надёжная проверка «мы внутри таблицы».
    Смотрим 5 строк ДО позиции i. Если ≥3 строки содержат единицы измерения
    или колонки (3+ пробела) — считаем, что мы внутри таблицы."""
    start = max(0, i - window)
    table_like = 0
    for j in range(start, i):
        s = lines[j].strip()
        if not s:
            continue
        if (re.search(r'\b(мм|см|кг|м|м/с|м/сут|‰|%|МПа|см²|м²|м³)\b', s)
                or re.search(r'\s{3,}', s)):
            table_like += 1
    return table_like >= 3


# ==================== ШАПКИ ТАБЛИЦ ====================

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


# ==================== ПРОВЕРКА ЗАГОЛОВКОВ ====================

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


# ==================== ПАРСИНГ ====================

def parse_document(text, filename):
    chunks = []

    doc_type = "ГОСТ" if "ГОСТ" in filename.upper() else "СП"
    doc_number_match = re.search(r'(\d+(?:\.\d+)*)', filename)
    doc_number = doc_number_match.group(1) if doc_number_match else ""

    lines = text.split('\n')
    total_lines = len(lines)

    chapter_full_re = re.compile(
        r'^(\d{1,2})\s+([А-ЯЁ][А-Яа-яЁё\s,\-\.\(\)]{4,100})$'
    )
    chapter_number_only_re = re.compile(r'^(\d{1,2})$')
    section_re = re.compile(
        r'^(\d{1,2}\.\d{1,2})\s+([А-ЯЁ][А-Яа-яЁё\s,\-\.\(\)]{4,100})$'
    )
    point_re = re.compile(r'^(\d{1,2}(?:\.\d{1,2}){1,3})[\s\.]+')
    table_re = re.compile(r'^Таблица\s+([А-ЯA-Z]?\.?\d+(?:\.\d+)?[а-яa-z]?)')
    appendix_re = re.compile(r'^Приложение\s+([А-ЯA-Z])')
    title_re = re.compile(r'^[А-ЯЁ][А-Яа-яЁё\s,\-\.\(\)]{4,100}$')

    # ✅ Фикс B: ссылки на таблицы из текста
    table_ref_re = re.compile(
        r'(?:приведен[ыо]?\s+в\s+таблиц[аеы]|см\.\s*таблиц|по\s+таблиц|'
        r'в\s+таблиц[аеы])\s+([А-ЯA-Z]?\.?\d+(?:\.\d+)?[а-яa-z]?)',
        re.IGNORECASE
    )
    pending_table_refs = {}  # {table_num: (chapter, chapter_title, section, section_title, point)}

    current_chapter = ""
    current_chapter_title = ""
    current_section = ""
    current_section_title = ""
    current_point = ""
    current_buffer = []
    in_appendix = False

    pending_small = []
    pending_small_meta = None

    def make_chunk(chunk_text, is_table=False, table_num="",
                   override_ctx=None):
        if override_ctx:
            ch, ch_t, sec, sec_t, pt = override_ctx
        else:
            ch = current_chapter
            ch_t = current_chapter_title
            sec = current_section
            sec_t = current_section_title
            pt = current_point
        return {
            "text": chunk_text,
            "source": filename,
            "doc_type": doc_type,
            "doc_number": doc_number,
            "chapter": ch,
            "chapter_title": ch_t,
            "section": sec,
            "section_title": sec_t,
            "point": pt,
            "is_table": is_table,
            "table_number": table_num,
        }

    def add_chunk(chunk_dict):
        nonlocal pending_small, pending_small_meta
        text = chunk_dict["text"].strip()
        if not text:
            return
        if (len(text) < MIN_CHUNK_SIZE
                and not chunk_dict["is_table"]
                and not chunk_dict["table_number"]):
            if pending_small_meta is None:
                pending_small_meta = {
                    k: chunk_dict[k] for k in (
                        "source", "doc_type", "doc_number",
                        "chapter", "chapter_title",
                        "section", "section_title",
                        "point",
                    )
                }
            pending_small.append(text)
            joined = "\n".join(pending_small)
            if len(joined) >= MIN_CHUNK_SIZE:
                merged = dict(pending_small_meta)
                merged["text"] = joined
                merged["is_table"] = False
                merged["table_number"] = ""
                chunks.append(merged)
                pending_small = []
                pending_small_meta = None
            return

        if pending_small:
            joined = "\n".join(pending_small)
            merged = dict(pending_small_meta)
            merged["text"] = joined
            merged["is_table"] = False
            merged["table_number"] = ""
            chunks.append(merged)
            pending_small = []
            pending_small_meta = None

        chunks.append(chunk_dict)

    def flush_small_pending():
        nonlocal pending_small, pending_small_meta
        if pending_small:
            joined = "\n".join(pending_small)
            if joined.strip():
                merged = dict(pending_small_meta or {
                    "source": filename, "doc_type": doc_type,
                    "doc_number": doc_number,
                    "chapter": "", "chapter_title": "",
                    "section": "", "section_title": "",
                    "point": "",
                })
                merged["text"] = joined
                merged["is_table"] = False
                merged["table_number"] = ""
                chunks.append(merged)
            pending_small = []
            pending_small_meta = None

    def flush_buffer(is_table=False, table_num=""):
        nonlocal current_buffer
        if not current_buffer:
            return
        chunk_text = '\n'.join(current_buffer).strip()
        if not chunk_text:
            current_buffer = []
            return
        if len(chunk_text) > MAX_CHUNK_SIZE:
            for sub in split_large_chunk(chunk_text, MAX_CHUNK_SIZE):
                add_chunk(make_chunk(sub, is_table, table_num))
        else:
            add_chunk(make_chunk(chunk_text, is_table, table_num))
        current_buffer = []

    i = 0
    while i < total_lines:
        line = lines[i]
        stripped = line.strip()

        is_excluded = bool(re.search(r'\(Исключен[а]?,?\s', stripped))
        in_table_context = looks_like_table_context(lines, i)
        inside_table = is_inside_table_block(lines, i)

        # ---------- 0. Сбор ссылок на таблицы ----------
        m_ref = table_ref_re.search(stripped)
        if m_ref and current_point:
            ref_num = m_ref.group(1)
            pending_table_refs[ref_num] = (
                current_chapter, current_chapter_title,
                current_section, current_section_title,
                current_point,
            )

        # ---------- 1. Таблица ----------
        table_match = table_re.match(stripped)
        if table_match:
            flush_buffer()
            table_lines = [line]
            table_num = table_match.group(1)

            # ✅ Фикс B: приоритет — контекст пункта, который сослался на таблицу
            override_ctx = None
            if table_num in pending_table_refs:
                override_ctx = pending_table_refs.pop(table_num)

            saved_chapter = current_chapter
            saved_chapter_title = current_chapter_title
            saved_section = current_section
            saved_section_title = current_section_title
            saved_point = current_point

            j = i + 1
            empty_count = 0
            while j < total_lines:
                next_line = lines[j]
                next_stripped = next_line.strip()

                if table_re.match(next_stripped):
                    break

                # ✅ Фикс A+C: обрываем только если явно НЕ внутри таблицы
                inside_tbl = is_inside_table_block(lines, j)
                if not inside_tbl:
                    if (re.match(r'^\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\s+[А-ЯЁ]',
                                 next_stripped)
                            and not re.search(r'\s{3,}', next_stripped)
                            and not in_appendix):
                        if not re.match(r'^\d+([.,]\d+)?\s*$', next_stripped):
                            break
                    # ✅ Фикс C: НЕ обрываем таблицу по ^N Название,
                    # если это выглядит как строка таблицы (числа, единицы, колонки)
                    if (chapter_full_re.match(next_stripped)
                            and not re.search(r'\s{3,}', next_stripped)
                            and not in_appendix):
                        # дополнительная защита: если строка содержит единицы или
                        # сильно похожа на строку таблицы — не обрываем
                        if not re.search(
                            r'\b(мм|см|кг|м|м/с|м/сут|‰|%|МПа|кгс|"
                            r'см²|м²|м³|\d+[,\.]\d+)\b',
                            next_stripped):
                            break

                if not next_stripped:
                    empty_count += 1
                    if empty_count >= 5:
                        break
                    table_lines.append(next_line)
                else:
                    empty_count = 0
                    table_lines.append(next_line)
                    if len(table_lines) > 500:
                        break
                j += 1

            table_text = '\n'.join(table_lines).strip()

            current_chapter = saved_chapter
            current_chapter_title = saved_chapter_title
            current_section = saved_section
            current_section_title = saved_section_title
            current_point = saved_point

            add_chunk(make_chunk(table_text, is_table=True,
                                 table_num=table_num,
                                 override_ctx=override_ctx))
            i = j
            continue

        # ---------- 2. Двухстрочный раздел ----------
        chapter_num_match = chapter_number_only_re.match(stripped)
        if (chapter_num_match
                and not in_table_context
                and not inside_table
                and not in_appendix
                and is_valid_chapter_num(chapter_num_match.group(1), filename)):
            j = i + 1
            while j < total_lines and not lines[j].strip():
                j += 1
            if j < total_lines:
                next_stripped = lines[j].strip()
                if (title_re.match(next_stripped)
                        and is_plausible_chapter_title(next_stripped)):
                    flush_buffer()
                    current_chapter = chapter_num_match.group(1)
                    current_chapter_title = next_stripped
                    current_section = ""
                    current_section_title = ""
                    current_point = ""
                    i = j + 1
                    continue

        # ---------- 3. Однострочный раздел ----------
        chapter_match = chapter_full_re.match(stripped)
        if (chapter_match
                and not inside_table
                and not in_appendix
                and is_valid_chapter_num(chapter_match.group(1), filename)):
            title = chapter_match.group(2).strip()
            # ✅ Фикс C: если «мы внутри таблицы» — не считаем заголовком
            if is_plausible_chapter_title(title) and not in_table_context:
                flush_buffer()
                current_chapter = chapter_match.group(1)
                current_chapter_title = title
                current_section = ""
                current_section_title = ""
                current_point = ""
                i += 1
                continue

        # ---------- 4. Подраздел ----------
        section_match = section_re.match(stripped)
        if section_match and not inside_table and not in_appendix:
            title = section_match.group(2).strip()
            if is_plausible_section_title(title):
                section_num = section_match.group(1)
                section_chapter = section_num.split('.')[0]
                if (current_chapter != section_chapter
                        and is_valid_chapter_num(section_chapter, filename)):
                    current_chapter = section_chapter
                    current_chapter_title = ""
                flush_buffer()
                current_section = section_num
                current_section_title = title
                current_point = ""
                i += 1
                continue

        # ---------- 5. Пункт ----------
        point_match = point_re.match(stripped)
        if (point_match
                and not is_excluded
                and not inside_table
                and not in_appendix):
            new_point = point_match.group(1)
            point_chapter = new_point.split('.')[0]
            if not is_valid_chapter_num(point_chapter, filename):
                current_buffer.append(line)
                i += 1
                continue
            if (current_chapter != point_chapter
                    and is_valid_chapter_num(point_chapter, filename)
                    and point_chapter.isdigit()):
                current_chapter = point_chapter
                current_chapter_title = ""
            flush_buffer()
            current_point = new_point
            current_buffer.append(line)
            i += 1
            continue

        # ---------- 6. Приложение ----------
        appendix_match = appendix_re.match(stripped)
        if appendix_match and not inside_table:
            flush_buffer()
            current_chapter = f"Приложение {appendix_match.group(1)}"
            current_chapter_title = stripped
            current_section = ""
            current_section_title = ""
            current_point = ""
            in_appendix = True
            i += 1
            continue

        if (in_appendix
                and chapter_full_re.match(stripped)
                and not inside_table):
            in_appendix = False

        # ---------- 7. Продолжение ----------
        if current_buffer or current_point:
            current_buffer.append(line)
        elif stripped:
            current_buffer.append(line)

        i += 1

    flush_buffer()
    flush_small_pending()
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