# test_parser.py — диагностика парсера db_builder.py
# Запуск: python test_parser.py
# Не требует API-ключа, не пишет в chroma_db.
#
# ВАЖНО: RE_CHAPTER_STRICT импортируется из db_builder, чтобы диагностика
# считала фейки тем же regex, что и сам парсер.

import os
import re
from collections import Counter

from db_builder import (
    parse_document,
    merge_short_chunks,
    RE_CHAPTER,
    RE_CHAPTER_STRICT,
    RE_POINT,
    RE_SUBITEM,
    RE_TABLE,
    VALID_CHAPTER_RANGES,
    get_valid_range,
    _is_real_chapter_lookahead,
)

DOCS_DIR = "documents"


def banner(text: str):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def analyze_document(path: str):
    fname = os.path.basename(path)
    source = fname.replace(".txt", "").strip()

    banner(f"ФАЙЛ: {fname}")
    print(f"source = {source!r}")
    valid_range = get_valid_range(source)
    print(f"VALID_CHAPTER_RANGES -> {valid_range}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().split("\n")]

    chunks_current = parse_document(path)
    chunks_current = merge_short_chunks(chunks_current)

    chapters_current = Counter(c["chapter"] for c in chunks_current if c["chapter"])
    tables_current = sorted(set(c["table_number"] for c in chunks_current if c["table_number"]))

    print(f"\n[ТЕКУЩИЙ ПАРСЕР]")
    print(f"  Чанков: {len(chunks_current)}")
    print(f"  Глав (chapter): {sorted(chapters_current.keys(), key=lambda x: int(x) if x.isdigit() else 999)}")
    print(f"  Таблиц (table_number): {tables_current}")
    print(f"  Чанков с пустым chapter: {sum(1 for c in chunks_current if not c['chapter'])}")

    candidates_chapter = []
    candidates_strict = []
    candidates_rejected = []
    candidates_rejected_by_lookahead = []

    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue

        m1 = RE_CHAPTER.match(s)
        m2 = RE_CHAPTER_STRICT.match(s)

        if m1:
            num = int(m1.group(1))
            candidates_chapter.append((i, num, s))
            if m2:
                if _is_real_chapter_lookahead(lines, i):
                    candidates_strict.append((i, num, s))
                else:
                    candidates_rejected_by_lookahead.append((i, num, s))
            else:
                candidates_rejected.append((i, num, s))

    print(f"\n[АНАЛИЗ СТРОК]")
    print(f"  RE_CHAPTER матчит строк: {len(candidates_chapter)}")
    print(f"  RE_CHAPTER_STRICT + lookahead матчит: {len(candidates_strict)}")
    print(f"  RE_CHAPTER матчит, но STRICT — нет: {len(candidates_rejected)}")
    print(f"  RE_CHAPTER матчит, STRICT — да, но lookahead — нет: {len(candidates_rejected_by_lookahead)}")

    inside_range = []
    if valid_range:
        lo, hi = valid_range
        for (i, num, s) in candidates_rejected + candidates_rejected_by_lookahead:
            if lo <= num <= hi:
                inside_range.append((i, num, s))

    print(f"\n[ФЕЙКОВЫЕ ГЛАВЫ В ДИАПАЗОНЕ {valid_range}]")
    print(f"  Таких строк: {len(inside_range)}")
    print(f"  Первые 30:")
    for (i, num, s) in inside_range[:30]:
        nxt = ""
        for j in range(i + 1, min(i + 4, len(lines))):
            if lines[j].strip():
                nxt = lines[j].strip()
                break
        print(f"    строка {i:5d}  |N={num:2d}| {s[:70]}")
        if nxt:
            print(f"                        -> продолжение: {nxt[:60]}")

    table_closure_candidates = []
    in_table = False
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        if RE_TABLE.match(s):
            in_table = True
            continue
        if not in_table:
            continue
        m_pt = RE_POINT.match(s)
        if m_pt and not RE_SUBITEM.match(s):
            table_closure_candidates.append((i, m_pt.group(1), s))

    print(f"\n[СТРОКИ N.M ВНУТРИ ТАБЛИЦ (потенциально ложное закрытие таблицы)]")
    print(f"  Таких строк: {len(table_closure_candidates)}")
    for (i, pn, s) in table_closure_candidates[:20]:
        print(f"    строка {i:5d}  |{pn:>6}| {s[:70]}")

    tables_in_file = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if RE_TABLE.match(s):
            tables_in_file.append((i, RE_TABLE.match(s).group(1)))

    print(f"\n[ТАБЛИЦЫ В ФАЙЛЕ]")
    print(f"  Найдено: {len(tables_in_file)}")
    for (i, tn) in tables_in_file[:30]:
        print(f"    строка {i:5d}  Таблица {tn}")

    print(f"\n[СВОДКА]")
    print(f"  Всего строк: {len(lines)}")
    print(f"  RE_CHAPTER матчит: {len(candidates_chapter)}")
    print(f"  Из них фейковых (не STRICT + lookahead): {len(candidates_rejected) + len(candidates_rejected_by_lookahead)}")
    print(f"  Из них фейковых и в диапазоне: {len(inside_range)}")
    print(f"  Таблиц в файле: {len(tables_in_file)}")
    print(f"  Строк N.M внутри таблиц: {len(table_closure_candidates)}")


def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"Папка {DOCS_DIR}/ не найдена. Запусти из корня проекта Snip/.")
        return

    files = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".txt"))
    print(f"Найдено документов: {len(files)}")
    for f in files:
        print(f"  - {f}")

    total_current = 0
    per_doc = []

    for fname in files:
        path = os.path.join(DOCS_DIR, fname)
        try:
            analyze_document(path)
        except Exception as e:
            banner(f"ОШИБКА при разборе {fname}: {e}")
            import traceback
            traceback.print_exc()
            continue

        chunks = merge_short_chunks(parse_document(path))
        total_current += len(chunks)
        per_doc.append((fname, len(chunks)))

    banner("ИТОГ ПО ВСЕМ ФАЙЛАМ (текущий парсер)")
    for fname, n in per_doc:
        print(f"  {n:5d}  {fname}")
    print(f"  {total_current:5d}  ВСЕГО")


if __name__ == "__main__":
    main()