"""
Аудит базы. Запуск: python audit.py
Печатает статистику по chroma_db.
"""
import chromadb
import re
from db_builder import DB_PATH, COLLECTION_NAME


def audit_database(collection):
    lines = []

    def out(s=""):
        lines.append(str(s))
        print(s)

    all_meta = collection.get(include=["metadatas", "documents"])
    sources = sorted(set(
        m["source"] for m in all_meta["metadatas"] if m.get("source")
    ))
    out(f"Всего чанков: {len(all_meta['documents'])}")
    out(f"Всего документов: {len(sources)}")
    out("=" * 60)

    for src in sources:
        r = collection.get(
            where={"source": src},
            include=["metadatas", "documents"]
        )
        metas = r["metadatas"]
        docs = r["documents"]
        n = len(docs)
        if n == 0:
            out(f"=== {src} === пусто")
            continue

        chapters = sorted(set(
            m.get("chapter", "") for m in metas if m.get("chapter")
        ))
        points = sorted(set(
            m.get("point", "") for m in metas if m.get("point")
        ))
        tables = sorted(set(
            m.get("table_number", "") for m in metas if m.get("table_number")
        ))

        no_chapter = sum(1 for m in metas if not m.get("chapter"))
        no_point = sum(1 for m in metas if not m.get("point"))
        short = sum(1 for d in docs if len(d) < 200)
        no_digits = sum(1 for d in docs if not re.search(r"\d", d))
        avg_len = sum(len(d) for d in docs) / n

        out(f"=== {src} ({n} чанков) ===")
        out(f"  Средняя длина: {avg_len:.0f}")
        out(f"  Пустой chapter: {no_chapter}")
        out(f"  Пустой point: {no_point}")
        out(f"  <200 символов: {short}")
        out(f"  Без цифр: {no_digits}")
        out(f"  Разделы ({len(chapters)}): {chapters}")
        out(f"  Пунктов ({len(points)}): {points[:40]}")
        out(f"  Таблицы ({len(tables)}): {tables}")
        out()

        # Спец-проверка Таблицы 13 в СП 46
        if "МОСТЫ И ТРУБЫ" in src.upper():
            out("  >>> Проверка Таблицы 13 в СП 46:")
            try:
                t13 = collection.get(
                    where={"table_number": "13"},
                    include=["metadatas", "documents"]
                )
                if t13.get("documents"):
                    out(f"      Чанков с table_number=13: {len(t13['documents'])}")
                    for k, d in enumerate(t13["documents"][:3], 1):
                        meta = t13["metadatas"][k - 1]
                        out(f"      Чанк {k} (длина {len(d)}), "
                            f"chapter={meta.get('chapter')}, "
                            f"section={meta.get('section')}, "
                            f"point={meta.get('point')}")
                        out(f"      {d[:500]}...")
                        out()
                else:
                    out("      ❌ Таблица 13 не найдена")
            except Exception as e:
                out(f"      Ошибка: {e}")

            for needle in ["продольной оси трубы",
                           "уступов в рядах",
                           "зазоров между секциями"]:
                try:
                    t = collection.get(
                        where_document={"$contains": needle},
                        include=["documents"]
                    )
                    cnt = len(t.get("documents") or [])
                    out(f"  >>> '{needle}': найдено чанков {cnt}")
                except Exception as e:
                    out(f"  >>> '{needle}': ошибка {e}")
            out()

        # Спец-проверка разделов СП 126
        if "126" in src:
            out("  >>> Проверка разделов 1-10 в СП 126:")
            for num in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
                cnt = sum(1 for m in metas if m.get("chapter") == num)
                out(f"      chapter={num}: {cnt} чанков")
            out()


if __name__ == "__main__":
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_collection(name=COLLECTION_NAME)
    audit_database(coll)