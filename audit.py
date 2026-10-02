"""
Отдельный скрипт аудита базы. Запуск: python audit.py
Выводит статистику по chroma_db в консоль.
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
            continue

        chapters = sorted(set(m.get("chapter", "") for m in metas if m.get("chapter")))
        tables = sorted(set(m.get("table_number", "") for m in metas if m.get("table_number")))
        no_chapter = sum(1 for m in metas if not m.get("chapter"))
        no_point = sum(1 for m in metas if not m.get("point"))
        short = sum(1 for d in docs if len(d) < 200)
        avg_len = sum(len(d) for d in docs) / n

        out(f"=== {src} ({n} чанков) ===")
        out(f"  Средняя длина: {avg_len:.0f}")
        out(f"  Пустой chapter: {no_chapter}")
        out(f"  Пустой point: {no_point}")
        out(f"  <200 символов: {short}")
        out(f"  Разделы ({len(chapters)}): {chapters}")
        out(f"  Таблицы ({len(tables)}): {tables}")
        out()


if __name__ == "__main__":
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_collection(name=COLLECTION_NAME)
    audit_database(coll)