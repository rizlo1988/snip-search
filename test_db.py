import chromadb
from db_builder import DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

all_data = collection.get(include=["documents", "metadatas"])
print(f"Всего чанков: {len(all_data['documents'])}\n")

# === 1. Общая статистика ===
checks = {
    "Приложение А": "Приложение А",
    "Таблица А.1": "Таблица А.1",
    "просвет под рейкой": "просвет под рейкой",
    "Толщина слоя": "Толщина слоя",
    "Поперечные уклоны": "Поперечные уклоны",
    "Высотные отметки": "Высотные отметки",
}

for name, keyword in checks.items():
    found = [d for d in all_data['documents'] if keyword in d]
    print(f"'{name}': {len(found)} чанков")

print()

# === 2. Ищем чанки с Приложением А + цифрами ===
found = []
for i, doc in enumerate(all_data['documents']):
    meta = all_data['metadatas'][i]
    has_app_a = "Приложение А" in doc or "Таблица А.1" in doc
    has_numbers = ("±5" in doc or "±10" in doc or "±0,005" in doc
                   or "0,005" in doc or "просвет под рейкой" in doc.lower()
                   or "±3" in doc)
    if has_app_a and has_numbers:
        found.append({
            'chapter': meta.get('chapter', ''),
            'is_table': meta.get('is_table', False),
            'table_number': meta.get('table_number', ''),
            'len': len(doc),
            'preview': doc[:500]
        })

print(f"=== Чанков с 'Приложение А' + цифры: {len(found)} ===\n")
for f in found[:5]:
    print(f"--- chapter={f['chapter']}, is_table={f['is_table']}, "
          f"table={f['table_number']}, len={f['len']} ---")
    print(f['preview'])
    print()

# === 3. Ищем чанки по метаданным (chapter = "Приложение А") ===
by_meta = []
for i, meta in enumerate(all_data['metadatas']):
    if meta.get('chapter', '').startswith('Приложение А'):
        by_meta.append({
            'chapter': meta.get('chapter', ''),
            'is_table': meta.get('is_table', False),
            'table_number': meta.get('table_number', ''),
            'len': len(all_data['documents'][i]),
            'preview': all_data['documents'][i][:800]
        })

print(f"=== Чанков с chapter = 'Приложение А*': {len(by_meta)} ===\n")
for f in by_meta[:10]:
    print(f"--- chapter={f['chapter']}, is_table={f['is_table']}, "
          f"table={f['table_number']}, len={f['len']} ---")
    print(f['preview'])
    print()

# === 4. Итог ===
print("=" * 60)
if by_meta:
    print(f"✅ Приложение А НАЙДЕНО в метаданных — {len(by_meta)} чанков")
    print("Значит, парсер работает. Проблема в поиске (app.py).")
else:
    print(f"❌ Приложение А НЕ в метаданных chapter")
    print("Значит, парсер не создал чанк с chapter='Приложение А'")
    print("Проверь db_builder.py — есть ли строка in_appendix = True")