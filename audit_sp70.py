import chromadb

c = chromadb.PersistentClient(path='chroma_db')
col = c.get_collection('snip_norms')

r = col.get(
    where={'source': 'СП 70.13330.2012 НЕСУЩИЕ И ОГРАЖДАЮЩИЕ КОНСТРУКЦИИ'},
    include=['metadatas', 'documents']
)
docs = r['documents']
metas = r['metadatas']

print(f'Всего чанков СП 70: {len(docs)}')
print('\n=== ПЕРВЫЕ 5 ЧАНКОВ ===')
for i in range(5):
    m = metas[i]
    print(f'\n--- Чанк {i}: chapter={m.get("chapter")}, '
          f'section={m.get("section")}, point={m.get("point")}, '
          f'table={m.get("table_number")}, len={len(docs[i])}')
    print(docs[i][:300])

print('\n=== 5 ЧАНКОВ ИЗ СЕРЕДИНЫ (индексы 200-204) ===')
for i in range(200, 205):
    m = metas[i]
    print(f'\n--- Чанк {i}: chapter={m.get("chapter")}, '
          f'section={m.get("section")}, point={m.get("point")}, '
          f'table={m.get("table_number")}, len={len(docs[i])}')
    print(docs[i][:300])