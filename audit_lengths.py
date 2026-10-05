import chromadb
from collections import Counter

c = chromadb.PersistentClient(path='chroma_db')
col = c.get_collection('snip_norms')

sources = sorted(set(m['source'] for m in col.get(include=['metadatas'])['metadatas']))
for s in sources:
    r = col.get(where={'source': s}, include=['metadatas', 'documents'])
    docs = r['documents']
    metas = r['metadatas']
    lengths = sorted(len(d) for d in docs)

    print(f'\n=== {s} ({len(docs)}) ===')
    print(f'  min={lengths[0]}, max={lengths[-1]}, '
          f'median={lengths[len(lengths)//2]}, avg={sum(lengths)//len(lengths)}')

    # распределение по диапазонам
    buckets = Counter()
    for L in lengths:
        if L < 100: buckets['<100'] += 1
        elif L < 300: buckets['100-299'] += 1
        elif L < 600: buckets['300-599'] += 1
        elif L < 1000: buckets['600-999'] += 1
        elif L < 1500: buckets['1000-1499'] += 1
        else: buckets['>=1500'] += 1
    print(f'  Длины: {dict(buckets)}')

    # сколько чанков с пустым chapter и пустым point
    no_ch = sum(1 for m in metas if not m.get('chapter'))
    no_pt = sum(1 for m in metas if not m.get('point'))
    no_tb = sum(1 for m in metas if not m.get('table_number'))
    print(f'  Пустой chapter: {no_ch}, пустой point: {no_pt}, пустой table: {no_tb}')