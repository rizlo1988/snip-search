import os
import re
import chromadb
from chromadb.utils import embedding_functions

# Папка с документами
DOCS_FOLDER = "documents"

client = chromadb.PersistentClient(path="./chroma_db")

ru_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="cointegrated/rubert-tiny2"
)

try:
    client.delete_collection(name="snip_docs")
    print("Старая база удалена. Создаю новую...")
except:
    print("Создаю новую базу...")

collection = client.get_or_create_collection(
    name="snip_docs",
    embedding_function=ru_ef
)

def find_context(text, chunk_start):
    """Ищет ближайший раздел/пункт перед началом кусочка"""
    # Ищем последний заголовок вида "12.3.4" или "9.2" перед кусочком
    pattern = r'(\d+(?:\.\d+)*)\s+[А-Яа-я]'
    matches = list(re.finditer(pattern, text[:chunk_start]))
    if matches:
        return matches[-1].group(1)
    return ""

def find_tables(text):
    """Ищет упоминания таблиц в тексте кусочка"""
    tables = re.findall(r'Таблиц[аы]\s+([А-ЯA-Z]?\.?\d+(?:\.\d+)?)', text)
    return tables

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((chunk, start))
        start = end - overlap
    return chunks

print("Начинаю загрузку документов в базу...")

for filename in os.listdir(DOCS_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunks = split_text(content)
        print(f"  {filename}: разбит на {len(chunks)} кусочков")
        
        for i, (chunk, start_pos) in enumerate(chunks):
            # Определяем ближайший раздел
            section = find_context(content, start_pos)
            # Ищем упоминания таблиц в кусочке
            tables = find_tables(chunk)
            table_info = ", ".join(tables) if tables else ""
            
            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": filename,
                    "chunk_id": i,
                    "section": section,
                    "tables": table_info
                }],
                ids=[f"{filename}_{i}"]
            )

print("=" * 60)
print("Готово! База данных создана с метаданными (разделы и таблицы).")