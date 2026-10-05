import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Читаем ключ
with open("key.txt", "r") as f:
    api_key = f.read().strip()

# Подключаемся к модели
client = OpenAI(
    api_key=api_key,
    base_url="https://foundation-models.api.cloud.ru/v1"
)

# Подключаемся к векторной базе
chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(
    name="snip_docs",
    embedding_function=default_ef
)

print("=" * 60)
print("ПОИСК ПО СТРОИТЕЛЬНЫМ НОРМАМ (ВЕКТОРНАЯ БАЗА)")
print("=" * 60)
print("Введите 'выход' или 'exit', чтобы завершить программу.")
print("=" * 60)

while True:
    question = input("\nВаш вопрос: ").strip()
    
    if question.lower() in ["выход", "exit", "quit", "q"]:
        print("До свидания!")
        break
    
    if not question:
        continue
    
    print("\nИщу в базе знаний...")
    print("-" * 60)
    
    # Ищем 10 самых похожих кусочков (увеличили!)
    results = collection.query(
        query_texts=[question],
        n_results=10
    )
    
    # Собираем текст найденных кусочков
    context = ""
    sources = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            context += f"\n\n--- Фрагмент из {source} ---\n{doc}"
            if source not in sources:
                sources.append(source)
    
    print(f"Найдено фрагментов: {len(results['documents'][0])}")
    print(f"Источники: {', '.join(sources)}")
    print("-" * 60)
    
    # Формируем запрос к модели
    prompt = f"""Ты — эксперт по строительным нормам и правилам.
Ответь на вопрос пользователя, используя ТОЛЬКО информацию из фрагментов документов ниже.
Если в фрагментах нет ответа — честно скажи об этом.
Обязательно укажи, из какого документа и пункта взята информация.

ФРАГМЕНТЫ ДОКУМЕНТОВ:
{context}

ВОПРОС:
{question}
"""
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3.6-35B-A3B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000
        )
        
        answer = response.choices[0].message.content
        print("ОТВЕТ:")
        print(answer)
        print("-" * 60)
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")