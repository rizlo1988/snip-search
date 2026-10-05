import os
from openai import OpenAI

with open("key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(
    api_key=api_key,
    base_url="https://foundation-models.api.cloud.ru/v1"
)

# Читаем документ
with open("document.txt", "r", encoding="utf-8") as f:
    document_text = f.read()

print(f"Документ загружен. Размер: {len(document_text)} символов")
print("-" * 50)

# Вопрос пользователя
question = "Какие дороги относятся к автомобильным дорогам общего пользования?"

print(f"Вопрос: {question}")
print("-" * 50)

# Формируем запрос к модели
prompt = f"""
Ты — эксперт по строительным нормам и правилам.
Ответь на вопрос пользователя, используя ТОЛЬКО информацию из документа ниже.
Если в документе нет ответа — скажи об этом.
Обязательно укажи, из какого раздела или пункта документа взята информация.

ДОКУМЕНТ:
{document_text}

ВОПРОС:
{question}
"""

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=2000
)

print("Ответ модели:")
print(response.choices[0].message.content)