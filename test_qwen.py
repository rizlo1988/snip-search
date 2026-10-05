import os
from openai import OpenAI

with open("key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(
    api_key=api_key,
    base_url="https://foundation-models.api.cloud.ru/v1"
)

question = "Что такое снеговая нагрузка? Ответь кратко."

print("Отправляю вопрос модели...")
print(f"Вопрос: {question}")
print("-" * 50)

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[
        {"role": "user", "content": question}
    ],
    max_tokens=2000
)

# Показываем ТОЛЬКО чистый ответ
print("Ответ модели:")
print(response.choices[0].message.content)