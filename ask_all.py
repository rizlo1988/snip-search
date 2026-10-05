import os
from openai import OpenAI

# Читаем ключ
with open("key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(
    api_key=api_key,
    base_url="https://foundation-models.api.cloud.ru/v1"
)

# Папка с документами
DOCS_FOLDER = "documents"

# Читаем все файлы из папки
all_text = ""
file_count = 0

for filename in os.listdir(DOCS_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            all_text += f"\n\n=== ДОКУМЕНТ: {filename} ===\n\n"
            all_text += text
            file_count += 1
            print(f"Загружен: {filename} ({len(text)} символов)")

print("=" * 60)
print(f"Всего загружено документов: {file_count}")
print(f"Общий размер текста: {len(all_text)} символов")
print("=" * 60)

# Увеличили лимит до 700 000 символов (чтобы точно влезло в контекст модели)
MAX_SIZE = 700000
if len(all_text) > MAX_SIZE:
    print(f"ВНИМАНИЕ: Текст слишком большой ({len(all_text)} символов).")
    print(f"Будет использована только первая часть (до {MAX_SIZE} символов).")
    all_text = all_text[:MAX_SIZE]
    print("=" * 60)

print("Введите 'выход' или 'exit', чтобы завершить программу.")
print("=" * 60)

# Бесконечный цикл для вопросов
while True:
    question = input("\nВаш вопрос: ").strip()
    
    if question.lower() in ["выход", "exit", "quit", "q"]:
        print("До свидания!")
        break
    
    if not question:
        print("Вы ничего не ввели. Попробуйте снова.")
        continue
    
    print("\nИщу ответ в документах...")
    print("-" * 60)
    
    prompt = f"""Отвечай кратко, без размышлений. Сразу давай ответ.
Ты — эксперт по строительным нормам и правилам.
Ответь на вопрос пользователя, используя ТОЛЬКО информацию из документов ниже.
Если в документах нет ответа — честно скажи об этом.
Обязательно укажи, из какого документа, раздела или пункта взята информация.

ДОКУМЕНТЫ:
{all_text}

ВОПРОС:
{question}
"""
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3.6-35B-A3B",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000
        )
        
        answer = response.choices[0].message.content
        print("ОТВЕТ:")
        print(answer)
        print("-" * 60)
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("Попробуйте задать вопрос ещё раз.")