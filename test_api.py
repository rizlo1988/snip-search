from openai import OpenAI
import time

# ЗАМЕНИ НА СВОЙ КЛЮЧ ИЗ key.txt
API_KEY = "YjYxNjAwZWMtOTEzOC00MWI4LWE0NGMtNTFhYjIzNjI5ZTE2.87a0b2bc725be13f403d45e6a5b1c950"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://foundation-models.api.cloud.ru/v1",
    timeout=120.0
)

print("=" * 60)
print("Тест 1: Qwen/Qwen3-30B-A3B с отключённым reasoning")
print("=" * 60)
try:
    t0 = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",
        messages=[{"role": "user", "content": "Сколько будет 2+2? Ответь цифрой."}],
        max_tokens=100,
        extra_body={"enable_thinking": False}
    )
    dt = time.time() - t0
    print(f"⏱ Время: {dt:.1f} сек")
    print(f"✅ Content: {repr(response.choices[0].message.content)}")
    print(f"   finish_reason: {response.choices[0].finish_reason}")
except Exception as e:
    print(f"❌ {type(e).__name__}: {str(e)}")

print()
print("=" * 60)
print("Тест 2: Qwen/Qwen3-30B-A3B БЕЗ отключения reasoning")
print("=" * 60)
try:
    t0 = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B",
        messages=[{"role": "user", "content": "Сколько будет 2+2? Ответь цифрой."}],
        max_tokens=2000
    )
    dt = time.time() - t0
    print(f"⏱ Время: {dt:.1f} сек")
    print(f"✅ Content: {repr(response.choices[0].message.content)}")
    print(f"   finish_reason: {response.choices[0].finish_reason}")
except Exception as e:
    print(f"❌ {type(e).__name__}: {str(e)}")