# check.py
import os, glob

p = glob.glob("documents/*Мосты*")[0]
print(f"Файл: {p}")
with open(p, encoding="utf-8") as f:
    lines = f.read().split("\n")

for i, l in enumerate(lines, 1):
    s = l.strip()
    if not s:
        continue
    # Ищем строки, начинающиеся с 9, 10, 11, 12, 13, 14 + пробел
    for n in ("9", "10", "11", "12", "13", "14"):
        if s.startswith(n + " "):
            print(f"{i:5d} |{repr(s[:100])}")
            break