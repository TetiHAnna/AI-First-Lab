import pandas as pd

# Завантаження даних
df = pd.read_csv("combined_water_quality.csv")

# Встановлення порогу
azot_limit = 2.0

# Фільтрація перевищень
exceed_df = df[df["Azot"] > azot_limit]

# 📊 Статистика
total_samples = len(df)
exceed_count = len(exceed_df)
exceed_percent = round((exceed_count / total_samples) * 100, 2)

print(f"🔍 Всього проб: {total_samples}")
print(f"⚠️ Перевищень азоту > {azot_limit} мг/дм³: {exceed_count} ({exceed_percent}%)")

# 📍 Найбільш проблемні пости
top_posts = exceed_df["Post_Name"].value_counts().head(10)
print("\n🚨 Топ-10 постів за кількістю перевищень:")
print(top_posts)
