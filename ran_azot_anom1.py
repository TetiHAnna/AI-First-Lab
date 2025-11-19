import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 📥 Завантаження даних
df = pd.read_csv("combined_water_quality.csv")
df = df.dropna(subset=["Azot", "BSK5", "Fosfat", "Nitrat", "month", "Latitude", "Longitude", "Post_Name"])

# 🎯 Ознаки та ціль
features = ["BSK5", "Fosfat", "Nitrat", "month", "Latitude", "Longitude"]
X = df[features]
y = df["Azot"]

# 📊 Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_test), index=y_test.index)

# 📉 Залишки та аномалії
residuals = y_test - y_pred
std_res = np.std(residuals)
threshold = 2 * std_res
anomalies = residuals[np.abs(residuals) > threshold]

# 📊 Побудова графіка
plt.figure(figsize=(8,6))

# Ідеальний прогноз
x_vals = np.linspace(min(y_test), max(y_test), 100)
plt.plot(x_vals, x_vals, linestyle="--", color="black", label="Ідеальний прогноз")

# Прогнозовані точки
plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Прогноз")

# Аномалії
plt.scatter(y_test.loc[anomalies.index], y_pred.loc[anomalies.index], color="red", label="Аномалії")

plt.xlabel("Реальні значення Azot")
plt.ylabel("Прогнозовані значення")
plt.title("Виявлення аномалій у прогнозі азоту (Random Forest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
