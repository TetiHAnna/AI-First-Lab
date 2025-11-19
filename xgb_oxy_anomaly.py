import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 📥 Завантаження даних
df = pd.read_csv("combined_water_quality.csv")

# 🎯 Цільова змінна
target = "Kisen"
features = ["BSK5", "Fosfat", "Nitrat", "month", "Latitude", "Longitude"]

# 🧼 Очищення
df = df.dropna(subset=[target] + features)
X = df[features]
y = df[target]

print(f"📦 Кількість рядків у моделі: {len(X)}")

# 📊 Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Побудова моделі
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_test), index=y_test.index)

# 📈 Метрики
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n XGBoost R² score: {r2:.4f}")
print(f" XGBoost RMSE: {rmse:.3f}")

# 📉 Обчислення залишків
residuals = y_test - y_pred
std_res = np.std(residuals)
threshold = 2 * std_res

# 🚨 Виявлення аномалій
anomalies = residuals[np.abs(residuals) > threshold]
print(f"\n Виявлено аномалій: {len(anomalies)} (поріг ±{threshold:.3f})")

# 📊 Візуалізація
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.4, label="Прогноз", color="blue")
plt.scatter(y_test.loc[anomalies.index], y_pred[anomalies.index], color="red", label="Аномалії")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label="Ідеальний прогноз")
plt.xlabel("Реальні значення Kisen")
plt.ylabel("Прогнозовані значення")
plt.title("📉 Виявлення аномалій у прогнозі кисню (XGBoost)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
