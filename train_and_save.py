import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, cross_validate
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pickle  # Для сохранения модели и энкодера
import os  # Для проверки путей
import joblib

# Создаем папку для графиков, если она не существует
os.makedirs('plots', exist_ok=True)

csv_path='data.xlsx'

# Используйте переменную окружения для пути
csv_path = os.getenv('DATA_PATH', 'data.xlsx')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Файл {csv_path} не найден! Проверьте volume или путь.")

df = pd.read_excel(csv_path)
if df.empty:
    raise ValueError("Данные пустые!")
    
# Краткий анализ данных (твой исходный код, оставляем для проверки)
print("Размер данных:", df.shape)
print("Описание:", df.describe())
print("Инфо:", df.info())
print("Пропуски:", df.isnull().sum())
print("Колонки:", df.columns)

# Гистограммы для числовых колонок (твой код, оставляем)
sns.set_style("whitegrid")
sns.set_palette("husl")
numeric_cols = df.select_dtypes(include=[float, int]).columns
n_cols = len(numeric_cols)
n_rows = int(n_cols ** 0.5) + 1
n_cols_grid = int(n_cols / n_rows) + 1
fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(12, 8), sharex=False, sharey=False)
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i], alpha=0.7)
    axes[i].set_title(f'Распределение {col}', fontsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Частота' if i % n_cols_grid == 0 else '')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.suptitle('Распределения числовых колонок', fontsize=12, y=0.98)
plt.savefig('plots/histograms.png', dpi=300, bbox_inches='tight')
#plt.show()



# Корреляционная матрица (твой код)
corr = df.select_dtypes(include=['float', 'int']).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')
plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
#plt.show()

# Подготовка данных
x = df.drop(['cond rate', 'gas rate', 'sum cond', 'sum gas', 'NPV'], axis=1)  # Входные признаки
y = df['NPV']  # Целевая переменная

# One-Hot Encoding для категориальной колонки GS
encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' убирает одну колонку, чтобы избежать мультиколлинеарности
encoded_cols = encoder.fit_transform(x[['GS']])  # Обучаем энкодер на GS
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['GS']))
X = pd.concat([x.drop(['GS'], axis=1), encoded_df], axis=1)  # Финальный X с закодированными данными

feature_columns = X.columns.tolist()  # Сохраняем порядок колонок

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение XGBoost с GridSearch
xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
param_grid = {
    'n_estimators': [10, 50, 100, 200, 1000],
    'max_depth': [3, 5, 7, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)
print(f"Лучшее значение R2 на кросс-валидации: {grid_search.best_score_:.4f}")

# Лучшая модель
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Метрики (убираем дублирование из твоего кода)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Тестовые метрики для XGBoost: MAE={mae:.2f}, R2={r2:.2f}, MAPE={mape:.2%}")

# Графики анализа (твой код, оставляем)
plt.figure(figsize=(4, 4))
plt.scatter(y_test, y_pred, alpha=0.5, color='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Фактические значения NPV')
plt.ylabel('Предсказанные значения NPV')
plt.title('Сравнение фактических и предсказанных значений')
plt.grid(True)
plt.savefig('plots/prediction_scatter.png', dpi=300, bbox_inches='tight')
#plt.show()

# Важность признаков
feature_importances = best_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Признак': feature_names, 'Важность': feature_importances}).sort_values(by='Важность', ascending=False)
print("\nТоп-10 важных признаков:")
print(importance_df.head(10))
plt.figure(figsize=(7, 3))
plt.barh(importance_df['Признак'][:20], importance_df['Важность'][:20])
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.title('Важность признаков в XGBoost')
plt.gca().invert_yaxis()
plt.savefig('plots/importance.png', dpi=300, bbox_inches='tight')
#plt.show()

os.makedirs('models', exist_ok=True)
# Сохранение модели и энкодера в .pkl файлы (ключевой шаг для продакшна!)
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(encoder, 'models/encoder.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl') 
print("Модель,энкодер и последовательность признаков сохранены в model.pkl и encoder.pkl")
