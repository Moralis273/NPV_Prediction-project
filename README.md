### 🎯 NPV Prediction - Oil & Gas ML Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MLOps](https://img.shields.io/badge/MLOps-DVC%20%2B%20MLflow-orange)
![API](https://img.shields.io/badge/API-FastAPI-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

Прогнозирование NPV (Net Present Value) для нефтегазовых скважин с использованием полного MLOps пайплайна на основании геологофизических и технологических характеристик.

## 🏗️ Архитектура проекта
oil_2/
├── 📊 data/ # Данные (версионируются через DVC)
│ ├── raw/ # Исходные данные
│ └── processed/ # Обработанные данные
├── 🔧 src/ # Исходный код пайплайна
│ ├── preprocess.py # Предобработка данных
│ ├── train.py # Обучение модели
│ ├── evaluate.py # Оценка модели
│ ├── generate_report.py # Генерация отчетов
│ └── register_model.py # Регистрация в MLflow
├── 🤖 models/ # Модели и артефакты (DVC)
├── 📈 mlruns/ # MLflow эксперименты
├── 📝 notebooks/ # EDA и исследования
├── 📄 reports/ # Автоматические отчеты
├── 🏷️ registry/ # Информация о моделях
└── ⚙️ config/ # Конфигурационные файлы


___

#### Предварительные требования

- **Python 3.9+**
- **Git**
- **DVC** 
- **MLflow** 
___
### Установка и настройка

1. **Клонирование репозитория**
```bash
git clone https://github.com/yourusername/oil_2.git
cd oil_2

Создание виртуального окружения
# При
bash
python -m venv oil
source oil/bin/activate  # Linux/Mac
# или
oil\Scripts\activate     # Windows
Установка зависимостей

bash
pip install -r requirements.txt
Инициализация DVC

bash
dvc init
# Настройка remote storage (пример для Google Drive)
dvc remote add -d myremote gdrive://your-folder-id
Загрузка данных

bash
dvc pull
Запуск полного пайплайна
bash
# Запуск всего ML пайплайна
dvc repro

Запуск сервисов
bash
# 1. FastAPI сервер (порт 8001)
python app.py

# 2. MLflow UI (порт 5005)
mlflow server --backend-store-uri file:mlruns --host localhost --port 5005

# 3. Streamlit UI (порт 8501)
streamlit run streamlit_app.py
После запуска откройте:

API Docs: http://localhost:8000/docs

MLflow UI: http://localhost:5005

Streamlit App: http://localhost:8501

###     📊 ML Pipeline
Проект использует DVC для управления полным ML пайплайном:

Этапы пайплайна (dvc.yaml)
Preprocessing - очистка и подготовка данных
Training - обучение XGBoost модели с кросс-валидацией
Evaluation - детальная оценка модели на тестовых данных
Reporting - генерация комплексного отчета
Registration - регистрация модели в MLflow Registry

Модель и метрики
Алгоритм: XGBoost Regressor

Целевая переменная: NPV (Net Present Value)

Метрики: R², MAE, MAPE, Cross-validation scores

Валидация: 5-fold cross-validation



# Пример запроса к FastAPI
python
import requests

# Данные для предсказания
data = {
    "Heff": 15.0,      # Эффективная толщина
    "Perm": 150.0,     # Проницаемость
    "Sg": 0.75,        # Газонасыщенность
    "L_hor": 600.0,    # Горизонтальная длина
    "GS": "S-TYPE",    # Тип ствола
    "temp": 25.0,      # Темп падения
    "C5": 0.6,         # Содержание C5+
    "GRP": 2,          # Количество стадий ГРП
    "nGS": 3           # Количество горизонтальных стволов
}

# Отправка запроса
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Predicted NPV: ${result['predicted_NPV']:,.2f}")