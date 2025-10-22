# NPV Prediction Platform

FastAPI API для инференса + DVC пайплайн обучения + MLflow трекинг/Registry + Streamlit UI, упакованные в Docker Compose. Проект решает регрессию по табличным данным для прогноза **NPV** (чистой приведённой стоимости) сценариев бурения/разработки нефтегазовых месторождений Ачимовских пластов.

---

## 📦 Состав репозитория

```
.
├── app.py                     # FastAPI: эндпоинты /predict, /model_info, /health
├── docker-compose.yml         # Сервисы: mlflow, api, streamlit
├── Dockerfile                 # Базовый образ Python + зависимости
├── dvc.yaml                   # DVC пайплайн: preprocess → train → evaluate → report → register
├── dvc.lock                   # Зафиксированные артефакты/хэши стадий
├── params.yaml                # Гиперпараметры модели и конфиг пайплайна
├── requirements.txt           # Python-зависимости
├── src/
│   ├── preprocess.py          # Загрузка/обработка данных, OHE, train/test split
│   ├── train.py               # Обучение XGBoost + логирование в MLflow
│   ├── evaluate.py            # Подсчёт метрик на тесте
│   ├── generate_report.py     # Сводный отчёт по эксперименту
│   └── register_model.py      # Регистрация в MLflow Model Registry
├── streamlit_app.py           # Веб-интерфейс для бизнеса
├── data/
│   └── raw/data.xlsx          # (пример) исходные данные
├── models/                    # Артефакты инференса: model.joblib, encoder.joblib, feature_columns.joblib
├── reports/                   # Итоговые отчёты (JSON)
└── registry/                  # Инфо о зарегистрированной модели (JSON)
```

---

## 🏗 Архитектура решения

* **DVC** управляет стадиями ML-пайплайна и версиями данных/метрик.
* **MLflow** хранит параметры, метрики, артефакты и версии модели (Model Registry).
* **FastAPI** — сервис инференса с валидацией входов и строгим порядком признаков.
* **Streamlit** — удобный UI для бизнес-пользователей.
* **Docker Compose** — единая оркестрация локально/на сервере.

Пайплайн DVC:

```
preprocess → train → evaluate → generate_report → register
```

---

## ⚙️ Требования

* Docker 20+
* Docker Compose v2+
* (Для локального запуска без Docker) Python 3.9, Git, DVC, MLflow

---

## 🚀 Быстрый старт (Docker Compose)

1. Скопируйте `.env` при необходимости (необязательно) и/или задайте переменные окружения.
2. Соберите и поднимите сервисы:

   ```bash
   docker compose up --build
   ```
3. Проверьте доступность сервисов:

   * MLflow UI: `http://localhost:5000`
   * API (FastAPI docs): `http://localhost:8001/docs`
   * Streamlit UI: `http://localhost:8501`

> По умолчанию API стартует без обучения — он просто считывает артефакты из `models/`. Обучение выполняется через DVC (см. раздел ниже) или из CI.

---

## 🔧 Переменные окружения

Сервисы используют сетевые имена Docker. Ключевые переменные:

* `MLFLOW_TRACKING_URI=http://mlflow:5000` — где логировать ран/артефакты (пробрасывается в `api`, `streamlit`).
* `API_URL=http://api:8001` — базовый URL FastAPI для Streamlit.
* `MLFLOW_URL=http://mlflow:5000` — ссылка для кнопки «Открыть MLflow» в Streamlit.

Пример блока из `docker-compose.yml` (уже настроено):

```yaml
  api:
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000

  streamlit:
    environment:
      - API_URL=http://api:8001
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MLFLOW_URL=http://mlflow:5000
```

---

## 🧪 Обучение и эксперименты (DVC + MLflow)

### Запуск пайплайна обучения

```bash
# Локально (без Docker) — из корня репозитория
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Запустить полный пайплайн
dvc repro
```

Стадии (из `dvc.yaml`):

* **preprocess** — чтение `data/raw/data.xlsx`, OHE по `GS`, split 80/20, сохранение:

  * `data/processed/train_test.joblib`
  * `models/encoder.joblib`
  * `models/feature_columns.joblib`
* **train** — XGBoostRegressor с параметрами из `params.yaml`, CV + тестовые метрики, логирование в MLflow, сохранение `models/model.joblib` и `models/metrics.json`.
* **evaluate** — подсчёт MAE/R²/MAPE и др., `models/evaluation.json`.
* **generate_report** — сводный `reports/model_report.json`.
* **register** — регистрация последнего лучшего ран-а в MLflow Model Registry, трекинг в `registry/model_info.json`.

> **Примечание.** При работе с MLflow Server путь к артефактам для регистрации должен быть вида `runs:/<RUN_ID>/model`.

### Просмотр метрик

```bash
dvc metrics show         # короткая сводка
cat models/metrics.json  # детали по тесту и CV
cat models/evaluation.json
```

### Эксперименты DVC

```bash
dvc exp run                       # запуск эксперимента
dvc exp show --only-changed -T     # сравнение метрик/параметров
```

---

## 🌐 API (FastAPI)

Документация: `http://localhost:8001/docs`

### Эндпоинты

* `GET /health` — статус сервиса и загрузки модели.
* `GET /` — краткая информация о сервисе.
* `GET /model_info` — тип модели, число и список признаков.
* `POST /predict` — расчёт NPV по входным параметрам.

### Пример запроса `/predict`

```bash
curl -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Heff": 15.0,
    "Perm": 150.0,
    "Sg": 0.75,
    "L_hor": 600.0,
    "GS": "S-TYPE",
    "temp": 25.0,
    "C5": 0.6,
    "GRP": 2,
    "nGS": 3
  }'
```

**Ответ**

```json
{
  "predicted_NPV": 123456.78,
  "status": "success"
}
```

> На инференсе вход приводится к порядку признаков из `models/feature_columns.joblib`. Категориальный `GS` кодируется тем же `OneHotEncoder`, что был на обучении.

---

## 🖥 Streamlit UI

* URL: `http://localhost:8501`
* Возможности:

  * форма ввода параметров, мгновенный запрос в API
  * показ метрик модели из артефактов DVC
  * ссылка на MLflow UI
  * кнопка перезапуска DVC-пайплайна (для разработки)

---

## ⚙️ Конфигурация модели (`params.yaml`)

```yaml
data:
  raw_path: "data/raw/data.xlsx"
  processed_path: "data/processed/train_test.joblib"

features:
  target: "NPV"
  drop_columns: ["cond rate", "gas rate", "sum cond", "sum gas"]
  categorical_columns: ["GS"]

preprocessing:
  test_size: 0.2
  random_state: 42

model:
  name: "xgboost"
  hyperparameters:
    n_estimators: 250
    max_depth: 15
    learning_rate: 0.01
    subsample: 0.8

training:
  cv_folds: 5
  scoring: "r2"
```

---



## ✍️ Автор / Контакты

* Автор: Ruslan Khaidarshin
* Стек: FastAPI · XGBoost · DVC · MLflow · Streamlit · Docker
