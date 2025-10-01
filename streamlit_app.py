import streamlit as st
import requests
import pandas as pd
import json
import os
from datetime import datetime

# Настройки страницы
st.set_page_config(
    page_title="NPV Prediction App",
    page_icon="📊",
    layout="wide"
)

# Заголовок приложения
st.title("💰 NPV Prediction App")
st.markdown("Прогнозирование NPV на основе параметров скважины")

# URL вашего API (используем переменные окружения для гибкости)
API_URL = st.sidebar.text_input(
    "URL API", 
    value=os.getenv('API_URL', 'http://localhost:8001'),  # Используем переменную окружения с дефолтом
    help="Введите URL вашего FastAPI сервера"
)

# MLflow URL для мониторинга
MLFLOW_URL = st.sidebar.text_input(
    "MLflow URL", 
    value=os.getenv('MLFLOW_URL', 'http://localhost:5000'),  # Используем переменную окружения с дефолтом
    help="Введите URL MLflow сервера"
)

# Функция для проверки соединения с API
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Функция для получения прогноза
def get_prediction(data):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Ошибка API: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Ошибка соединения: {str(e)}"}

# Функция для получения информации о модели
def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model_info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Функция для получения метрик из DVC
def get_dvc_metrics():
    try:
        if os.path.exists('models/metrics.json'):
            with open('models/metrics.json', 'r') as f:
                return json.load(f)
        return None
    except:
        return None

# Функция для получения параметров модели
def get_model_params():
    try:
        if os.path.exists('params.yaml'):
            import yaml
            with open('params.yaml', 'r') as f:
                return yaml.safe_load(f)
        return None
    except:
        return None

# Проверка соединения с API
st.sidebar.header("🔗 Соединение")
if st.sidebar.button("Проверить соединение с API"):
    if check_api_health():
        st.sidebar.success("✅ API доступен")
    else:
        st.sidebar.error("❌ API недоступен")

# Основная форма для ввода данных
st.header("📝 Ввод параметров")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Геологические параметры")
    heff = st.number_input("Эффективная высота (Heff)", min_value=0.0, value=10.0, step=0.1, 
                          help="Эффективная высота пласта")
    perm = st.number_input("Проницаемость (Perm)", min_value=0.0, value=100.0, step=1.0,
                          help="Проницаемость породы")
    sg = st.slider("Газонасыщенность (Sg)", min_value=0.0, max_value=1.0, value=0.8, step=0.01,
                  help="Доля газа в пласте")
    c5 = st.number_input("Содержание C5+", min_value=0.0, value=0.5, step=0.1,
                        help="Содержание тяжелых углеводородов")

with col2:
    st.subheader("Технологические параметры")
    l_hor = st.number_input("Горизонтальная длина (L_hor)", min_value=0.0, value=500.0, step=10.0,
                           help="Длина горизонтального участка")
    gs = st.selectbox("Тип проводки ствола", ["S-TYPE", "U-TYPE", "VGS", "GS", "NGS"],
                     help="Конфигурация скважины")
    temp = st.number_input("Темп падения", min_value=0.0, value=20.0, step=0.1,
                          help="Температура в пласте")
    grp = st.number_input("Количество стадий ГРП", min_value=0, value=1, step=1,
                         help="Количество стадий гидроразрыва пласта")
    ngs = st.number_input("Количество горизонтальных стволов", min_value=0, value=2, step=1,
                         help="Количество ветвей скважины")

# Кнопка предсказания
if st.button("🎯 Рассчитать NPV", type="primary"):
    # Подготовка данных для API
    input_data = {
        "Heff": heff,
        "Perm": perm,
        "Sg": sg,
        "L_hor": l_hor,
        "GS": gs,
        "temp": temp,
        "C5": c5,
        "GRP": grp,
        "nGS": ngs
    }
    
    # Получение прогноза
    with st.spinner("Получение прогноза..."):
        result = get_prediction(input_data)
    
    # Отображение результатов
    if "predicted_NPV" in result:
        # Красивое отображение результата
        st.success(f"## Прогнозируемый NPV: **${result['predicted_NPV']:,.2f}**")
        
        # Дополнительная аналитика
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("NPV", f"${result['predicted_NPV']:,.2f}")
        with col2:
            # Пример дополнительной метрики
            st.metric("Статус", "✅ Успешно")
        with col3:
            st.metric("Время", datetime.now().strftime("%H:%M:%S"))
        
        # Детальная информация
        with st.expander("📊 Детали запроса и ответа"):
            st.subheader("Входные параметры")
            st.json(input_data)
            
            st.subheader("Ответ API")
            st.json(result)
            
            # История предсказаний (в сессии)
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'input': input_data,
                'prediction': result['predicted_NPV']
            })
            
            st.subheader("История предсказаний")
            if st.session_state.prediction_history:
                history_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(history_df)
    else:
        st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")

# Боковая панель с информацией
st.sidebar.header("ℹ️ Информация о системе")

# Информация о модели из API
model_info = get_model_info()
if model_info and "error" not in model_info:
    st.sidebar.success("✅ Модель загружена")
    st.sidebar.write(f"**Тип модели:** {model_info.get('model_type', 'Неизвестно')}")
    st.sidebar.write(f"**Количество признаков:** {model_info.get('n_features', 'Неизвестно')}")
    
    if "features" in model_info:
        with st.sidebar.expander("Список признаков"):
            for feature in model_info["features"]:
                st.write(f"• {feature}")
else:
    st.sidebar.warning("⚠️ Модель не загружена")

# Метрики из DVC
dvc_metrics = get_dvc_metrics()
if dvc_metrics:
    with st.sidebar.expander("📈 Метрики модели (DVC)"):
        st.write(f"**R²:** {dvc_metrics.get('r2', 'N/A'):.4f}")
        st.write(f"**MAE:** {dvc_metrics.get('mae', 'N/A'):.2f}")
        st.write(f"**MAPE:** {dvc_metrics.get('mape', 'N/A'):.2%}")
        st.write(f"**CV R²:** {dvc_metrics.get('cv_mean', 'N/A'):.4f} ± {dvc_metrics.get('cv_std', 'N/A'):.4f}")

# Параметры модели
model_params = get_model_params()
if model_params:
    with st.sidebar.expander("⚙️ Параметры модели"):
        st.write(f"**Алгоритм:** {model_params.get('model', {}).get('name', 'N/A')}")
        params = model_params.get('model', {}).get('hyperparameters', {})
        for key, value in params.items():
            st.write(f"**{key}:** {value}")

# Ссылки на мониторинг
st.sidebar.header("📊 Мониторинг")
if st.sidebar.button("📈 Открыть MLflow"):
    st.markdown(f'<a href="{MLFLOW_URL}" target="_blank">📈 Перейти к MLflow</a>', unsafe_allow_html=True)

if st.sidebar.button("🔄 Перезапустить пайплайн"):
    with st.sidebar:
        with st.spinner("Запуск DVC пайплайна..."):
            import subprocess
            try:
                result = subprocess.run(['dvc', 'repro'], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Пайплайн успешно выполнен")
                else:
                    st.error(f"❌ Ошибка: {result.stderr}")
            except Exception as e:
                st.error(f"❌ Ошибка запуска: {e}")

# Примеры запросов
st.sidebar.header("🎯 Примеры запросов")

example_data = [
    {
        "name": "Пример 1: Стандартная скважина",
        "data": {"Heff": 15.0, "Perm": 150.0, "Sg": 0.75, "L_hor": 600.0, 
                "GS": "S-TYPE", "temp": 25.0, "C5": 0.6, "GRP": 2, "nGS": 3}
    },
    {
        "name": "Пример 2: Высокопродуктивная", 
        "data": {"Heff": 25.0, "Perm": 300.0, "Sg": 0.85, "L_hor": 800.0,
                "GS": "U-TYPE", "temp": 30.0, "C5": 0.7, "GRP": 3, "nGS": 4}
    }
]

# Для загрузки примеров используем session_state для обновления полей
if 'load_example' not in st.session_state:
    st.session_state.load_example = None

for i, example in enumerate(example_data):
    if st.sidebar.button(example["name"]):
        st.session_state.load_example = example["data"]
        st.rerun()  # Перезапускаем app для обновления полей

# Если пример загружен, обновляем поля
if st.session_state.load_example:
    heff = st.session_state.load_example["Heff"]
    perm = st.session_state.load_example["Perm"]
    sg = st.session_state.load_example["Sg"]
    l_hor = st.session_state.load_example["L_hor"]
    gs = st.session_state.load_example["GS"]
    temp = st.session_state.load_example["temp"]
    c5 = st.session_state.load_example["C5"]
    grp = st.session_state.load_example["GRP"]
    ngs = st.session_state.load_example["nGS"]
    st.session_state.load_example = None  # Сбрасываем после загрузки

# Инструкция
with st.expander("📖 Инструкция по использованию"):
    st.markdown("""
    ### 🚀 Быстрый старт
    
    1. **Запустите FastAPI сервер**: `python app.py` (порт 8001)
    2. **Запустите MLflow**: `mlflow server --backend-store-uri file:mlruns --host localhost --port 5000`
    3. **Заполните параметры** скважины или используйте примеры
    4. **Нажмите 'Рассчитать NPV'** для получения прогноза
    
    ### 🔧 Для разработчиков
    
    - **DVC пайплайн**: `dvc repro` - перезапуск обучения
    - **Метрики**: `dvc metrics show` - просмотр метрик
    - **Эксперименты**: `dvc exp run` - запуск экспериментов
    
    ### 📊 Мониторинг
    
    - **MLflow**: Отслеживание экспериментов и моделей
    - **DVC**: Версионирование данных и метрик
    - **FastAPI**: Документация API по `/docs`
    
    **Примечание:** Убедитесь что все сервисы запущены!
    """)

# Футер
st.markdown("---")
st.caption(f"""
NPV Prediction App • Powered by FastAPI + Streamlit + XGBoost + DVC + MLflow
• Версия: 2.0 • {datetime.now().year}
""")
