import streamlit as st
import requests
import pandas as pd
import json

# Настройки страницы
st.set_page_config(
    page_title="NPV Prediction App",
    page_icon="📊",
    layout="wide"
)

# Заголовок приложения
st.title("💰 NPV Prediction App")
st.markdown("Прогнозирование NPV на основе параметров скважины")

# URL вашего API (по умолчанию localhost)
API_URL = st.sidebar.text_input(
    "URL API", 
    value="http://localhost:8000",
    help="Введите URL вашего FastAPI сервера"
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

# Проверка соединения с API
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
    heff = st.number_input("Эффективная высота (Heff)", min_value=0.0, value=10.0, step=0.1)
    perm = st.number_input("Проницаемость (Perm)", min_value=0.0, value=100.0, step=1.0)
    sg = st.slider("Газонасыщенность (Sg)", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    c5 = st.number_input("Содержание C5", min_value=0.0, value=0.5, step=0.1)

with col2:
    st.subheader("Технологические параметры")
    l_hor = st.number_input("Горизонтальная длина (L_hor)", min_value=0.0, value=500.0, step=10.0)
    gs = st.selectbox("Тип проводки ствола", ["S-TYPE", "U-TYPE", "VGS", "GS", "NGS"])
    temp = st.number_input("Темп падения", min_value=0.0, value=20.0, step=0.1)
    grp = st.number_input("стадий ГРП)", min_value=0, value=1, step=1)
    ngs = st.number_input("Количество горизонтальных стволов", min_value=0, value=2, step=1)

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
        st.success(f"## Прогнозируемый NPV: **{result['predicted_NPV']:,.2f}**")
        
        # Дополнительная информация
        with st.expander("📊 Детали запроса"):
            st.json(input_data)
            st.json(result)
    else:
        st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")

# Информация о модели
st.sidebar.header("ℹ️ Информация о модели")
model_info = get_model_info()
if model_info and "error" not in model_info:
    st.sidebar.write(f"**Тип модели:** {model_info.get('model_type', 'Неизвестно')}")
    st.sidebar.write(f"**Количество признаков:** {model_info.get('n_features', 'Неизвестно')}")
    
    if "features" in model_info:
        with st.sidebar.expander("Список признаков"):
            for feature in model_info["features"]:
                st.write(f"• {feature}")
else:
    st.sidebar.info("Информация о модели недоступна")

# Примеры запросов
st.sidebar.header("📋 Примеры запросов")
if st.sidebar.button("Загрузить пример 1"):
    st.experimental_set_query_params(example=1)
    st.rerun()

if st.sidebar.button("Загрузить пример 2"):
    st.experimental_set_query_params(example=2)
    st.rerun()

# Инструкция
with st.expander("📖 Инструкция по использованию"):
    st.markdown("""
    1. **Запустите FastAPI сервер** на порту 8000
    2. **Убедитесь что API доступен** (кнопка проверки соединения)
    3. **Заполните все параметры** скважины
    4. **Нажмите 'Рассчитать NPV'** для получения прогноза
    5. **Используйте примеры** для быстрого заполнения
    
    **Примечание:** Убедитесь что ваш Docker контейнер с API запущен!
    """)

# Футер
st.markdown("---")
st.caption("NPV Prediction App • Powered by FastAPI + Streamlit + XGBoost")