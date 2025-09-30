FROM python:3.9-slim

WORKDIR /app

# Установка DVC и git (необходимо для работы с DVC)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install dvc

# Копирование исходного кода
COPY . .

# Инициализация DVC (для продакшена можно использовать remote storage)
RUN dvc init --no-scm

# Экспорт порта
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]