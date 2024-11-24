# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл requirements.txt и устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы вашего проекта в рабочую директорию контейнера
COPY . .

# Команда для запуска вашего приложения
CMD ["uvicorn", "qa_api:app", "--host", "0.0.0.0", "--port", "80"]
