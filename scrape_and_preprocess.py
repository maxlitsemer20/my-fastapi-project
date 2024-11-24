import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from html import unescape

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Функция для скрапинга веб-страницы
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Извлечение параграфов
        paragraphs = [unescape(paragraph.get_text()) for paragraph in soup.find_all('p')]
        if not paragraphs:
            logging.warning(f"No <p> tags found on {url}")
        else:
            logging.info(f"Found {len(paragraphs)} <p> tags on {url}")
        return paragraphs
    except requests.RequestException as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return []

# Функция для предобработки данных
def preprocess_data(data):
    df = pd.DataFrame(data, columns=['text'])
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['text'] = df['text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)  # Удаляем нежелательные символы
    df.dropna(inplace=True)
    logging.info(f"Processed {len(df)} records")
    return df

# Пример использования
if __name__ == "__main__":
    url = 'https://example.com'
    data = fetch_data(url)
    if data:
        processed_data = preprocess_data(data)
        print(processed_data.head())
    else:
        logging.warning("No data to process.")
