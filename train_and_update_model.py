import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from html import unescape
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Функция для скрапинга веб-страницы
def fetch_data(url):
    try:
        response = requests.get(url, timeout=10)  # Увеличьте значение таймаута
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

# Класс для датасета
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encodings.input_ids.squeeze(), encodings.attention_mask.squeeze()

# Функция для создания модели и токенизатора
def build_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем pad_token
    return model, tokenizer

# Функция для обучения модели
def train_model(model, tokenizer, train_data):
    train_dataset = TextDataset(train_data, tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(3):  # Пример обучения на 3 эпохи
        for batch in train_loader:
            inputs, attention_masks = batch
            outputs = model(input_ids=inputs, attention_mask=attention_masks, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logging.info(f"Epoch {epoch+1} complete. Loss: {loss.item()}")

    return model

# Функция для обновления модели
def update_model(model, tokenizer, new_data):
    new_dataset = TextDataset(new_data, tokenizer, max_length=128)
    if len(new_dataset) == 0:
        logging.warning("No new data to update the model.")
        return model
    
    new_loader = DataLoader(new_dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(1):  # Пример обновления на 1 эпоху
        for batch in new_loader:
            inputs, attention_masks = batch
            outputs = model(input_ids=inputs, attention_mask=attention_masks, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logging.info(f"Model updated. Loss: {loss.item()}")

    return model

# Основной блок кода
if __name__ == "__main__":
    from scrape_and_preprocess import fetch_data, preprocess_data
    
    # Получение и предобработка данных
    urls = [
        'https://example.com',
        'https://another-valid-url.com',  # Замените на доступные URL
        'https://yet-another-valid-url.com'
    ]

    data = []
    for url in urls:
        data.extend(fetch_data(url))

    processed_data = preprocess_data(data)
    print(processed_data.head())
    
    # Обучение модели
    model, tokenizer = build_model()
    model = train_model(model, tokenizer, processed_data['text'])
    
    # Для переобучения с новыми данными:
    new_data = fetch_data('https://new-example.com')
    processed_new_data = preprocess_data(new_data)
    if not processed_new_data.empty:
        model = update_model(model, tokenizer, processed_new_data['text'])
    else:
        logging.warning("No new data to update the model.")
