import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextGenerator:
    def __init__(self, pretrained_model="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qa_pairs = {
            "Что такое Python?": "Python - это высокоуровневый интерпретируемый язык программирования, известный своей читаемостью и универсальностью.",
            "Что такое функция?": "Функция - это блок кода, который выполняется только при вызове. Она может принимать входные данные и возвращать результаты.",
            "Что такое класс?": "Класс - это шаблон для создания объектов в объектно-ориентированном программировании. Он инкапсулирует данные для объекта.",
            "Что такое API?": "API (Application Programming Interface) - это набор правил, который позволяет различным программным системам взаимодействовать друг с другом.",
            "Что такое машинное обучение?": "Машинное обучение - это подмножество искусственного интеллекта, которое включает использование алгоритмов для обучения компьютеров на основе данных и предсказания на их основе.",
            "Сколько будет 2+2?": "2+2 равно 4.",
            "Сколько будет 3+3?": "3+3 равно 6."
        }

    def add_qa_pair(self, question: str, answer: str):
        if not question or answer.strip() == "":
            raise ValueError("И вопрос, и ответ должны быть непустыми строками.")
        self.qa_pairs[question] = answer

    def train_model(self, data, epochs=3, batch_size=4):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                inputs = self.tokenizer(data[i:i+batch_size], return_tensors='pt', padding=True, truncation=True)
                attention_mask = inputs.pop("attention_mask")
                outputs = self.model(**inputs, attention_mask=attention_mask, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i % 100 == 0:
                    print(f'Эпоха: {epoch}, Шаг: {i}, Потеря: {loss.item()}')

    def generate_answer(self, question: str) -> str:
        greetings = ["здравствуйте", "привет", "добрый день", "доброе утро", "добрый вечер", "hello", "hi", "greetings", "hey"]
        if any(word in question.lower() for word in greetings):
            return random.choice([
                "Привет! Как я могу вам помочь?",
                "Здравствуйте! Чем могу помочь?",
                "Добрый день! Что вас интересует?",
                "Доброе утро! Как я могу вам помочь?",
                "Добрый вечер! Что бы вы хотели обсудить?"
            ])
        if question in self.qa_pairs:
            return self.qa_pairs[question]
        else:
            try:
                inputs = self.tokenizer.encode(question, return_tensors='pt')
                attention_mask = (inputs != self.tokenizer.pad_token_id).long()
                outputs = self.model.generate(inputs, attention_mask=attention_mask, max_length=100, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                return f"Произошла ошибка при генерации ответа: {str(e)}"

# Example usage of TextGenerator
if __name__ == "__main__":
    text_generator = TextGenerator()
    text_generator.add_qa_pair("Что такое искусственный интеллект?", "Искусственный интеллект (ИИ) - это моделирование процессов человеческого интеллекта машинами, особенно компьютерными системами.")
    text_generator.add_qa_pair("Как работает интернет?", "Интернет - это глобальная сеть взаимосвязанных компьютеров, которые общаются с помощью стандартных протоколов для обмена данными.")
    
    # Example training data
    data = [
        "Пример текста для обучения модели.",
        "Модель должна обучаться на разнообразных данных.",
        "Это помогает улучшить качество генерируемого текста."
    ]
    
    text_generator.train_model(data)
