from transformers import BertTokenizer, BertForQuestionAnswering

# Загрузка токенизатора и модели
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Определение функции для ответа на вопросы
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)

    # Получение позиций начала и конца ответа
    start = outputs.start_logits.argmax()
    end = outputs.end_logits.argmax()

    # Конвертация токенов обратно в текст
    answer_tokens = inputs.input_ids[0][start:end+1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

# Пример использования
if __name__ == "__main__":
    context = "Hugging Face - это компания, специализирующаяся на обработке естественного языка."
    question = "На чем специализируется Hugging Face?"
    answer = answer_question(question, context)
    print(f"Ответ: {answer}")
