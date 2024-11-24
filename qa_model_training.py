from datasets import load_dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments

# Загрузка датасета SQuAD
dataset = load_dataset("squad")

# Предпросмотр данных
print(dataset["train"][0])  # Просмотр примера

# Загрузка токенизатора для модели BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Токенизация данных
def preprocess_data(examples):
    return tokenizer(examples['question'], examples['context'], truncation=True, padding=True, max_length=512)

train_data = dataset['train'].map(preprocess_data, batched=True)
val_data = dataset['validation'].map(preprocess_data, batched=True)

# Загрузка предобученной модели BERT для QA
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Определение аргументов для обучения
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Определение тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Обучение модели
trainer.train()

# Оценка модели
results = trainer.evaluate()
print(results)
