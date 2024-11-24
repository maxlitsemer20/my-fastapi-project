from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scrape_and_preprocess import preprocess_data  # Добавьте импорт функции предобработки данных
from train_and_update_model import update_model, build_model  # Добавьте импорт функции обновления модели и создания модели

# Загрузка модели и токенизатора
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем pad_token

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/generate")
def generate_text(query: Query):
    inputs = tokenizer(query.text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs['input_ids'], 
        max_length=150, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        temperature=0.7, 
        top_k=50
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

@app.post("/update")
def update_model_endpoint(query: Query):
    new_data = [query.text]
    processed_new_data = preprocess_data(new_data)
    if not processed_new_data.empty:
        global model  # Сделать модель глобальной, чтобы обновлять ее внутри функции
        model = update_model(model, tokenizer, processed_new_data['text'])
        return {"status": "Model updated successfully"}
    else:
        return {"status": "No new data to update the model"}

# Новый endpoint для загрузки данных через файл
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    new_data = [contents.decode('utf-8')]
    processed_new_data = preprocess_data(new_data)
    if not processed_new_data.empty:
        global model
        model = update_model(model, tokenizer, processed_new_data['text'])
        return {"status": "Model updated successfully with uploaded data"}
    else:
        return {"status": "No new data to update the model"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Используйте доступный порт
