from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    DistilBertTokenizerFast, TFDistilBertForSequenceClassification
)


if Path('/.dockerenv').is_file():
    model_dir = Path('/app')
else:
    from src.constants import root_dir
    model_dir = root_dir / 'price-prediction/results-long'

model = TFDistilBertForSequenceClassification.from_pretrained(
    model_dir / 'tf_model.h5', config=model_dir / 'config.json'
)
tokenizer = DistilBertTokenizerFast.from_pretrained(
    'distilbert-base-cased', return_tensors='tf'
)


class Reviews(BaseModel):
    reviews: list[str]


app = FastAPI()


@app.get('/ping')
async def ping():
    return 'OK'


@app.post("/predict")
async def predict(input_text: Reviews):
    tokenized_input = tokenizer(
        input_text.reviews, truncation=True,
        padding='max_length', max_length=128,
        return_tensors='tf'
    )
    preds = model(**tokenized_input).logits.numpy()

    return {
        'raw_predictions': preds.reshape(-1).tolist(),
        'expected price': int(preds.mean().round())
    }
