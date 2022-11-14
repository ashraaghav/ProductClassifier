import pickle
import os
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI
from uvicorn import run

from lib.embedder import Embedder
from lib.dataset import clean_and_truncate

app = FastAPI(
    title="Category Classifier API",
    description="API for product category classification",
    version="1.0"
)

# load models & embedders once
with open('./results/category_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./results/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
embedder = Embedder(max_len=25)


@app.get("/")
async def root():
    return {"message": "Welcome to the Category Classifier API!"}


@app.post('/predict', tags=["predictions"])
async def get_prediction(text_json):
    data_json = json.loads(text_json)

    if ('manufacturer' not in data_json.keys()) and \
            ('add_text' not in data_json.keys()) and \
            ('main_text' not in data_json.keys()):
        return {'prediction': '',
                'message': 'no valid data provided'}

    # format data
    data_1 = pd.Series(data_json.get('manufacturer', np.nan)).apply(lambda x: clean_and_truncate(x, 1))
    data_2 = pd.Series(data_json.get('add_text', np.nan)).apply(lambda x: clean_and_truncate(x, 2))
    data_3 = pd.Series(data_json.get('main_text', np.nan)).apply(lambda x: clean_and_truncate(x, 20))

    texts = data_1 + ' ' + data_2 + ' ' + data_3
    texts = texts.to_list()

    X = embedder.transform(texts)
    X = X.reshape(X.shape[0], -1)

    # get predictions and provide correct labels
    y_pred = model.predict(X)
    predictions = encoder.inverse_transform(y_pred).tolist()

    return {"prediction": predictions,
            "message": "success"}


def main():
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
