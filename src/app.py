"""

Usage

>>> uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

"""

from functools import singledispatch
import json

import numpy as np
from fastapi import FastAPI
from omegaconf import DictConfig

from src.inference import EWSONNXPredictor
from src.pl_data.datamodule import MyDataModule

app = FastAPI(title="MLOps Basics App")

predictor = EWSONNXPredictor("./models/model.onnx")
predictor.set_up(
    datamodule=MyDataModule(
        datasets=None, num_workers=DictConfig({"test":4}),
        batch_size=16,
        max_length=512, tokenizer="google/bert_uncased_L-2_H-512_A-8"
    )
)

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def get_prediction(text: str):
    result = predictor.predict(text)
    result = json.dumps(result, default=to_serializable)
    return result
