import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import Dict, Any
import os

wdir = os.path.dirname(__file__)


def load_model():
    """
    Cargar modelo serializado de LightFM
    Retorna modelo deserializado
    """
    path = os.path.join(wdir, "ml/models/trained_model_lightfm-sample-1.0.pkl")
    model = pickle.load(open(path, "rb"))
    return model


def load_mapping():
    """
    Cargar mapping de users e items ids externos e internos de LightFM
    Retorna diccionarios des-serializados
    """
    path = os.path.join(wdir, "ml/models/mapping.pkl")
    mapping = pickle.load(open(path, "rb"))
    users_ext_int = mapping["users"]  # external to internal
    items_ext_int = mapping["items"]  # external to internal
    items_int_ext = {value: key for key, value in items_ext_int.items()}

    return users_ext_int, items_int_ext


model = load_model()
users, items = load_mapping()


class UserRequest(BaseModel):
    id: int


class RecosResponse(BaseModel):
    rut: int
    recommendations: Dict[Any, Any]


app = FastAPI()


@app.post("/recommendations", response_model=RecosResponse)
def get_recommendations(user_request: UserRequest):
    user_id = users[user_request.id]
    scores = model.predict(user_id, np.arange(len(items)))
    scores = np.argsort(-scores)[:8]
    recos = {i: str(items[scores[i]]) for i in range(len(scores))}
    recos = {"rut": user_request.id, "recommendations": recos}
    return recos


@app.post("/")
def hello():
    return {"message": "Bienvenido al Sistema Recomendador v. 1.0 beta"}
