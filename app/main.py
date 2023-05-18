from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predictOutput

app = FastAPI()

class url(BaseModel):
    url: str

class Prediction(BaseModel):
    prediction: list

@app.get("/")
def home():
    return {"Test":"OK"}

@app.post("/predict",response_model=Prediction)
def predict(payload: url):
    prediction = predictOutput(payload.url)
    return {"prediction":prediction}



