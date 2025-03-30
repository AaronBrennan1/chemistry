from fastapi import FastAPI
from .models import Prediction
from .predictor import predict_mechanism

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Mechanism Predictor API"}

@app.post("/predict")
def predict(prediction: Prediction):
    
    result = predict_mechanism(prediction.smiles, prediction.solvent, prediction.acid_base, prediction.temperature, prediction.pressure)
    return {"mechanism": result}
