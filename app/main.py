from fastapi import FastAPI
from .models import Prediction
from .predictor import predict_mechanism
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your GitHub Pages domain
origins = [
    "https://aaronbrennan1.github.io",
    "http://localhost:3000",  # Optional, for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Mechanism Predictor API"}

@app.post("/predict")
def predict(prediction: Prediction):
    
    result = predict_mechanism(prediction.smiles, prediction.solvent, prediction.acid_base, prediction.temperature, prediction.pressure)
    return {"mechanism": result}
