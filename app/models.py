from pydantic import BaseModel

class Prediction(BaseModel):
    smiles: str
    solvent: str
    acid_base: str
    temperature: float
    pressure: float
