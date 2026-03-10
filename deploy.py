from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

model = pickle.load(open("model_weights.pkl","rb"))
scaler = pickle.load(open("scaler_weights.pkl","rb"))


class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    zipcode: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float
    month: float
    year: float


@app.post("/predict")
def predict(data: HouseFeatures):

    try:

        features = np.array([[
            data.bedrooms,
            data.bathrooms,
            data.sqft_living,
            data.sqft_lot,
            data.floors,
            data.waterfront,
            data.view,
            data.condition,
            data.grade,
            data.sqft_above,
            data.sqft_basement,
            data.yr_built,
            data.yr_renovated,
            data.zipcode,
            data.lat,
            data.long,
            data.sqft_living15,
            data.sqft_lot15,
            data.month,
            data.year
        ]])

        scaled = scaler.transform(features)

        prediction = model.predict(scaled)

        return {"predicted_price": float(prediction[0][0])}

    except Exception as e:
        return {"error": str(e)}