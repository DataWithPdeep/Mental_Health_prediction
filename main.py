from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(title="NYC House Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood",
]

# -------------------------
# Load Model
# -------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "Model_Pipeline.pkl"
)

model = joblib.load(MODEL_PATH)


# -------------------------
# Pydantic Input Model
# -------------------------

class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    price: float = Field(..., ge=0)
    minimum_nights: int = Field(..., ge=0)
    number_of_reviews: int = Field(..., ge=0)
    reviews_per_month: float = Field(..., ge=0)
    calculated_host_listings_count: int = Field(..., ge=0)
    availability_365: int = Field(..., ge=0, le=365)
    neighbourhood_group: str
    neighbourhood: str


# -------------------------
# Frontend
# -------------------------

@app.get("/")
def home():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "index.html")
    )


# -------------------------
# API Health Check
# -------------------------

@app.get("/api")
def api_home():
    return {
        "message": "NYC House Classification API is Running Successfully 🚀"
    }


# -------------------------
# Prediction API
# -------------------------

@app.post("/predict")
def predict(features: Features):

    try:
        row = pd.DataFrame(
            [features.model_dump()],
            columns=COLUMNS
        )

        prediction = model.predict(row)[0]

        probability = model.predict_proba(row)[0].tolist()

        return {
            "Predicted_room_type": prediction,
            "Probability": probability
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -------------------------
# Static Files
# -------------------------

app.mount(
    "/static",
    StaticFiles(directory=os.path.dirname(__file__)),
    name="static"
)
