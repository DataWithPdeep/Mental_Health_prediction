import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Mental_Health_Model.pkl"
)

INDEX_PATH = os.path.join(
    BASE_DIR,
    "index.html"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    print("Mental Health model loaded successfully.")

except Exception as e:
    model = None
    print(f"Error loading model: {e}")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Mental Health Signal API",
    description="Student Wellness Analytics Prediction API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC INPUT MODEL
# ============================================================

class StudentData(BaseModel):

    age: int = Field(
        ...,
        ge=10,
        le=100
    )

    gender: Literal[
        "Male",
        "Female"
    ]

    country: str

    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]

    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]

    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]

    avg_daily_usage_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    daily_unlocks: int = Field(
        ...,
        ge=0
    )

    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    physical_activity_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    sleep_hours_per_night: float = Field(
        ...,
        ge=0,
        le=24
    )

    stress_level: Literal[
        "Medium",
        "Low",
        "Very High",
        "High"
    ]


# ============================================================
# RESPONSE MODEL
# ============================================================

class PredictionResponse(BaseModel):

    predicted_mental_health_score: float


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():

    if not os.path.exists(INDEX_PATH):
        raise HTTPException(
            status_code=404,
            detail="index.html not found"
        )

    return FileResponse(INDEX_PATH)


# ============================================================
# API HEALTH CHECK
# ============================================================

@app.get("/api")
def health_check():

    return {
        "status": "success",
        "message": "Welcome to Shreyan AI",
        "model_loaded": model is not None
    }


# ============================================================
# COUNTRY GROUPING
# ============================================================

top_countries = [
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
]


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: StudentData):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Mental Health model is not loaded."
        )

    try:

        # Group country
        country_group = (
            data.country
            if data.country in top_countries
            else "Other"
        )

        # Create input DataFrame
        input_row = pd.DataFrame([
            {
                "Age": data.age,
                "Gender": data.gender,
                "Country": data.country,
                "Academic_Level": data.academic_level,
                "Most_Used_Platform": data.most_used_platform,
                "Purpose_Of_Use": data.purpose_of_use,
                "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
                "Daily_Unlocks": data.daily_unlocks,
                "Study_Hours": data.study_hours,
                "Physical_Activity_Hours": data.physical_activity_hours,
                "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
                "Stress_Level": data.stress_level,

                # Keep this spelling exactly as used
                # by your trained model
                "Gouped_Country": country_group
            }
        ])

        # Prediction
        prediction = model.predict(input_row)[0]

        return PredictionResponse(
            predicted_mental_health_score=round(
                float(prediction),
                2
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR),
    name="static"
)
