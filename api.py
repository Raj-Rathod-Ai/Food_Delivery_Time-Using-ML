from fastapi import FastAPI, HTTPException, Security, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import uvicorn
import logging
import secrets
import json
import os
from typing import Optional, List, Dict
from datetime import datetime

# Configure detailed application logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeliveryAPI")

app = FastAPI(
    title="AI Food Delivery Predictor API",
    description="Production-ready REST API with integrated custom API Key Authorization and Daily Rate Limiting.",
    version="1.3.0"
)

# Enable CORS for cross-origin platform connectivity (Web, Flutter, React Native, iOS, Android)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# API KEY DATABASE LAYER (JSON PERSISTENT FILE)
# -------------------------------------------------------
from db_manager import load_keys_db, save_keys_db, init_keys_db, FREE_DEMO_KEY, DAILY_LIMIT_DEFAULT

# Initialize database cache
api_keys_db = init_keys_db()

# Define API key headers verification dependency
API_KEY_NAME = "X-API-Key"
api_key_header_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key_header: Optional[str] = Security(api_key_header_scheme)):
    db = load_keys_db()
    if not api_key_header:
        raise HTTPException(
            status_code=401, 
            detail=f"Authorization header '{API_KEY_NAME}' is missing. Supply a valid API Key to use this pipeline."
        )
    # Direct matching for Master Admin Key from environment variable
    env_master_key = os.environ.get("MASTER_API_KEY")
    if env_master_key and api_key_header == env_master_key:
        return api_key_header
        
    if api_key_header not in db or not db[api_key_header].get("active", False):
        raise HTTPException(
            status_code=403, 
            detail="Provided API key is invalid or has been revoked."
        )
    
    # -------------------------------------------------------
    # RATE LIMITING LOGIC (100 requests per day)
    # -------------------------------------------------------
    key_details = db[api_key_header]
    role = key_details.get("role", "client")
    
    # Admin keys are exempt from rate limiting
    if role != "admin":
        current_date = datetime.now().strftime("%Y-%m-%d")
        saved_date = key_details.get("today_date", "")
        requests_today = key_details.get("today_requests", 0)
        daily_limit = key_details.get("daily_limit", DAILY_LIMIT_DEFAULT)
        
        if saved_date == current_date:
            if requests_today >= daily_limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate Limit Exceeded: This key is capped at {daily_limit} requests per day. Try again tomorrow."
                )
            # Increment request count
            db[api_key_header]["today_requests"] = requests_today + 1
        else:
            # Reset daily counter for the new day
            db[api_key_header]["today_date"] = current_date
            db[api_key_header]["today_requests"] = 1
            
        save_keys_db(db)
        
    return api_key_header

# -------------------------------------------------------
# ML PIPELINE LOAD & CALIBRATION (With fallbacks)
# -------------------------------------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("poly.pkl", "rb") as f:
        poly = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    is_fallback = False
    logger.info("Successfully loaded ML pipeline files (model, poly, feature_names, label_encoder).")
except Exception as e:
    logger.warning(f"Pipeline pickle files missing or incompatible: {str(e)}. Initializing API Fallback simulation engine.")
    
    class FallbackModel:
        def predict(self, X):
            dist = X["Distance_km"].values[0] if "Distance_km" in X.columns else 5.0
            prep = X["Preparation_Time_min"].values[0] if "Preparation_Time_min" in X.columns else 15.0
            exp = X["Courier_Experience_yrs"].values[0] if "Courier_Experience_yrs" in X.columns else 2.0
            val = 11.5 + (dist * 2.85) + (prep * 1.1) - (exp * 0.75)
            return [max(8.0, val)]
    
    class FallbackPoly:
        def transform(self, X):
            return X

    feature_names = [
        "Order_ID", "Distance_km", "Preparation_Time_min", "Courier_Experience_yrs",
        "\tWeather_0", "\tWeather_1", "\tWeather_2", "\tWeather_3", "\tWeather_4",
        "Traffic_Level_0", "Traffic_Level_1", "Traffic_Level_2",
        "\rTime_of_Day_0", "\rTime_of_Day_1", "\rTime_of_Day_2", "\rTime_of_Day_3",
        "Vehicle_Type_0", "Vehicle_Type_1", "Vehicle_Type_2"
    ]
    model = FallbackModel()
    poly = FallbackPoly()
    label_encoder = None
    is_fallback = True

# Standard categorized vocabularies mapped alphabetically matching pandas get_dummies order
WEATHER_CATS = ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
TRAFFIC_CATS = ["High", "Low", "Medium"]
TOD_CATS = ["Afternoon", "Evening", "Morning", "Night"]
VEHICLE_CATS = ["Bike", "Car", "Scooter"]

# -------------------------------------------------------
# DATA SCHEMAS
# -------------------------------------------------------
class DeliveryRequest(BaseModel):
    order_id: int = Field(..., example=522, description="Unique numeric identifier for the order tracking reference")
    distance_km: float = Field(..., example=7.93, description="Travel distance in kilometers from vendor to customer dropoff")
    preparation_time_min: int = Field(..., example=12, description="Calculated store preparation time in minutes")
    courier_experience_yrs: float = Field(..., example=3.0, description="Courier rider work experience in years")
    weather: str = Field(..., example="Clear", description="Weather scenario: Clear, Sunny, Foggy, Rainy, Snowy, Windy")
    traffic_level: str = Field(..., example="Low", description="Traffic conditions: Low, Medium, High")
    time_of_day: str = Field(..., example="Afternoon", description="Time categorization: Morning, Afternoon, Evening, Night")
    vehicle_type: str = Field(..., example="Scooter", description="Rider transit vehicle: Bike, Scooter, Car")

class PredictionResponse(BaseModel):
    order_id: int
    predicted_delivery_time_minutes: float
    pipeline_mode: str
    timestamp: str

class KeyGenRequest(BaseModel):
    owner: str = Field(..., example="UberEats Team", description="Name of external system or team receiving this credential")

class KeyGenResponse(BaseModel):
    api_key: str
    owner: str
    created_at: str
    active: bool

class RevokeRequest(BaseModel):
    api_key: str

# -------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------

# API Status Endpoint
@app.get("/api/status")
def api_status():
    return {
        "status": "online",
        "api_name": "AI Food Delivery Predictor Engine API",
        "security": "API Key Required (100 req/day limit)",
        "accuracy": "81.0% R² Fitted",
        "pipeline_fallback_active": is_fallback,
        "swagger_docs": "/docs"
    }

# Predict Endpoint Protected by verify_api_key dependency
@app.post("/predict", response_model=PredictionResponse)
def predict_delivery_time(payload: DeliveryRequest, api_key: str = Depends(verify_api_key)):
    try:
        # 1. Initialize empty prediction template mapping directly to model target columns
        processed_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
        
        # 2. Cap continuous variable parameters to training distribution ranges to prevent ML extrapolation errors
        capped_order_id = max(1.0, min(1000.0, float(payload.order_id)))
        capped_distance = max(0.5, min(20.0, float(payload.distance_km)))
        capped_prep_time = max(5.0, min(60.0, float(payload.preparation_time_min)))
        capped_experience = max(0.0, min(9.0, float(payload.courier_experience_yrs)))

        processed_df.at[0, "Order_ID"] = capped_order_id
        processed_df.at[0, "Distance_km"] = capped_distance
        processed_df.at[0, "Preparation_Time_min"] = capped_prep_time
        processed_df.at[0, "Courier_Experience_yrs"] = capped_experience
        
        # 3. Clean and sanitize user textual string inputs (case-insensitive & synonyms)
        input_weather = payload.weather.strip().title()
        if input_weather == "Sunny":
            input_weather = "Clear"
        elif "Sunny" in input_weather or "Clear" in input_weather:
            input_weather = "Clear"
            
        input_traffic = payload.traffic_level.strip().title()
        input_tod = payload.time_of_day.strip().title()
        input_vehicle = payload.vehicle_type.strip().title()

        # 4. Map weather dummies, supporting both with and without tab prefix
        if input_weather in WEATHER_CATS:
            w_idx = WEATHER_CATS.index(input_weather)
            for col in [f"\tWeather_{w_idx}", f"Weather_{w_idx}"]:
                if col in feature_names:
                    processed_df.at[0, col] = 1.0

        # 5. Map traffic level dummies
        if input_traffic in TRAFFIC_CATS:
            t_idx = TRAFFIC_CATS.index(input_traffic)
            for col in [f"Traffic_Level_{t_idx}"]:
                if col in feature_names:
                    processed_df.at[0, col] = 1.0

        # 6. Map time of day dummies, supporting both with and without carriage return prefix
        if input_tod in TOD_CATS:
            tod_idx = TOD_CATS.index(input_tod)
            for col in [f"\rTime_of_Day_{tod_idx}", f"Time_of_Day_{tod_idx}"]:
                if col in feature_names:
                    processed_df.at[0, col] = 1.0

        # 7. Map vehicle type dummies
        if input_vehicle in VEHICLE_CATS:
            v_idx = VEHICLE_CATS.index(input_vehicle)
            for col in [f"Vehicle_Type_{v_idx}"]:
                if col in feature_names:
                    processed_df.at[0, col] = 1.0

        # 8. Transform to multi-variable polynomial degree feature expansion
        X_model_input = poly.transform(processed_df)

        # 9. Feed the preprocessed vector through the predictive model estimator
        prediction_array = model.predict(X_model_input)
        prediction_val = float(prediction_array[0])
        
        # Guard against NaN or Infinite predictions
        if np.isnan(prediction_val) or np.isinf(prediction_val):
            prediction_val = 56.73

        # Set realistic safety boundaries matching the dataset range (8 min to 150 min)
        final_prediction = max(8.0, min(150.0, prediction_val))

        return PredictionResponse(
            order_id=payload.order_id,
            predicted_delivery_time_minutes=round(final_prediction, 2),
            pipeline_mode="Fallback Predictor" if is_fallback else "Polynomial Regression (Degree-2)",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Inference pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference processing failed internally: {str(e)}")

# -------------------------------------------------------
# API KEY MANAGEMENT ENDPOINTS
# -------------------------------------------------------

# Public request endpoint (guarantees uniqueness and rate limiting tags)
@app.post("/api-key/request", response_model=KeyGenResponse)
def request_api_key(payload: KeyGenRequest):
    db = load_keys_db()
    
    # Generate a guaranteed unique API token
    while True:
        new_token = "DELIVERY-" + secrets.token_hex(16).upper()
        if new_token not in db:
            break
            
    db[new_token] = {
        "owner": payload.owner.strip(),
        "created_at": datetime.now().isoformat(),
        "active": True,
        "role": "client",
        "today_date": datetime.now().strftime("%Y-%m-%d"),
        "today_requests": 0,
        "daily_limit": DAILY_LIMIT_DEFAULT
    }
    save_keys_db(db)
    
    return KeyGenResponse(
        api_key=new_token,
        owner=payload.owner,
        created_at=db[new_token]["created_at"],
        active=True
    )

# Admin Secured key creation
@app.post("/api-key/generate", response_model=KeyGenResponse)
def generate_api_key(payload: KeyGenRequest, api_key: str = Depends(verify_api_key)):
    db = load_keys_db()
    if db.get(api_key, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin permissions required to generate API tokens.")
    
    while True:
        new_token = "DELIVERY-" + secrets.token_hex(16).upper()
        if new_token not in db:
            break
            
    db[new_token] = {
        "owner": payload.owner.strip(),
        "created_at": datetime.now().isoformat(),
        "active": True,
        "role": "client",
        "today_date": datetime.now().strftime("%Y-%m-%d"),
        "today_requests": 0,
        "daily_limit": DAILY_LIMIT_DEFAULT
    }
    save_keys_db(db)
    
    return KeyGenResponse(
        api_key=new_token,
        owner=payload.owner,
        created_at=db[new_token]["created_at"],
        active=True
    )

# Admin Secured list endpoint
@app.get("/api-key/list")
def list_api_keys(api_key: str = Depends(verify_api_key)):
    db = load_keys_db()
    if db.get(api_key, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin permissions required to view key logs.")
    
    return [
        {
            "api_key": key,
            "owner": details["owner"],
            "created_at": details["created_at"],
            "active": details["active"],
            "role": details.get("role", "client")
        }
        for key, details in db.items()
    ]

# Admin Secured revoke endpoint
@app.post("/api-key/revoke")
def revoke_api_key(payload: RevokeRequest, api_key: str = Depends(verify_api_key)):
    db = load_keys_db()
    if db.get(api_key, {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin permissions required to revoke tokens.")
    
    target_key = payload.api_key
    if target_key not in db:
        raise HTTPException(status_code=404, detail="Requested token does not exist.")
    if db[target_key].get("role") == "admin":
        raise HTTPException(status_code=400, detail="Cannot revoke system administrator credentials.")
        
    db[target_key]["active"] = False
    save_keys_db(db)
    return {"message": "Credentials revoked successfully.", "revoked_key": target_key}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    production = os.environ.get("ENV", "development").lower() == "production"
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=not production)
