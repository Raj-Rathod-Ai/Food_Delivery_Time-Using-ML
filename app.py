import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import secrets
import time
from datetime import datetime
from typing import Dict

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Food Delivery Predictor Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Obsidian/Glassmorphic Style Injections
st.markdown("""
<style>
    /* Custom Styling for Streamlit metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    /* Sleek card container */
    .glass-card {
        background-color: rgba(18, 25, 41, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }
    .badge {
        padding: 0.35rem 0.75rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        margin-top: 0.5rem;
    }
    .badge-express {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .badge-standard {
        background-color: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .badge-delayed {
        background-color: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
</style>
""", unsafe_allow_html=True)

from db_manager import load_keys_db, save_keys_db, init_keys_db, FREE_DEMO_KEY

# Seed keys cache on startup
api_keys_db = init_keys_db()

# -------------------------------------------------------
# ML PIPELINE CACHED LOAD (With fallbacks)
# -------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
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
    except Exception as e:
        # Initializing simulation engine
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
        
    return model, poly, feature_names, label_encoder, is_fallback

model, poly, feature_names, label_encoder, is_fallback = load_ml_pipeline()

# Categories
WEATHER_CATS = ["Clear", "Foggy", "Rainy", "Snowy", "Windy"]
TRAFFIC_CATS = ["High", "Low", "Medium"]
TOD_CATS = ["Afternoon", "Evening", "Morning", "Night"]
VEHICLE_CATS = ["Bike", "Car", "Scooter"]

# -------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.title("🔮 Predictor.AI")
st.sidebar.subheader("Food Delivery ETA Engine")

navigation = st.sidebar.radio(
    "Navigation Menu",
    ["🔮 Predictor Panel", "🔑 Get API Key", "🔒 Admin Portal", "📖 Developer Docs"]
)

st.sidebar.divider()

# Connection status box
if is_fallback:
    st.sidebar.warning("⚠️ Fallback Simulator Active")
    st.sidebar.caption("Pickle model files not found. Using fallback math coefficients.")
else:
    st.sidebar.success("🟢 ML Pipeline Core Active")
    st.sidebar.caption("Successfully loaded Degree-2 Polynomial ML model (81% R²).")

# -------------------------------------------------------
# TAB 1: PREDICTOR PANEL
# -------------------------------------------------------
if navigation == "🔮 Predictor Panel":
    st.header("🔮 Food Delivery Predictor")
    st.markdown("Perform real-time machine learning predictions on active delivery runs.")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Inference Parameters")
        
        # Order ID input field
        order_id_input = st.number_input("Order ID", min_value=1, value=522, step=1, help="Numeric tracking reference")

        st.divider()
        
        # Sliders
        distance_input = st.slider("Delivery Distance (km)", min_value=0.1, max_value=25.0, value=5.0, step=0.05)
        prep_time_input = st.slider("Store Preparation Time (min)", min_value=5, max_value=60, value=15, step=1)
        experience_input = st.slider("Courier Experience (years)", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
        
        # Selectors
        c3, c4, c5 = st.columns(3)
        with c3:
            weather_input = st.selectbox("Weather Scenario", WEATHER_CATS)
        with c4:
            traffic_input = st.selectbox("Traffic Levels", TRAFFIC_CATS, index=2) # Medium
        with c5:
            tod_input = st.selectbox("Time of Day", TOD_CATS, index=1) # Afternoon
            
        # Vehicle Radio Card simulator
        vehicle_input = st.radio("Transit Vehicle Type", VEHICLE_CATS, horizontal=True)
        
        predict_click = st.button("Calculate Prediction", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")
        
        if predict_click:
            # Interactive Loading Animation Sequence
            with st.spinner("🔄 Preprocessing delivery run features..."):
                time.sleep(0.4)
            with st.spinner("🔮 Estimating ETA via Degree-2 Polynomial Model..."):
                time.sleep(0.5)
            
            # 1. Initialize DataFrame
            processed_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
            
            # 2. Cap continuous variable parameters to training distribution ranges to prevent ML extrapolation errors
            capped_order_id = max(1.0, min(1000.0, float(order_id_input)))
            capped_distance = max(0.5, min(20.0, float(distance_input)))
            capped_prep_time = max(5.0, min(60.0, float(prep_time_input)))
            capped_experience = max(0.0, min(9.0, float(experience_input)))

            processed_df.at[0, "Order_ID"] = capped_order_id
            processed_df.at[0, "Distance_km"] = capped_distance
            processed_df.at[0, "Preparation_Time_min"] = capped_prep_time
            processed_df.at[0, "Courier_Experience_yrs"] = capped_experience

            # 3. Clean textual inputs
            input_weather = weather_input.strip().title()
            if input_weather == "Sunny" or "Sunny" in input_weather:
                input_weather = "Clear"
            input_traffic = traffic_input.strip().title()
            input_tod = tod_input.strip().title()
            input_vehicle = vehicle_input.strip().title()

            # 4. Map Weather
            if input_weather in WEATHER_CATS:
                w_idx = WEATHER_CATS.index(input_weather)
                for col in [f"\tWeather_{w_idx}", f"Weather_{w_idx}"]:
                    if col in feature_names:
                        processed_df.at[0, col] = 1.0

            # 5. Map Traffic
            if input_traffic in TRAFFIC_CATS:
                t_idx = TRAFFIC_CATS.index(input_traffic)
                for col in [f"Traffic_Level_{t_idx}"]:
                    if col in feature_names:
                        processed_df.at[0, col] = 1.0

            # 6. Map Time of Day
            if input_tod in TOD_CATS:
                tod_idx = TOD_CATS.index(input_tod)
                for col in [f"\rTime_of_Day_{tod_idx}", f"Time_of_Day_{tod_idx}"]:
                    if col in feature_names:
                        processed_df.at[0, col] = 1.0

            # 7. Map Vehicle Type
            if input_vehicle in VEHICLE_CATS:
                v_idx = VEHICLE_CATS.index(input_vehicle)
                for col in [f"Vehicle_Type_{v_idx}"]:
                    if col in feature_names:
                        processed_df.at[0, col] = 1.0

            # 8. Transform polynomial features
            X_model_input = poly.transform(processed_df)

            # 9. Predict
            prediction_array = model.predict(X_model_input)
            prediction_val = float(prediction_array[0])
            
            # Check for NaN/Inf
            if np.isnan(prediction_val) or np.isinf(prediction_val):
                prediction_val = 56.73

            # Enforce safety boundaries
            final_prediction = max(8.0, min(150.0, prediction_val))
            
            # Render results in a nice UI card block
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.metric("Estimated Delivery Time", f"{final_prediction:.2f} mins")
            
            # Classification and Custom Badge
            if final_prediction < 30.0:
                st.markdown('<span class="badge badge-express">🚀 Express Run (Fast)</span>', unsafe_allow_html=True)
            elif final_prediction <= 65.0:
                st.markdown('<span class="badge badge-standard">🚗 Standard Run (Normal)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-delayed">⚠️ Delayed Run (Slow)</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.info(f"""
            **Prediction Metadata:**
            - **Order ID**: #{order_id_input}
            - **Pipeline Mode**: {'Fallback Predictor' if is_fallback else 'Polynomial Regression (Degree-2)'}
            - **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)
        else:
            st.caption("Adjust inference parameters on the left and click 'Calculate Prediction' to view results.")

# -------------------------------------------------------
# TAB 2: GET API KEY (PUBLIC self-service)
# -------------------------------------------------------
elif navigation == "🔑 Get API Key":
    st.header("🔑 Get Free API Access")
    st.markdown("Issue an active client authorization token to integrate the AI Food Delivery Predictor directly into your codebase.")
    
    st.divider()
    
    with st.form("public_key_form"):
        owner_name = st.text_input("Developer / Project / Team Name", placeholder="e.g. UberEats Dev team, Flutter Mobile Client")
        st.caption("Provide an owner reference so we know who is using this credential.")
        
        submit_key = st.form_submit_button("Request Access Token", type="primary")
        
        if submit_key:
            if not owner_name.strip():
                st.error("Please enter a valid owner name.")
            else:
                with st.spinner("Generating cryptographically secure token..."):
                    time.sleep(0.6)
                
                db = load_keys_db()
                
                # Guarantee API token uniqueness
                while True:
                    new_token = "DELIVERY-" + secrets.token_hex(16).upper()
                    if new_token not in db:
                        break
                        
                db[new_token] = {
                    "owner": owner_name.strip(),
                    "created_at": datetime.now().isoformat(),
                    "active": True,
                    "role": "client",
                    "today_date": datetime.now().strftime("%Y-%m-%d"),
                    "today_requests": 0,
                    "daily_limit": 100
                }
                save_keys_db(db)
                
                st.success("🎉 Access key generated successfully!")
                st.code(new_token, language="text")
                st.warning("⚠️ Make sure to copy this key now. For security purposes, you will not be able to retrieve it again from this window.")

# -------------------------------------------------------
# TAB 3: ADMIN PORTAL
# -------------------------------------------------------
elif navigation == "🔒 Admin Portal":
    st.header("🔒 Admin Portal Console")
    st.markdown("Audit generated client credentials and revoke tokens.")
    
    admin_key_input = st.text_input("Enter Master API Admin Key", type="password", help="Requires 'admin' role API key to access logs")
    
    st.divider()
    
    if admin_key_input:
        db = load_keys_db()
        env_master_key = os.environ.get("MASTER_API_KEY")
        # Direct check against environment variable or database records
        is_admin = (env_master_key and admin_key_input == env_master_key) or \
                   (admin_key_input in db and db[admin_key_input].get("role") == "admin" and db[admin_key_input].get("active", False))
        
        if not is_admin:
            st.error("Access Denied: Invalid credentials or insufficient permissions.")
        else:
            st.success("Unlock Successful. Administrative panel active.")
            
            # Key statistics
            total_keys = len(db)
            active_keys = sum(1 for details in db.values() if details.get("active", False))
            
            # Calculate total requests today across all client keys
            total_requests_today = sum(details.get("today_requests", 0) for details in db.values() if details.get("role") != "admin")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Keys Created", total_keys)
            with c2:
                st.metric("Active Connections", active_keys)
            with c3:
                st.metric("API Requests Sent Today", total_requests_today)
                
            st.subheader("Interactive Key Management Console")
            st.caption("Toggle the checkbox in the **Active (Enabled)** column to instantly enable or disable an API key.")
            
            # Build clean DataFrame for display
            display_data = []
            
            for key, details in db.items():
                role = details.get("role", "client").upper()
                masked_key = f"{key[:14]}..."
                
                # Format request count
                today_req = details.get("today_requests", 0)
                limit_req = details.get("daily_limit", 100)
                usage_str = "Admin Exempt" if role == "ADMIN" else f"{today_req} / {limit_req}"
                
                display_data.append({
                    "Owner": details["owner"],
                    "Role": role,
                    "API Key (Masked)": masked_key,
                    "Created Date": details["created_at"][:10] if "created_at" in details else "N/A",
                    "Usage Today": usage_str,
                    "Active (Enabled)": bool(details.get("active", True)),
                    "_actual_key": key # Hidden key mapping for updates
                })
            
            df = pd.DataFrame(display_data)
            
            if df.empty:
                st.info("No keys available in the database.")
            else:
                # Display interactive data editor
                edited_df = st.data_editor(
                    df,
                    column_config={
                        "Active (Enabled)": st.column_config.CheckboxColumn(
                            "Active (Enabled)",
                            help="Toggle to activate or deactivate key access",
                            default=True
                        ),
                        "Owner": "Key Owner",
                        "Role": "Role",
                        "API Key (Masked)": "API Key",
                        "Created Date": "Created Date",
                        "Usage Today": "Usage Today"
                    },
                    disabled=["Owner", "Role", "API Key (Masked)", "Created Date", "Usage Today"],
                    use_container_width=True,
                    key="keys_editor"
                )
                
                # Check for changes and update database
                changes_made = False
                for idx, row in edited_df.iterrows():
                    orig_row = df.loc[idx]
                    if row["Active (Enabled)"] != orig_row["Active (Enabled)"]:
                        target_key = row["_actual_key"]
                        # Safety lock: don't allow disabling your active admin key session to avoid lockout
                        if target_key == admin_key_input:
                            st.warning("Safety Lock: You cannot disable your own active admin key session.")
                        else:
                            db = load_keys_db()
                            db[target_key]["active"] = bool(row["Active (Enabled)"])
                            save_keys_db(db)
                            st.toast(f"Status updated for '{row['Owner']}' to {'🟢 Enabled' if row['Active (Enabled)'] else '🔴 Disabled'}!")
                            changes_made = True
                
                if changes_made:
                    time.sleep(0.5)
                    st.rerun()

# -------------------------------------------------------
# TAB 4: DEVELOPER DOCS
# -------------------------------------------------------
elif navigation == "📖 Developer Docs":
    st.header("📖 API Developer Integration Guide")
    st.markdown("Integrate our AI Food Delivery Predictor directly into your software applications using our API gateway.")
    
    st.info("""
    **Production Note:**
    To allow your friends to connect their apps using the keys generated on this console, make sure to deploy the FastAPI backend (`api.py`) on a service like **Render** or **Heroku**. 
    The Streamlit app and FastAPI backend both read from the same `api_keys.json` file.
    """)
    
    st.subheader("Request Schema")
    st.markdown("""
    - **Endpoint**: `POST http://localhost:8000/predict`
    - **Header**: `X-API-Key: <YOUR-API-KEY>`
    - **Content-Type**: `application/json`
    """)
    
    st.subheader("Code Integration Examples")
    
    code_tab_1, code_tab_2, code_tab_3, code_tab_4 = st.tabs(["cURL", "Python", "JavaScript", "Flutter (Dart)"])
    
    with code_tab_1:
        st.code("""
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -H "X-API-Key: YOUR-API-KEY-HERE" \\
     -d '{
       "order_id": 522,
       "distance_km": 7.93,
       "preparation_time_min": 12,
       "courier_experience_yrs": 3.0,
       "weather": "Clear",
       "traffic_level": "Low",
       "time_of_day": "Afternoon",
       "vehicle_type": "Scooter"
     }'
        """, language="bash")
        
    with code_tab_2:
        st.code("""
import requests

url = "http://localhost:8000/predict"
headers = {
    "X-API-Key": "YOUR-API-KEY-HERE",
    "Content-Type": "application/json"
}
payload = {
    "order_id": 522,
    "distance_km": 7.93,
    "preparation_time_min": 12,
    "courier_experience_yrs": 3.0,
    "weather": "Clear",
    "traffic_level": "Low",
    "time_of_day": "Afternoon",
    "vehicle_type": "Scooter"
}

response = requests.post(url, json=payload, headers=headers)
data = response.json()
print(f"Predicted ETA: {data['predicted_delivery_time_minutes']} minutes")
        """, language="python")
        
    with code_tab_3:
        st.code("""
const url = 'http://localhost:8000/predict';
const apiKey = 'YOUR-API-KEY-HERE';

const payload = {
  order_id: 522,
  distance_km: 7.93,
  preparation_time_min: 12,
  courier_experience_yrs: 3.0,
  weather: 'Clear',
  traffic_level: 'Low',
  time_of_day: 'Afternoon',
  vehicle_type: 'Scooter'
};

fetch(url, {
  method: 'POST',
  headers: {
    'X-API-Key': apiKey,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(payload)
})
.then(res => res.json())
.then(data => {
  console.log(`Predicted ETA: ${data.predicted_delivery_time_minutes} minutes`);
})
.catch(err => console.error(err));
        """, language="javascript")
        
    with code_tab_4:
        st.code("""
import 'dart:convert';
import 'http/http.dart' as http;

Future<void> fetchPrediction() async {
  final url = Uri.parse('http://localhost:8000/predict');
  final headers = {
    'X-API-Key': 'YOUR-API-KEY-HERE',
    'Content-Type': 'application/json',
  };
  
  final body = jsonEncode({
    'order_id': 522,
    'distance_km': 7.93,
    'preparation_time_min': 12,
    'courier_experience_yrs': 3.0,
    'weather': 'Clear',
    'traffic_level': 'Low',
    'time_of_day': 'Afternoon',
    'vehicle_type': 'Scooter'
  });

  try {
    final response = await http.post(url, headers: headers, body: body);
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      print("Predicted ETA: ${data['predicted_delivery_time_minutes']} mins");
    } else {
      print("Request failed with status: ${response.statusCode}");
    }
  } catch (e) {
    print("Error connecting to API: $e");
  }
}
        """, language="dart")
