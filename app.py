import streamlit as st
import pickle
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# EPIALERT STREAMLIT APP - Real-time Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EpiAlert - Disease Outbreak Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ EpiAlert - Disease Outbreak Detection System")
st.markdown("**Real-time anomaly detection for epidemic surveillance in India**")

# ═══════════════════════════════════════════════════════════════════════════
# PINCODE COORDINATES DATABASE
# ═══════════════════════════════════════════════════════════════════════════

PINCODE_COORDINATES = {
    110001: {'city': 'Delhi', 'state': 'Delhi', 'latitude': 28.6139, 'longitude': 77.2090},
    400001: {'city': 'Mumbai', 'state': 'Maharashtra', 'latitude': 18.9388, 'longitude': 72.8354},
    560001: {'city': 'Bangalore', 'state': 'Karnataka', 'latitude': 12.9716, 'longitude': 77.5946},
    600001: {'city': 'Chennai', 'state': 'Tamil Nadu', 'latitude': 13.0827, 'longitude': 80.2707},
    700001: {'city': 'Kolkata', 'state': 'West Bengal', 'latitude': 22.5726, 'longitude': 88.3639},
    500001: {'city': 'Hyderabad', 'state': 'Telangana', 'latitude': 17.3850, 'longitude': 78.4867},
    380001: {'city': 'Ahmedabad', 'state': 'Gujarat', 'latitude': 23.0225, 'longitude': 72.5714},
    411001: {'city': 'Pune', 'state': 'Maharashtra', 'latitude': 18.5204, 'longitude': 73.8567},
    302001: {'city': 'Jaipur', 'state': 'Rajasthan', 'latitude': 26.9124, 'longitude': 75.7873},
    226001: {'city': 'Lucknow', 'state': 'Uttar Pradesh', 'latitude': 26.8467, 'longitude': 80.9462},
    160001: {'city': 'Chandigarh', 'state': 'Chandigarh', 'latitude': 30.7333, 'longitude': 76.7794},
    800001: {'city': 'Patna', 'state': 'Bihar', 'latitude': 25.5941, 'longitude': 85.1376},
    751001: {'city': 'Bhubaneswar', 'state': 'Odisha', 'latitude': 20.2961, 'longitude': 85.8245},
    682001: {'city': 'Kochi', 'state': 'Kerala', 'latitude': 9.9312, 'longitude': 76.2673},
    695001: {'city': 'Thiruvananthapuram', 'state': 'Kerala', 'latitude': 8.5241, 'longitude': 76.9366},
    781001: {'city': 'Guwahati', 'state': 'Assam', 'latitude': 26.1445, 'longitude': 91.7362},
    641001: {'city': 'Coimbatore', 'state': 'Tamil Nadu', 'latitude': 11.0168, 'longitude': 76.9558},
    626001: {'city': 'Virudhunagar', 'state': 'Tamil Nadu', 'latitude': 9.5810, 'longitude': 77.9624},
    462001: {'city': 'Bhopal', 'state': 'Madhya Pradesh', 'latitude': 23.2599, 'longitude': 77.4126},
    492001: {'city': 'Raipur', 'state': 'Chhattisgarh', 'latitude': 21.2514, 'longitude': 81.6296},
    834001: {'city': 'Ranchi', 'state': 'Jharkhand', 'latitude': 23.3441, 'longitude': 85.3096},
    244001: {'city': 'Moradabad', 'state': 'Uttar Pradesh', 'latitude': 28.8389, 'longitude': 78.7378},
    134001: {'city': 'Ambala', 'state': 'Haryana', 'latitude': 30.3752, 'longitude': 76.7821},
    141001: {'city': 'Ludhiana', 'state': 'Punjab', 'latitude': 30.9010, 'longitude': 75.8573},
    144001: {'city': 'Jalandhar', 'state': 'Punjab', 'latitude': 31.3260, 'longitude': 75.5762},
    301001: {'city': 'Alwar', 'state': 'Rajasthan', 'latitude': 27.5530, 'longitude': 76.6346},
    324001: {'city': 'Kota', 'state': 'Rajasthan', 'latitude': 25.2138, 'longitude': 75.8648},
    342001: {'city': 'Jodhpur', 'state': 'Rajasthan', 'latitude': 26.2389, 'longitude': 73.0243},
    390001: {'city': 'Vadodara', 'state': 'Gujarat', 'latitude': 22.3072, 'longitude': 73.1812},
    360001: {'city': 'Rajkot', 'state': 'Gujarat', 'latitude': 22.3039, 'longitude': 70.8022},
    421001: {'city': 'Ulhasnagar', 'state': 'Maharashtra', 'latitude': 19.2183, 'longitude': 73.1382},
    440001: {'city': 'Nagpur', 'state': 'Maharashtra', 'latitude': 21.1458, 'longitude': 79.0882},
    416001: {'city': 'Kolhapur', 'state': 'Maharashtra', 'latitude': 16.7050, 'longitude': 74.2433},
    831001: {'city': 'Jamshedpur', 'state': 'Jharkhand', 'latitude': 22.8046, 'longitude': 86.2029},
    827001: {'city': 'Bokaro', 'state': 'Jharkhand', 'latitude': 23.6693, 'longitude': 86.1511},
    533001: {'city': 'Rajahmundry', 'state': 'Andhra Pradesh', 'latitude': 17.0005, 'longitude': 81.8040},
    522503: {'city': 'Amaravati', 'state': 'Andhra Pradesh', 'latitude': 16.5062, 'longitude': 80.5120},
    530001: {'city': 'Visakhapatnam', 'state': 'Andhra Pradesh', 'latitude': 17.6868, 'longitude': 83.2185},
    585101: {'city': 'Gulbarga', 'state': 'Karnataka', 'latitude': 17.3297, 'longitude': 76.8343},
    575001: {'city': 'Mangalore', 'state': 'Karnataka', 'latitude': 12.9141, 'longitude': 74.8560},
    673001: {'city': 'Calicut', 'state': 'Kerala', 'latitude': 11.2588, 'longitude': 75.7804},
    686001: {'city': 'Kottayam', 'state': 'Kerala', 'latitude': 9.5916, 'longitude': 76.5222},
    688001: {'city': 'Alappuzha', 'state': 'Kerala', 'latitude': 9.4981, 'longitude': 76.3388},
    735101: {'city': 'Jalpaiguri', 'state': 'West Bengal', 'latitude': 26.5163, 'longitude': 88.7279},
    713101: {'city': 'Durgapur', 'state': 'West Bengal', 'latitude': 23.5204, 'longitude': 87.3119},
    737101: {'city': 'Gangtok', 'state': 'Sikkim', 'latitude': 27.3389, 'longitude': 88.6065},
    742101: {'city': 'Berhampore', 'state': 'West Bengal', 'latitude': 24.0958, 'longitude': 88.2636},
    788001: {'city': 'Silchar', 'state': 'Assam', 'latitude': 24.8333, 'longitude': 92.7789},
    791111: {'city': 'Itanagar', 'state': 'Arunachal Pradesh', 'latitude': 27.0844, 'longitude': 93.6053},
    793001: {'city': 'Shillong', 'state': 'Meghalaya', 'latitude': 25.5788, 'longitude': 91.8933},
}

def get_pincode_coordinates(pincode):
    """Get coordinates for a given pincode."""
    return PINCODE_COORDINATES.get(pincode, None)

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_season(month_num):
    """Map month to season (India)."""
    if month_num in [6, 7, 8, 9]:  # June-Sep
        return 'monsoon'
    elif month_num in [10, 11]:     # Oct-Nov
        return 'post_monsoon'
    elif month_num in [12, 1, 2]:   # Dec-Feb
        return 'winter'
    else:                            # Mar-May
        return 'summer'

def engineer_features(df):
    """Apply feature engineering to input dataframe."""
    # Convert date and add temporal features
    df['report_date'] = pd.to_datetime(df['report_date'])
    df['month'] = df['report_date'].dt.month
    df['day_of_week'] = df['report_date'].dt.dayofweek
    df['week_of_year'] = df['report_date'].dt.isocalendar().week
    df['is_weekend'] = df['report_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Daily aggregation
    daily_agg = (
        df.groupby(['report_date', 'pincode', 'symptom_cluster', 'month', 'day_of_week', 'week_of_year', 'is_weekend'])
        .size()
        .reset_index(name='daily_count')
        .sort_values(['pincode', 'symptom_cluster', 'report_date'])
    )
    
    # Baseline estimates (typical values for each symptom cluster)
    baseline_estimates = {
        'fever': 5.0, 'fever_cough': 4.0, 'fever_rash': 2.0, 'fever_jointpain': 2.0,
        'cough': 6.0, 'cough_cold': 7.0, 'cough_fever': 4.0,
        'diarrhea': 3.0, 'cholera_like': 1.5, 'other': 4.0
    }
    
    def get_baseline(symptom):
        return baseline_estimates.get(symptom, 3.0)
    
    # Add estimated features
    daily_agg['30d_avg'] = daily_agg['symptom_cluster'].apply(get_baseline)
    daily_agg['7d_avg'] = daily_agg['30d_avg'] * 1.1  # Slightly higher for 7-day
    daily_agg['count_vs_30d_avg'] = daily_agg['daily_count'] / (daily_agg['30d_avg'] + 0.001)
    daily_agg['prev_day_count'] = daily_agg['30d_avg'] * 0.9
    daily_agg['delta_7d'] = daily_agg['daily_count'] - daily_agg['7d_avg']
    daily_agg['rate_of_change'] = daily_agg['daily_count'] / (daily_agg['30d_avg'] * 0.9 + 0.001)
    
    # Add season
    daily_agg['season'] = daily_agg['month'].apply(get_season)
    
    # Encode categorical variables
    daily_agg['season'] = daily_agg['season'].fillna('unknown').astype(str)
    daily_agg['symptom_cluster'] = daily_agg['symptom_cluster'].fillna('other').astype(str)
    
    le_season = LabelEncoder()
    le_cluster = LabelEncoder()
    
    daily_agg['season_enc'] = le_season.fit_transform(daily_agg['season'])
    daily_agg['cluster_enc'] = le_cluster.fit_transform(daily_agg['symptom_cluster'])
    
    return daily_agg

# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

# Sidebar for model upload
st.sidebar.header("⚙️ Configuration")
model_file = st.sidebar.file_uploader(
    "Upload Trained Model (model.pkl)",
    type=['pkl'],
    help="Upload the IsolationForest model file"
)

if model_file is not None:
    try:
        if_model = pickle.load(model_file)
        st.sidebar.success("✓ Model loaded successfully!")
        st.sidebar.info(f"Model expects {if_model.n_features_in_} features")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        if_model = None
else:
    st.sidebar.warning("⚠ Please upload a trained model file")
    if_model = None

# Main content area
st.header("📊 Input Data")

# Option to upload CSV or manual entry
input_method = st.radio(
    "Choose input method:",
    ["Upload CSV", "Manual Entry", "Use Example Data"],
    horizontal=True
)

input_data = None

if input_method == "Upload CSV":
    st.markdown("**CSV Format:** `report_date, pincode, symptom_cluster`")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(input_data)} records")
        st.dataframe(input_data.head())

elif input_method == "Manual Entry":
    col1, col2, col3 = st.columns(3)
    with col1:
        date_input = st.date_input("Report Date", datetime.now())
    with col2:
        pincode_input = st.selectbox("Pincode", list(PINCODE_COORDINATES.keys()))
    with col3:
        symptom_input = st.selectbox(
            "Symptom Cluster",
            ['fever', 'fever_cough', 'fever_rash', 'fever_jointpain', 
             'cough', 'cough_cold', 'cough_fever', 'diarrhea', 'cholera_like', 'other']
        )
    
    num_records = st.slider("Number of cases to report", 1, 20, 5)
    
    if st.button("Add Data"):
        input_data = pd.DataFrame({
            'report_date': [date_input.strftime('%Y-%m-%d')] * num_records,
            'pincode': [pincode_input] * num_records,
            'symptom_cluster': [symptom_input] * num_records
        })
        st.success(f"✓ Added {num_records} records")
        st.dataframe(input_data)

else:  # Use Example Data
    st.info("Using example data with potential anomalies...")
    input_data = pd.DataFrame([
        ('2024-04-15', 522503, 'fever_rash'),
        ('2024-04-15', 522503, 'fever_rash'),
        ('2024-04-15', 522503, 'fever_rash'),
        ('2024-04-15', 522503, 'fever_rash'),
        ('2024-04-15', 791111, 'fever_jointpain'),
        ('2024-04-15', 791111, 'fever_jointpain'),
        ('2024-04-15', 791111, 'fever_jointpain'),
        ('2024-04-15', 600001, 'fever'),
        ('2024-04-15', 600001, 'fever'),
        ('2024-04-15', 600001, 'fever'),
        ('2024-04-15', 600001, 'fever'),
        ('2024-04-15', 110001, 'cough'),
        ('2024-04-15', 110001, 'cough'),
        ('2024-04-15', 400001, 'fever_cough'),
        ('2024-04-15', 400001, 'fever_cough'),
    ], columns=['report_date', 'pincode', 'symptom_cluster'])
    st.dataframe(input_data)

# Run analysis button
if st.button("🔍 Run Anomaly Detection", type="primary", disabled=(input_data is None or if_model is None)):
    if input_data is not None and if_model is not None:
        with st.spinner("Processing data and detecting anomalies..."):
            # Feature engineering
            features_pd = engineer_features(input_data)
            
            # Prepare features for model
            FEATURE_COLS = [
                'daily_count', '7d_avg', '30d_avg', 'count_vs_30d_avg', 'delta_7d', 'rate_of_change',
                'season_enc', 'month', 'week_of_year', 'is_weekend', 'day_of_week'
            ]
            
            # Fill any missing values
            for col_name in FEATURE_COLS:
                features_pd[col_name] = pd.to_numeric(features_pd[col_name], errors='coerce').fillna(0)
            
            # Extract feature matrix
            X = features_pd[FEATURE_COLS].values
            
            # Run anomaly detection
            predictions = if_model.predict(X)
            anomaly_scores = if_model.score_samples(X)
            
            # Add predictions
            features_pd['is_anomaly'] = (predictions == -1)
            features_pd['anomaly_score'] = anomaly_scores
            
            # Filter anomalies
            anomalies = features_pd[features_pd['is_anomaly']].copy()
            anomalies = anomalies.sort_values('anomaly_score')
            
        # Display results
        st.header("📈 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(features_pd))
        with col2:
            st.metric("Anomalies Detected", len(anomalies), delta=f"{len(anomalies)/len(features_pd)*100:.1f}%")
        with col3:
            st.metric("Normal Cases", len(features_pd) - len(anomalies))
        
        # Anomaly details
        if len(anomalies) > 0:
            st.subheader("🚨 Detected Anomalies")
            
            for idx, row in anomalies.iterrows():
                coords = get_pincode_coordinates(row['pincode'])
                city = coords['city'] if coords else 'Unknown'
                state = coords['state'] if coords else 'Unknown'
                
                with st.expander(f"⚠️ {city}, {state} - {row['symptom_cluster']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Date:** {row['report_date']}")
                        st.write(f"**Pincode:** {row['pincode']}")
                        st.write(f"**Daily Count:** {int(row['daily_count'])} cases")
                    with col2:
                        st.write(f"**Baseline:** {row['30d_avg']:.1f} cases")
                        st.write(f"**Spike Ratio:** {row['count_vs_30d_avg']:.2f}x")
                        st.write(f"**Anomaly Score:** {row['anomaly_score']:.3f}")
        else:
            st.info("✓ No anomalies detected in the input data")
        
        # Map visualization
        st.header("🗺️ Geographic Visualization")
        
        if len(features_pd) > 0:
            # Calculate map center
            valid_coords = [get_pincode_coordinates(p) for p in features_pd['pincode'].unique()]
            valid_coords = [c for c in valid_coords if c is not None]
            
            if len(valid_coords) == 0:
                center_lat, center_lon = 23.0, 79.0
            else:
                center_lat = np.mean([c['latitude'] for c in valid_coords])
                center_lon = np.mean([c['longitude'] for c in valid_coords])
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=5,
                tiles='CartoDB positron'
            )
            
            # Add markers
            for idx, row in features_pd.iterrows():
                coords = get_pincode_coordinates(row['pincode'])
                if coords is None:
                    continue
                
                lat = coords['latitude']
                lon = coords['longitude']
                city = coords['city']
                state = coords['state']
                is_anomaly = row['is_anomaly']
                
                if is_anomaly:
                    color = '#DC143C'
                    fill_color = '#FF0000'
                    marker_size = 16
                    label = '🚨 ANOMALY'
                else:
                    color = '#4169E1'
                    fill_color = '#1E90FF'
                    marker_size = 10
                    label = '✓ Normal'
                
                popup_html = f"""
                <div style="font-family: Arial; width: 250px;">
                    <h4 style="color: {color}; margin: 0;">{label}</h4>
                    <hr style="margin: 5px 0;">
                    <b>Location:</b> {city}, {state}<br>
                    <b>Pincode:</b> {row['pincode']}<br>
                    <b>Date:</b> {row['report_date']}<br>
                    <b>Symptom:</b> {row['symptom_cluster']}<br>
                    <b>Cases:</b> {int(row['daily_count'])}<br>
                    <b>Baseline:</b> {row['30d_avg']:.1f}<br>
                    <b>Spike Ratio:</b> {row['count_vs_30d_avg']:.2f}x<br>
                    <b>Score:</b> {row['anomaly_score']:.3f}
                </div>
                """
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=marker_size,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{city} - {label}",
                    color=color,
                    fill=True,
                    fillColor=fill_color,
                    fillOpacity=0.8 if is_anomaly else 0.6,
                    weight=3 if is_anomaly else 2
                ).add_to(m)
            
            # Display map
            folium_static(m, width=1200, height=600)
            
            # Download results
            st.subheader("💾 Export Results")
            if len(anomalies) > 0:
                csv = anomalies[['report_date', 'pincode', 'symptom_cluster', 'daily_count', '30d_avg', 'count_vs_30d_avg', 'anomaly_score']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Anomaly Report (CSV)",
                    data=csv,
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown(
    """<div style='text-align: center; color: #666;'>
    <p>EpiAlert v1.0 | Disease Outbreak Detection System</p>
    <p>🔴 RED markers indicate anomalies | 🔵 BLUE markers indicate normal cases</p>
    </div>""",
    unsafe_allow_html=True
)
