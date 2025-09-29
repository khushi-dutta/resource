"""
Interactive Web Interface for Disaster Prediction Model
=======================================================

A Streamlit web application for predicting disaster victims in India
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from disaster_prediction_model import DisasterPredictionModel

# Page configuration
st.set_page_config(
    page_title="India Disaster Prediction System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_disaster_model():
    """Load the trained disaster prediction model"""
    try:
        model = DisasterPredictionModel(r"c:\Clg projects\SIH\public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv")
        
        # Check if saved model exists
        if os.path.exists("disaster_prediction_model.pkl"):
            model.load_model("disaster_prediction_model.pkl")
            return model, True
        else:
            # Train model if not saved
            model.load_and_preprocess_data()
            model.train_models()
            model.save_model("disaster_prediction_model.pkl")
            return model, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

def main():
    # Header
    st.markdown('<h1 class="main-header">🚨 India Disaster Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Predict disaster casualties based on location, area, disaster type, and severity
    This system uses machine learning to predict the number of deaths, injuries, and affected people 
    for natural disasters in India based on historical data from EM-DAT database.
    """)
    
    # Load model
    with st.spinner("Loading prediction model..."):
        model, model_loaded = load_disaster_model()
    
    if model is None:
        st.error("Failed to load the prediction model. Please check the data file.")
        return
    
    if not model_loaded:
        st.success("Model trained successfully!")
    else:
        st.success("Model loaded successfully!")
    
    # Sidebar for inputs
    st.sidebar.header("🎯 Prediction Parameters")
    
    # Indian states
    states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Delhi', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
        'Meghalaya', 'Mizoram', 'Nagaland', 'Orissa', 'Punjab', 'Rajasthan',
        'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
        'Uttarakhand', 'West Bengal'
    ]
    
    # Disaster types
    disaster_types = [
        'Flood', 'Earthquake', 'Tropical cyclone', 'Drought', 'Landslide',
        'Heat wave', 'Cold wave', 'Storm', 'Epidemic', 'Wildfire',
        'Hail', 'Lightning', 'Avalanche'
    ]
    
    # Input parameters
    selected_state = st.sidebar.selectbox("🏛️ Select State", states, index=states.index('West Bengal'))
    
    disaster_type = st.sidebar.selectbox("💥 Disaster Type", disaster_types, index=0)
    
    area_input = st.sidebar.number_input(
        "📏 Affected Area (sq km)", 
        min_value=1, 
        max_value=100000, 
        value=1000,
        help="Enter the estimated area that will be affected by the disaster"
    )
    
    severity_level = st.sidebar.slider(
        "⚠️ Severity Level", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="1 = Very Low, 2 = Low, 3 = Moderate, 4 = High, 5 = Very High"
    )
    
    # Severity level description
    severity_descriptions = {
        1: "Very Low - Minor impact expected",
        2: "Low - Limited damage and casualties",
        3: "Moderate - Significant but manageable impact",
        4: "High - Serious damage and many casualties",
        5: "Very High - Catastrophic damage and casualties"
    }
    
    st.sidebar.info(f"**Severity Level {severity_level}:** {severity_descriptions[severity_level]}")
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Victims", type="primary"):
        with st.spinner("Generating predictions..."):
            try:
                # Make prediction
                predictions = model.predict_victims(selected_state, area_input, disaster_type, severity_level)
                
                # Display results
                st.header("📊 Prediction Results")
                
                # Main prediction (ensemble)
                ensemble_pred = predictions['ensemble']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "💀 Deaths",
                        f"{ensemble_pred['Deaths']*5:,}",
                        help="Predicted number of fatalities"
                    )
                
                with col2:
                    st.metric(
                        "🩹 Injured",
                        f"{ensemble_pred['Injured']*27:,}",
                        help="Predicted number of injured people"
                    )
                
                with col3:
                    st.metric(
                        "👥 Affected",
                        f"{ensemble_pred['Affected']:,}",
                        help="Predicted number of affected people"
                    )
                
                with col4:
                    st.metric(
                        "📈 Total Victims",
                        f"{ensemble_pred['Total_Victims']:,}",
                        help="Total predicted victims (deaths + injured + affected)"
                    )
                
                # Resource Summary Metrics
                if 'resources' in predictions:
                    st.subheader("📊 Key Resource Requirements")
                    resources = predictions['resources']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # with col1:
                    #     st.metric(
                    #         "🏥 Medical Staff",
                    #         f"{resources['medical']['doctors'] + resources['medical']['nurses']}",
                    #         help="Total doctors and nurses needed"
                    #     )
                    
                    with col1:
                        st.metric(
                            "🏠 Tents",
                            f"{resources['shelter']['tents']:,}",
                            help="Emergency shelter tents required"
                        )
                    
                    with col2:
                        st.metric(
                            "🍽️ Food (tons)",
                            f"{resources['food_water']['food_kg'] / 1000:.1f}",
                            help="Food required in metric tons"
                        )
                    
                    with col3:
                        st.metric(
                            "👮 Personnel",
                            f"{resources['personnel']['search_rescue_teams'] + resources['personnel']['police_officers']}",
                            help="Emergency personnel required"
                        )
                    
                    with col4:
                        st.metric(
                            "🚛 Vehicles",
                            f"{resources['logistics']['buses'] + resources['logistics']['trucks']}",
                            help="Transportation vehicles needed"
                        )
                
                
                # Risk assessment
                st.subheader("🎯 Risk Assessment")
                
                total_victims = ensemble_pred['Total_Victims']
                
                if total_victims < 100:
                    risk_level = "Low Risk"
                    risk_color = "green"
                elif total_victims < 1000:
                    risk_level = "Moderate Risk"
                    risk_color = "orange"
                elif total_victims < 10000:
                    risk_level = "High Risk"
                    risk_color = "red"
                else:
                    risk_level = "Critical Risk"
                    risk_color = "darkred"
                
                st.markdown(f"""
                <div style="background-color: {risk_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h3>Risk Level: {risk_level}</h3>
                    <p>Based on predicted total victims: {total_victims:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Resource Requirements
                if 'resources' in predictions:
                    st.subheader("📦 Required Resources")
                    
                    resources = predictions['resources']
                    
                    # Create tabs for different resource categories
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([ "🏠 Shelter", "🍽️ Food & Water", "👮 Personnel", "🚛 Logistics", "📡 Equipment"])
                    
                    # with tab1:
                    #     medical = resources['medical']
                    #     col1, col2 = st.columns(2)
                    #     with col1:
                    #         st.metric("Doctors", medical['doctors'])
                    #         st.metric("Nurses", medical['nurses'])
                    #         st.metric("Paramedics", medical['paramedics'])
                    #     with col2:
                    #         st.metric("Ambulances", medical['ambulances'])
                    #         st.metric("Stretchers", medical['stretchers'])
                    #         st.metric("Oxygen Cylinders", medical['oxygen_cylinders'])
                        
                    #     if 'blood_units' in medical:
                    #         st.metric("Blood Units Required", medical['blood_units'])
                    
                    with tab1:
                        shelter = resources['shelter']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Tents", shelter['tents'])
                            st.metric("Blankets", f"{shelter['blankets']:,}")
                            st.metric("Mattresses", f"{shelter['mattresses']:,}")
                        with col2:
                            st.metric("Portable Toilets", shelter['portable_toilets'])
                            st.metric("Cooking Stoves", shelter['cooking_stoves'])
                            st.metric("Clothing Sets", f"{shelter['clothing_sets']:,}")
                    
                    with tab2:
                        food_water = resources['food_water']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Food (kg)", f"{food_water['food_kg']:,}")
                            st.metric("Rice (kg)", f"{food_water['rice_kg']:,}")
                            st.metric("Dal (kg)", f"{food_water['dal_kg']:,}")
                        with col2:
                            st.metric("Water (Liters)", f"{food_water['water_liters']:,}")
                            st.metric("Cooking Oil (L)", f"{food_water['oil_liters']:,}")
                            st.metric("Baby Food (kg)", f"{food_water['baby_food_kg']:,}")
                    
                    with tab3:
                        personnel = resources['personnel']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Search & Rescue Teams", personnel['search_rescue_teams'])
                            st.metric("Police Officers", personnel['police_officers'])
                            st.metric("Coordinators", personnel['coordinators'])
                        with col2:
                            st.metric("Trained Volunteers", personnel['trained_volunteers'])
                            if 'sniffer_dogs' in personnel and personnel['sniffer_dogs'] > 0:
                                st.metric("Sniffer Dogs", personnel['sniffer_dogs'])
                            if 'water_rescue_specialists' in personnel:
                                st.metric("Water Rescue Specialists", personnel['water_rescue_specialists'])
                    
                    with tab4:
                        logistics = resources['logistics']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Buses", logistics['buses'])
                            st.metric("Trucks", logistics['trucks'])
                            st.metric("Fuel (Liters)", f"{logistics['fuel_liters']:,}")
                        with col2:
                            if 'excavators' in logistics:
                                st.metric("Excavators", logistics['excavators'])
                            if 'boats' in logistics:
                                st.metric("Boats", logistics['boats'])
                            if 'water_pumps' in logistics:
                                st.metric("Water Pumps", logistics['water_pumps'])
                    
                    with tab5:
                        equipment = resources['equipment']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Satellite Phones", equipment['satellite_phones'])
                            st.metric("Walkie Talkies", equipment['walkie_talkies'])
                            st.metric("Generators", equipment['generators'])
                        with col2:
                            st.metric("LED Floodlights", equipment['led_floodlights'])
                            st.metric("First Aid Kits", equipment['first_aid_kits'])
                            if 'life_jackets' in equipment:
                                st.metric("Life Jackets", equipment['life_jackets'])
                
                # Recommendations
                st.subheader("💡 Emergency Response Recommendations")
                
                recommendations = generate_recommendations(disaster_type, severity_level, total_victims)
                
                for i, rec in enumerate(recommendations, 1):
                    st.info(f"**{i}.** {rec}")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # Historical data analysis
    st.header("📈 Historical Disaster Analysis")
    
    if st.button("Show Historical Analysis"):
        with st.spinner("Analyzing historical data..."):
            try:
                # Load historical data
                df = model.df
                
                # State-wise analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Most Affected States")
                    state_victims = df.groupby('State')['Total Victims'].sum().sort_values(ascending=False).head(10)
                    
                    fig = px.bar(
                        x=state_victims.values,
                        y=state_victims.index,
                        orientation='h',
                        title="Total Victims by State (Historical)",
                        labels={'x': 'Total Victims', 'y': 'State'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Disaster Types Distribution")
                    disaster_counts = df['Disaster Type'].value_counts().head(10)
                    
                    fig = px.pie(
                        values=disaster_counts.values,
                        names=disaster_counts.index,
                        title="Most Common Disasters"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Time series analysis
                st.subheader("Disaster Trends Over Time")
                yearly_stats = df.groupby('Start Year').agg({
                    'Total Deaths': 'sum',
                    'No. Injured': 'sum',
                    'No. Affected': 'sum'
                }).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=yearly_stats['Start Year'], y=yearly_stats['Total Deaths'], 
                                       mode='lines+markers', name='Deaths'))
                fig.add_trace(go.Scatter(x=yearly_stats['Start Year'], y=yearly_stats['No. Injured'], 
                                       mode='lines+markers', name='Injured'))
                fig.add_trace(go.Scatter(x=yearly_stats['Start Year'], y=yearly_stats['No. Affected'], 
                                       mode='lines+markers', name='Affected'))
                
                fig.update_layout(
                    title="Disaster Impact Trends (1990-2024)",
                    xaxis_title="Year",
                    yaxis_title="Number of People",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Dataset Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Disasters", f"{len(df):,}")
                
                with col2:
                    st.metric("Total Deaths", f"{df['Total Deaths'].sum():,}")
                
                with col3:
                    st.metric("Total Injured", f"{df['No. Injured'].sum():,}")
                
                with col4:
                    st.metric("Total Affected", f"{df['No. Affected'].sum():,}")
                
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")

def generate_recommendations(disaster_type, severity_level, total_victims):
    """Generate emergency response recommendations"""
    recommendations = []
    
    # General recommendations based on disaster type
    if 'flood' in disaster_type.lower():
        recommendations.extend([
            "Activate flood warning systems and evacuation procedures",
            "Prepare boats and amphibious vehicles for rescue operations",
            "Set up temporary shelters on higher ground",
            "Coordinate with water and sanitation teams"
        ])
    elif 'earthquake' in disaster_type.lower():
        recommendations.extend([
            "Deploy search and rescue teams immediately",
            "Set up medical triage centers near affected areas",
            "Inspect critical infrastructure for damage",
            "Prepare for potential aftershocks"
        ])
    elif 'cyclone' in disaster_type.lower() or 'storm' in disaster_type.lower():
        recommendations.extend([
            "Issue early warning alerts to coastal populations",
            "Evacuate vulnerable coastal areas",
            "Prepare emergency shelters inland",
            "Stock emergency supplies for extended isolation"
        ])
    
    # Severity-based recommendations
    if severity_level >= 4:
        recommendations.extend([
            "Declare state of emergency",
            "Request national/international assistance",
            "Deploy military for large-scale operations"
        ])
    
    # Victim count-based recommendations
    if total_victims >= 10000:
        recommendations.extend([
            "Establish multiple emergency operation centers",
            "Coordinate with international relief organizations",
            "Set up mass casualty treatment facilities"
        ])
    elif total_victims >= 1000:
        recommendations.extend([
            "Deploy additional medical teams",
            "Set up temporary hospitals",
            "Coordinate with neighboring states for resources"
        ])
    
    # Add general recommendations
    recommendations.extend([
        "Maintain communication with all emergency services",
        "Monitor weather conditions continuously",
        "Prepare public information campaigns",
        "Coordinate with local authorities and volunteers"
    ])
    
    return recommendations[:8]  # Return top 8 recommendations

if __name__ == "__main__":
    main()