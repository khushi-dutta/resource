"""
Real-time Disaster Monitoring Dashboard
=====================================

Advanced Streamlit dashboard for real-time disaster monitoring and resource management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
from disaster_prediction_model import DisasterPredictionModel
from resource_planner import DisasterResourcePlanner

# Page configuration
st.set_page_config(
    page_title="Disaster Command Center",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
    }
    .resource-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the disaster prediction model"""
    try:
        data_path = r"c:\Clg projects\SIH\public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv"
        model = DisasterPredictionModel(data_path)
        model.load_model("disaster_prediction_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_threat_level_gauge(severity):
    """Create a threat level gauge"""
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    color = colors[min(int(severity)-1, 4)]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = severity,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Threat Level"},
        delta = {'reference': 3},
        gauge = {
            'axis': {'range': [None, 5]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 1], 'color': "lightgray"},
                {'range': [1, 2], 'color': "gray"},
                {'range': [2, 3], 'color': "yellow"},
                {'range': [3, 4], 'color': "orange"},
                {'range': [4, 5], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 4
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_resource_timeline(timeline_data):
    """Create resource deployment timeline"""
    fig = go.Figure()
    
    phases = [phase['phase'] for phase in timeline_data]
    timeframes = [phase['timeframe'] for phase in timeline_data]
    priorities = [phase['priority'] for phase in timeline_data]
    
    colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'green'}
    
    for i, (phase, timeframe, priority) in enumerate(zip(phases, timeframes, priorities)):
        fig.add_trace(go.Scatter(
            x=[i, i+1],
            y=[1, 1],
            mode='lines+markers',
            name=phase,
            line=dict(color=colors.get(priority, 'blue'), width=8),
            marker=dict(size=15),
            hovertemplate=f"<b>{phase}</b><br>Time: {timeframe}<br>Priority: {priority}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Resource Deployment Timeline",
        xaxis_title="Deployment Phase",
        yaxis=dict(showticklabels=False),
        showlegend=True,
        height=300
    )
    
    return fig

def create_cost_breakdown_chart(costs):
    """Create cost breakdown pie chart"""
    categories = []
    values = []
    
    for category, cost_data in costs.items():
        if category != 'grand_total' and isinstance(cost_data, dict) and 'total' in cost_data:
            categories.append(category.replace('_', ' ').title())
            values.append(cost_data['total'])
    
    fig = px.pie(
        values=values,
        names=categories,
        title="Resource Cost Breakdown",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def simulate_real_time_data():
    """Simulate real-time disaster monitoring data"""
    base_time = datetime.now()
    
    # Simulate multiple active disasters
    active_disasters = [
        {
            'id': 'DIS001',
            'type': 'Flood',
            'location': 'West Bengal',
            'severity': np.random.uniform(3.5, 4.5),
            'area': 1500 + np.random.normal(0, 100),
            'status': 'ACTIVE',
            'time_elapsed': timedelta(hours=6),
            'victims_rescued': np.random.randint(150, 300)
        },
        {
            'id': 'DIS002', 
            'type': 'Earthquake',
            'location': 'Gujarat',
            'severity': np.random.uniform(2.8, 3.8),
            'area': 800 + np.random.normal(0, 50),
            'status': 'MONITORING',
            'time_elapsed': timedelta(hours=2),
            'victims_rescued': np.random.randint(50, 120)
        }
    ]
    
    return active_disasters

def main():
    """Main dashboard application"""
    st.title("🚨 Disaster Command Center")
    st.markdown("*Real-time disaster monitoring and resource management*")
    
    # Load model
    model = load_model()
    if not model:
        st.error("Failed to load prediction model. Please check the setup.")
        return
    
    planner = DisasterResourcePlanner(model)
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Refresh controls
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    if st.sidebar.button("🔄 Manual Refresh"):
        st.rerun()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Live Monitoring", 
        "📊 Predictions", 
        "📦 Resource Planning", 
        "💰 Cost Analysis",
        "📈 Analytics"
    ])
    
    # Tab 1: Live Monitoring
    with tab1:
        st.header("Live Disaster Monitoring")
        
        # Simulate real-time data
        active_disasters = simulate_real_time_data()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Disasters", len(active_disasters), delta=1)
        
        with col2:
            total_victims = sum(d['victims_rescued'] for d in active_disasters)
            st.metric("People Rescued", f"{total_victims:,}", delta=25)
        
        with col3:
            total_area = sum(d['area'] for d in active_disasters)
            st.metric("Affected Area", f"{total_area:,.0f} km²", delta=150)
        
        with col4:
            avg_severity = np.mean([d['severity'] for d in active_disasters])
            st.metric("Avg Severity", f"{avg_severity:.1f}/5", delta=0.2)
        
        # Active disasters table
        st.subheader("Active Disasters")
        
        for disaster in active_disasters:
            with st.expander(f"🚨 {disaster['type']} - {disaster['location']} ({disaster['status']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Severity", f"{disaster['severity']:.1f}/5")
                    st.metric("Area Affected", f"{disaster['area']:.0f} km²")
                    st.metric("Time Elapsed", str(disaster['time_elapsed']))
                
                with col2:
                    # Threat level gauge
                    gauge = create_threat_level_gauge(disaster['severity'])
                    st.plotly_chart(gauge, use_container_width=True)
                
                # Status alerts
                if disaster['severity'] >= 4:
                    st.markdown("""
                    <div class="alert-critical">
                        <strong>CRITICAL ALERT:</strong> High severity disaster requiring immediate maximum response
                    </div>
                    """, unsafe_allow_html=True)
                elif disaster['severity'] >= 3:
                    st.markdown("""
                    <div class="alert-warning">
                        <strong>WARNING:</strong> Moderate to high severity disaster requiring significant resources
                    </div>
                    """, unsafe_allow_html=True)
    
    # Tab 2: Predictions
    with tab2:
        st.header("Disaster Impact Predictions")
        
        # Prediction form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                state = st.selectbox("State", [
                    "Andhra Pradesh", "Gujarat", "West Bengal", "Tamil Nadu", 
                    "Maharashtra", "Karnataka", "Kerala", "Rajasthan"
                ])
                disaster_type = st.selectbox("Disaster Type", [
                    "Flood", "Earthquake", "Tropical cyclone", "Drought",
                    "Heat wave", "Cold wave", "Landslide"
                ])
            
            with col2:
                area = st.number_input("Affected Area (sq km)", min_value=1, max_value=50000, value=1000)
                severity = st.slider("Severity Level", min_value=1, max_value=5, value=3)
            
            submit = st.form_submit_button("🔮 Generate Predictions")
        
        if submit:
            with st.spinner("Generating predictions and resource plans..."):
                # Generate predictions
                predictions = model.predict_victims(state, area, disaster_type, severity)
                
                # Generate resource plan
                resource_plan = planner.generate_resource_plan(
                    state, disaster_type, area, severity, duration_days=7
                )
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                ensemble = predictions['ensemble']
                with col1:
                    st.metric("Deaths", f"{ensemble['Deaths']:,}")
                with col2:
                    st.metric("Injured", f"{ensemble['Injured']:,}")
                with col3:
                    st.metric("Affected", f"{ensemble['Affected']:,}")
                with col4:
                    st.metric("Total Victims", f"{ensemble['Total_Victims']:,}")
                
                # Model comparison chart
                st.subheader("Model Predictions Comparison")
                
                models = ['Random Forest', 'XGBoost', 'Neural Network', 'Ensemble']
                deaths = [
                    predictions['random_forest']['Deaths'],
                    predictions['xgboost']['Deaths'], 
                    predictions['neural_network']['Deaths'],
                    predictions['ensemble']['Deaths']
                ]
                injured = [
                    predictions['random_forest']['Injured'],
                    predictions['xgboost']['Injured'],
                    predictions['neural_network']['Injured'], 
                    predictions['ensemble']['Injured']
                ]
                
                fig = go.Figure(data=[
                    go.Bar(name='Deaths', x=models, y=deaths),
                    go.Bar(name='Injured', x=models, y=injured)
                ])
                
                fig.update_layout(
                    title="Casualty Predictions by Model",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Resource Planning
    with tab3:
        st.header("Resource Planning & Deployment")
        
        if 'resource_plan' in locals():
            # Resource requirements
            st.subheader("Resource Requirements")
            
            resources = resource_plan['resources']
            
            # Create resource cards
            for category, items in resources.items():
                with st.expander(f"📦 {category.replace('_', ' ').title()}"):
                    
                    # Create columns for items
                    cols = st.columns(3)
                    for i, (item, quantity) in enumerate(items.items()):
                        if isinstance(quantity, (int, float)):
                            with cols[i % 3]:
                                st.metric(
                                    item.replace('_', ' ').title(),
                                    f"{quantity:,}" if quantity >= 1 else f"{quantity:.1f}"
                                )
            
            # Deployment timeline
            st.subheader("Deployment Timeline")
            timeline_chart = create_resource_timeline(resource_plan['timeline'])
            st.plotly_chart(timeline_chart, use_container_width=True)
            
            # Timeline details
            for phase in resource_plan['timeline']:
                priority_class = f"alert-{'critical' if phase['priority'] == 'CRITICAL' else 'warning' if phase['priority'] == 'HIGH' else 'info'}"
                st.markdown(f"""
                <div class="{priority_class}">
                    <strong>{phase['phase']} ({phase['timeframe']})</strong><br>
                    Priority: {phase['priority']}<br>
                    {phase['description']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Generate predictions first to see resource planning details.")
    
    # Tab 4: Cost Analysis
    with tab4:
        st.header("Cost Analysis & Budget Planning")
        
        if 'resource_plan' in locals():
            costs = resource_plan['costs']
            
            # Total cost overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Cost", 
                    f"₹{costs['grand_total']:,.2f}",
                    help="Total estimated cost for 7-day operation"
                )
            
            with col2:
                daily_cost = costs['grand_total'] / 7
                st.metric("Daily Cost", f"₹{daily_cost:,.2f}")
            
            with col3:
                cost_per_victim = costs['grand_total'] / ensemble['Total_Victims'] if ensemble['Total_Victims'] > 0 else 0
                st.metric("Cost per Victim", f"₹{cost_per_victim:,.2f}")
            
            # Cost breakdown chart
            st.subheader("Cost Breakdown by Category")
            cost_chart = create_cost_breakdown_chart(costs)
            st.plotly_chart(cost_chart, use_container_width=True)
            
            # Detailed cost table
            st.subheader("Detailed Cost Analysis")
            
            cost_data = []
            for category, category_costs in costs.items():
                if category != 'grand_total' and isinstance(category_costs, dict):
                    for item, item_data in category_costs.items():
                        if item != 'total' and isinstance(item_data, dict):
                            cost_data.append({
                                'Category': category.replace('_', ' ').title(),
                                'Item': item.replace('_', ' ').title(),
                                'Quantity': item_data['quantity'],
                                'Unit Cost': f"₹{item_data['unit_cost']:,.2f}",
                                'Total Cost': f"₹{item_data['total_cost']:,.2f}"
                            })
            
            if cost_data:
                cost_df = pd.DataFrame(cost_data)
                st.dataframe(cost_df, use_container_width=True)
        else:
            st.info("Generate predictions first to see cost analysis.")
    
    # Tab 5: Analytics
    with tab5:
        st.header("Historical Analytics & Trends")
        
        # Simulate historical data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        disaster_counts = np.random.poisson(3, len(dates))
        severity_avg = np.random.uniform(2.5, 4.0, len(dates))
        
        # Historical trends
        col1, col2 = st.columns(2)
        
        with col1:
            # Disaster frequency chart
            fig1 = px.line(
                x=dates, 
                y=disaster_counts,
                title="Monthly Disaster Frequency",
                labels={'x': 'Month', 'y': 'Number of Disasters'}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Severity trends
            fig2 = px.line(
                x=dates,
                y=severity_avg, 
                title="Average Disaster Severity",
                labels={'x': 'Month', 'y': 'Average Severity'}
            )
            fig2.add_hline(y=3.0, line_dash="dash", line_color="red", 
                          annotation_text="High Severity Threshold")
            st.plotly_chart(fig2, use_container_width=True)
        
        # State-wise statistics
        st.subheader("State-wise Disaster Statistics")
        
        states = ["West Bengal", "Gujarat", "Tamil Nadu", "Andhra Pradesh", "Maharashtra"]
        disaster_types = ["Flood", "Earthquake", "Cyclone", "Drought", "Heat Wave"]
        
        # Create sample data
        state_data = []
        for state in states:
            for disaster in disaster_types:
                count = np.random.poisson(2)
                if count > 0:
                    state_data.append({
                        'State': state,
                        'Disaster Type': disaster,
                        'Occurrences': count,
                        'Avg Severity': np.random.uniform(2.0, 4.5)
                    })
        
        state_df = pd.DataFrame(state_data)
        
        # Heatmap of disaster occurrences
        pivot_data = state_df.pivot_table(
            index='State', 
            columns='Disaster Type', 
            values='Occurrences', 
            fill_value=0
        )
        
        fig3 = px.imshow(
            pivot_data,
            title="Disaster Occurrence Heatmap by State and Type",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🚨 Disaster Command Center | Real-time Monitoring & Resource Management<br>
        Last Updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()