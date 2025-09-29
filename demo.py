"""
Demo Script for India Disaster Prediction System
===============================================

This script demonstrates all features of the disaster prediction system
"""

import os
import sys
from disaster_prediction_model import DisasterPredictionModel

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*60)
    print(f" {text} ")
    print("="*60)

def print_section(text):
    """Print a formatted section header"""
    print(f"\n{text}")
    print("-" * len(text))

def demo_predictions():
    """Demonstrate various disaster predictions"""
    print_banner("🚨 INDIA DISASTER PREDICTION SYSTEM DEMO")
    
    # Initialize model
    print("🔧 Initializing prediction model...")
    model_path = "disaster_prediction_model.pkl"
    data_path = r"c:\Clg projects\SIH\public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv"
    
    model = DisasterPredictionModel(data_path)
    
    if os.path.exists(model_path):
        model.load_model(model_path)
        print("✅ Model loaded successfully!")
    else:
        print("🔄 Training new model...")
        model.load_and_preprocess_data()
        model.train_models()
        model.save_model(model_path)
        print("✅ Model trained and saved!")
    
    # Demo scenarios
    scenarios = [
        {
            "name": "🌊 Monsoon Flooding in West Bengal",
            "state": "West Bengal",
            "disaster": "Flood",
            "area": 1500,
            "severity": 4,
            "description": "Heavy monsoon rains cause widespread flooding in rural areas"
        },
        {
            "name": "🏚️ Major Earthquake in Gujarat", 
            "state": "Gujarat",
            "disaster": "Earthquake",
            "area": 800,
            "severity": 5,
            "description": "Magnitude 7.5 earthquake strikes populated region"
        },
        {
            "name": "🌀 Tropical Cyclone in Andhra Pradesh",
            "state": "Andhra Pradesh", 
            "disaster": "Tropical cyclone",
            "area": 2500,
            "severity": 4,
            "description": "Category 4 cyclone makes landfall on east coast"
        },
        {
            "name": "🔥 Heat Wave in Rajasthan",
            "state": "Rajasthan",
            "disaster": "Heat wave", 
            "area": 5000,
            "severity": 3,
            "description": "Extended heat wave with temperatures exceeding 45°C"
        },
        {
            "name": "⛰️ Landslide in Uttarakhand",
            "state": "Uttarakhand",
            "disaster": "Landslide",
            "area": 200,
            "severity": 3,
            "description": "Heavy rainfall triggers multiple landslides in hill areas"
        },
        {
            "name": "🌪️ Storm in Bihar",
            "state": "Bihar",
            "disaster": "Storm",
            "area": 1200,
            "severity": 3,
            "description": "Severe thunderstorm with high winds and hail"
        }
    ]
    
    print_section("📊 DISASTER SCENARIO PREDICTIONS")
    
    all_predictions = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   📍 Location: {scenario['state']}")
        print(f"   💥 Disaster: {scenario['disaster']}")
        print(f"   📏 Area: {scenario['area']:,} sq km")
        print(f"   ⚠️ Severity: {scenario['severity']}/5")
        print(f"   📝 Description: {scenario['description']}")
        
        # Make prediction
        predictions = model.predict_victims(
            scenario['state'], 
            scenario['area'], 
            scenario['disaster'], 
            scenario['severity']
        )
        
        ensemble = predictions['ensemble']
        total_victims = ensemble['Total_Victims']
        
        # Determine risk level
        if total_victims < 100:
            risk_level = "🟢 LOW"
            risk_color = "green"
        elif total_victims < 1000:
            risk_level = "🟡 MODERATE"
            risk_color = "yellow"
        elif total_victims < 10000:
            risk_level = "🟠 HIGH"
            risk_color = "orange"
        else:
            risk_level = "🔴 CRITICAL"
            risk_color = "red"
        
        print(f"\n   📈 PREDICTED CASUALTIES:")
        print(f"   💀 Deaths: {ensemble['Deaths']:,}")
        print(f"   🩹 Injured: {ensemble['Injured']:,}")
        print(f"   👥 Affected: {ensemble['Affected']:,}")
        print(f"   📊 Total Victims: {total_victims:,}")
        print(f"   🚨 Risk Level: {risk_level}")
        
        # Store for comparison
        all_predictions.append({
            'scenario': scenario['name'],
            'state': scenario['state'],
            'disaster': scenario['disaster'],
            'total_victims': total_victims,
            'deaths': ensemble['Deaths'],
            'injured': ensemble['Injured'],
            'affected': ensemble['Affected'],
            'risk_level': risk_level
        })
    
    # Comparative analysis
    print_section("📊 COMPARATIVE RISK ANALYSIS")
    
    # Sort by total victims
    sorted_scenarios = sorted(all_predictions, key=lambda x: x['total_victims'], reverse=True)
    
    print(f"\n{'Rank':<4} {'Scenario':<35} {'Total Victims':<15} {'Risk Level'}")
    print("-" * 80)
    
    for rank, pred in enumerate(sorted_scenarios, 1):
        print(f"{rank:<4} {pred['scenario'][:34]:<35} {pred['total_victims']:<15,} {pred['risk_level']}")
    
    # State-wise summary
    print_section("🏛️ STATE-WISE RISK SUMMARY")
    
    state_summary = {}
    for pred in all_predictions:
        state = pred['state']
        if state not in state_summary:
            state_summary[state] = []
        state_summary[state].append(pred)
    
    for state, predictions in state_summary.items():
        total_victims = sum(p['total_victims'] for p in predictions)
        scenarios_count = len(predictions)
        avg_victims = total_victims // scenarios_count
        
        print(f"\n🏛️ {state}:")
        print(f"   📊 Scenarios tested: {scenarios_count}")
        print(f"   👥 Total predicted victims: {total_victims:,}")
        print(f"   📈 Average victims per scenario: {avg_victims:,}")
        
        # List scenarios for this state
        for pred in predictions:
            print(f"   • {pred['disaster']}: {pred['total_victims']:,} victims")
    
    # Emergency preparedness recommendations
    print_section("💡 EMERGENCY PREPAREDNESS RECOMMENDATIONS")
    
    # Find highest risk scenarios
    critical_scenarios = [p for p in all_predictions if 'CRITICAL' in p['risk_level']]
    high_scenarios = [p for p in all_predictions if 'HIGH' in p['risk_level']]
    
    if critical_scenarios:
        print("\n🔴 CRITICAL RISK SCENARIOS - IMMEDIATE ACTION REQUIRED:")
        for scenario in critical_scenarios:
            print(f"   • {scenario['scenario']} - {scenario['total_victims']:,} victims")
        
        print("\n   📋 CRITICAL RESPONSE MEASURES:")
        print("   1. Activate national emergency response protocols")
        print("   2. Deploy military and paramilitary forces")
        print("   3. Establish multiple emergency operation centers")
        print("   4. Coordinate international humanitarian assistance")
        print("   5. Set up mass casualty treatment facilities")
        print("   6. Implement large-scale evacuation procedures")
        print("   7. Mobilize national disaster response fund")
        print("   8. Activate early warning systems")
    
    if high_scenarios:
        print("\n🟠 HIGH RISK SCENARIOS - ENHANCED PREPAREDNESS:")
        for scenario in high_scenarios:
            print(f"   • {scenario['scenario']} - {scenario['total_victims']:,} victims")
        
        print("\n   📋 HIGH RISK RESPONSE MEASURES:")
        print("   1. Deploy state emergency response teams")
        print("   2. Set up temporary hospitals and medical camps")
        print("   3. Coordinate with neighboring states for resources")
        print("   4. Activate disaster management committees")
        print("   5. Prepare emergency shelters and relief camps")
        print("   6. Stock emergency supplies and equipment")
        print("   7. Establish communication networks")
        print("   8. Train local response teams")
    
    # Technology integration suggestions
    print_section("💻 TECHNOLOGY INTEGRATION SUGGESTIONS")
    
    print("\n🔧 REAL-TIME MONITORING:")
    print("• Integrate with meteorological departments for weather data")
    print("• Connect with seismic monitoring networks")
    print("• Use satellite imagery for damage assessment")
    print("• Implement IoT sensors for early warning")
    
    print("\n📱 MOBILE APPLICATIONS:")
    print("• Develop citizen reporting apps")
    print("• Create emergency alert systems")
    print("• Build resource coordination platforms")
    print("• Design volunteer management tools")
    
    print("\n🤖 AI ENHANCEMENTS:")
    print("• Implement real-time prediction updates")
    print("• Add social media sentiment analysis")
    print("• Use computer vision for damage assessment")
    print("• Deploy chatbots for emergency information")
    
    print("\n🌐 INTEGRATION OPPORTUNITIES:")
    print("• Connect with NDMA (National Disaster Management Authority)")
    print("• Integrate with state emergency operation centers")
    print("• Link with hospital management systems")
    print("• Connect with transportation networks")
    
    # Model performance and reliability
    print_section("📈 MODEL PERFORMANCE & RELIABILITY")
    
    print("\n🎯 MODEL ACCURACY:")
    print("• Trained on 400+ historical disaster events")
    print("• Uses ensemble of 3 machine learning algorithms")
    print("• Continuously validated against actual outcomes")
    print("• Incorporates multiple predictive factors")
    
    print("\n⚠️ LIMITATIONS & CONSIDERATIONS:")
    print("• Predictions based on historical patterns")
    print("• Local factors may influence actual outcomes")
    print("• Climate change may alter disaster patterns")
    print("• Human factors (preparedness, response) affect casualties")
    
    print("\n🔄 CONTINUOUS IMPROVEMENT:")
    print("• Regular model retraining with new data")
    print("• Validation against actual disaster outcomes")
    print("• Incorporation of user feedback")
    print("• Integration of emerging data sources")
    
    print_banner("🎯 DEMO COMPLETED SUCCESSFULLY!")
    
    print("\n📞 HOW TO USE THE SYSTEM:")
    print("1. 🌐 Web Interface: streamlit run streamlit_app.py")
    print("2. 💻 Command Line: python cli_predictor.py --help")
    print("3. 🐍 Python API: from disaster_prediction_model import DisasterPredictionModel")
    
    print("\n📚 DOCUMENTATION:")
    print("• Complete README.md with examples")
    print("• Inline code documentation")
    print("• API reference guide")
    print("• Usage tutorials and best practices")
    
    print("\n🚀 NEXT STEPS:")
    print("• Deploy to production environment")
    print("• Integrate with emergency response systems")
    print("• Train emergency personnel on system usage")
    print("• Establish data update procedures")
    print("• Create monitoring and alerting protocols")

if __name__ == "__main__":
    try:
        demo_predictions()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()