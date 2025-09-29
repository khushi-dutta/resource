"""
Simple Test Script for India Disaster Prediction System
======================================================

Quick test of the disaster prediction functionality
"""

from disaster_prediction_model import DisasterPredictionModel
import os

def main():
    print("🚨 INDIA DISASTER PREDICTION SYSTEM - QUICK TEST")
    print("=" * 55)
    
    # Test scenarios
    test_cases = [
        ("West Bengal", "Flood", 1000, 4),
        ("Gujarat", "Earthquake", 500, 5), 
        ("Andhra Pradesh", "Tropical cyclone", 2000, 4),
        ("Rajasthan", "Heat wave", 3000, 3),
        ("Uttarakhand", "Landslide", 200, 3)
    ]
    
    # Load model
    data_path = r"c:\Clg projects\SIH\public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv"
    model = DisasterPredictionModel(data_path)
    
    if os.path.exists("disaster_prediction_model.pkl"):
        model.load_model("disaster_prediction_model.pkl")
        print("✅ Model loaded successfully!\n")
    else:
        print("❌ Model not found. Please run disaster_prediction_model.py first\n")
        return
    
    # Test predictions
    for i, (state, disaster, area, severity) in enumerate(test_cases, 1):
        print(f"{i}. Predicting {disaster} in {state}")
        print(f"   Area: {area:,} sq km, Severity: {severity}/5")
        
        try:
            predictions = model.predict_victims(state, area, disaster, severity)
            ensemble = predictions['ensemble']
            
            print(f"   👥 Total Victims: {ensemble['Total_Victims']:,}")
            print(f"   💀 Deaths: {ensemble['Deaths']:,}")
            print(f"   🩹 Injured: {ensemble['Injured']:,}")
            print(f"   📊 Affected: {ensemble['Affected']:,}")
            
            # Risk level
            total = ensemble['Total_Victims']
            if total < 100:
                risk = "🟢 LOW"
            elif total < 1000:
                risk = "🟡 MODERATE"
            elif total < 10000:
                risk = "🟠 HIGH"
            else:
                risk = "🔴 CRITICAL"
            
            print(f"   🚨 Risk: {risk}")
            
            # Show key resource requirements
            if 'resources' in predictions:
                resources = predictions['resources']
                print(f"   📦 Key Resources:")
                print(f"      Medical: {resources['medical']['doctors']} doctors, {resources['medical']['nurses']} nurses")
                print(f"      Shelter: {resources['shelter']['tents']} tents, {resources['shelter']['blankets']:,} blankets")
                print(f"      Food: {resources['food_water']['food_kg']:,} kg, Water: {resources['food_water']['water_liters']:,} L")
                print(f"      Personnel: {resources['personnel']['search_rescue_teams']} rescue teams")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}\n")
    
    print("🎯 Test completed!")
    print("\nTo run the web interface: streamlit run streamlit_app.py")
    print("To use CLI: python cli_predictor.py --help")

if __name__ == "__main__":
    main()