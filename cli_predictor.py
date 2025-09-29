"""
Command Line Interface for Disaster Prediction Model
===================================================

Simple CLI tool for quick disaster victim predictions
"""

import argparse
import sys
import os
from disaster_prediction_model import DisasterPredictionModel

def main():
    parser = argparse.ArgumentParser(
        description="Predict disaster victims in India",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_predictor.py --state "West Bengal" --disaster "Flood" --area 1000 --severity 4
  python cli_predictor.py --state "Gujarat" --disaster "Earthquake" --area 500 --severity 5
  python cli_predictor.py --state "Andhra Pradesh" --disaster "Tropical cyclone" --area 2000 --severity 4
        """
    )
    
    parser.add_argument(
        "--state", 
        required=True,
        help="Indian state name (e.g., 'West Bengal', 'Gujarat')"
    )
    
    parser.add_argument(
        "--disaster", 
        required=True,
        help="Disaster type (e.g., 'Flood', 'Earthquake', 'Tropical cyclone')"
    )
    
    parser.add_argument(
        "--area", 
        type=int, 
        required=True,
        help="Affected area in square kilometers"
    )
    
    parser.add_argument(
        "--severity", 
        type=int, 
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Severity level (1=Very Low, 2=Low, 3=Moderate, 4=High, 5=Very High)"
    )
    
    parser.add_argument(
        "--model-path",
        default="disaster_prediction_model.pkl",
        help="Path to saved model file"
    )
    
    parser.add_argument(
        "--data-path",
        default=r"c:\Clg projects\SIH\public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv",
        help="Path to disaster data CSV file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.area <= 0:
        print("Error: Area must be positive")
        sys.exit(1)
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)
    
    print("🚨 India Disaster Prediction System")
    print("=" * 40)
    
    try:
        # Initialize model
        if args.verbose:
            print("Loading disaster prediction model...")
        
        model = DisasterPredictionModel(args.data_path)
        
        # Load or train model
        if os.path.exists(args.model_path):
            model.load_model(args.model_path)
            if args.verbose:
                print("✅ Model loaded from saved file")
        else:
            if args.verbose:
                print("Training new model (this may take a few minutes)...")
            model.load_and_preprocess_data()
            model.train_models()
            model.save_model(args.model_path)
            if args.verbose:
                print("✅ Model trained and saved")
        
        # Make prediction
        if args.verbose:
            print(f"\nMaking prediction for:")
            print(f"  State: {args.state}")
            print(f"  Disaster: {args.disaster}")
            print(f"  Area: {args.area:,} sq km")
            print(f"  Severity: {args.severity}")
            print()
        
        predictions = model.predict_victims(args.state, args.area, args.disaster, args.severity)
        
        # Display results
        print("📊 PREDICTION RESULTS")
        print("-" * 25)
        
        ensemble = predictions['ensemble']
        
        print(f"💀 Deaths:        {ensemble['Deaths']:,}")
        print(f"🩹 Injured:       {ensemble['Injured']:,}")
        print(f"👥 Affected:      {ensemble['Affected']:,}")
        print(f"📈 Total Victims: {ensemble['Total_Victims']:,}")
        
        # Show resource requirements
        if 'resources' in predictions:
            resources = predictions['resources']
            print(f"\n📦 REQUIRED RESOURCES")
            print("-" * 22)
            
            # Medical resources
            medical = resources['medical']
            print(f"🏥 Medical:")
            print(f"   Doctors: {medical['doctors']}, Nurses: {medical['nurses']}")
            print(f"   Ambulances: {medical['ambulances']}, Stretchers: {medical['stretchers']}")
            
            # Shelter resources  
            shelter = resources['shelter']
            print(f"🏠 Shelter:")
            print(f"   Tents: {shelter['tents']}, Blankets: {shelter['blankets']:,}")
            print(f"   Portable Toilets: {shelter['portable_toilets']}")
            
            # Food and water
            food_water = resources['food_water']
            print(f"🍽️ Food & Water:")
            print(f"   Food: {food_water['food_kg']:,} kg, Water: {food_water['water_liters']:,} L")
            print(f"   Rice: {food_water['rice_kg']:,} kg, Dal: {food_water['dal_kg']:,} kg")
            
            # Personnel
            personnel = resources['personnel']
            print(f"👮 Personnel:")
            print(f"   Rescue Teams: {personnel['search_rescue_teams']}")
            print(f"   Police: {personnel['police_officers']}, Volunteers: {personnel['trained_volunteers']}")
            
            # Logistics
            logistics = resources['logistics']
            print(f"🚛 Logistics:")
            print(f"   Buses: {logistics['buses']}, Trucks: {logistics['trucks']}")
            if 'excavators' in logistics:
                print(f"   Excavators: {logistics['excavators']}, Cranes: {logistics['cranes']}")
            if 'boats' in logistics:
                print(f"   Boats: {logistics['boats']}, Water Pumps: {logistics['water_pumps']}")
        
        # Risk assessment
        total_victims = ensemble['Total_Victims']
        
        if total_victims < 100:
            risk_level = "🟢 LOW RISK"
        elif total_victims < 1000:
            risk_level = "🟡 MODERATE RISK"
        elif total_victims < 10000:
            risk_level = "🟠 HIGH RISK"
        else:
            risk_level = "🔴 CRITICAL RISK"
        
        print(f"\n⚠️  Risk Level: {risk_level}")
        
        # Show model comparison if verbose
        if args.verbose:
            print("\n🔬 MODEL COMPARISON")
            print("-" * 20)
            for model_name, pred in predictions.items():
                if model_name != 'ensemble':
                    print(f"{model_name.replace('_', ' ').title():15} | "
                          f"Deaths: {pred['Deaths']:6,} | "
                          f"Injured: {pred['Injured']:8,} | "
                          f"Affected: {pred['Affected']:9,} | "
                          f"Total: {pred['Total_Victims']:8,}")
        
        # Emergency recommendations
        print("\n💡 EMERGENCY RECOMMENDATIONS")
        print("-" * 30)
        
        recommendations = generate_emergency_recommendations(args.disaster, args.severity, total_victims)
        
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"{i}. {rec}")
        
        if args.verbose and len(recommendations) > 5:
            print("\nAdditional recommendations:")
            for i, rec in enumerate(recommendations[5:], 6):
                print(f"{i}. {rec}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def generate_emergency_recommendations(disaster_type, severity_level, total_victims):
    """Generate emergency response recommendations"""
    recommendations = []
    
    # Disaster-specific recommendations
    disaster_lower = disaster_type.lower()
    
    if 'flood' in disaster_lower:
        recommendations.extend([
            "Activate flood warning systems immediately",
            "Prepare boats and rescue equipment",
            "Evacuate low-lying areas",
            "Set up temporary shelters on higher ground",
            "Monitor dam and embankment conditions"
        ])
    elif 'earthquake' in disaster_lower:
        recommendations.extend([
            "Deploy search and rescue teams",
            "Inspect critical infrastructure for damage",
            "Set up medical triage centers",
            "Prepare for potential aftershocks",
            "Check gas lines and electrical systems"
        ])
    elif 'cyclone' in disaster_lower or 'storm' in disaster_lower:
        recommendations.extend([
            "Issue early warning to coastal areas",
            "Evacuate vulnerable populations",
            "Secure loose objects and structures",
            "Prepare emergency shelters",
            "Stock emergency supplies"
        ])
    elif 'drought' in disaster_lower:
        recommendations.extend([
            "Implement water conservation measures",
            "Distribute emergency water supplies",
            "Provide agricultural assistance",
            "Monitor food security situation",
            "Plan for livestock welfare"
        ])
    elif 'landslide' in disaster_lower:
        recommendations.extend([
            "Evacuate areas prone to further slides",
            "Deploy heavy machinery for rescue",
            "Monitor slope stability",
            "Block access to dangerous areas",
            "Set up emergency shelters"
        ])
    else:
        recommendations.extend([
            "Activate emergency response protocols",
            "Deploy appropriate rescue teams",
            "Set up emergency shelters",
            "Monitor situation continuously",
            "Coordinate with local authorities"
        ])
    
    # Severity-based recommendations
    if severity_level >= 4:
        recommendations.extend([
            "Declare state of emergency",
            "Request national assistance",
            "Deploy military resources",
            "Coordinate international aid if needed"
        ])
    
    # Victim count-based recommendations
    if total_victims >= 10000:
        recommendations.extend([
            "Establish multiple emergency centers",
            "Set up mass casualty facilities",
            "Coordinate media and public information",
            "Prepare for extended operations"
        ])
    elif total_victims >= 1000:
        recommendations.extend([
            "Deploy additional medical teams",
            "Set up temporary hospitals",
            "Coordinate with neighboring regions"
        ])
    
    # General recommendations
    recommendations.extend([
        "Maintain communication systems",
        "Monitor weather conditions",
        "Coordinate volunteer efforts",
        "Prepare public information campaigns",
        "Document damage for recovery planning"
    ])
    
    return recommendations

if __name__ == "__main__":
    main()