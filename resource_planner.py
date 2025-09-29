"""
Disaster Resource Planning Tool
==============================

Comprehensive resource planning and optimization for disaster response
"""

import pandas as pd
import numpy as np
from disaster_prediction_model import DisasterPredictionModel
import json

class DisasterResourcePlanner:
    def __init__(self, model: DisasterPredictionModel):
        """Initialize resource planner with trained model"""
        self.model = model
        self.resource_costs = self._initialize_resource_costs()
        self.resource_suppliers = self._initialize_suppliers()
    
    def _initialize_resource_costs(self):
        """Initialize cost estimates for different resources (in INR)"""
        return {
            'medical': {
                'doctor_per_day': 5000,
                'nurse_per_day': 2000,
                'paramedic_per_day': 1500,
                'ambulance_per_day': 3000,
                'stretcher': 2500,
                'oxygen_cylinder': 8000,
                'blood_unit': 1500,
                'bandages_per_roll': 50,
                'antiseptic_per_liter': 200,
                'pain_medication_per_tablet': 5,
                'antibiotics_per_dose': 25,
                'iv_fluids_per_bag': 150
            },
            'shelter': {
                'tent': 15000,
                'blanket': 500,
                'mattress': 2000,
                'tarpaulin': 800,
                'portable_toilet': 25000,
                'cooking_stove': 3000,
                'gas_cylinder': 800,
                'clothing_set': 1000
            },
            'food_water': {
                'rice_per_kg': 40,
                'dal_per_kg': 80,
                'vegetables_per_kg': 30,
                'oil_per_liter': 120,
                'water_per_liter': 2,
                'baby_food_per_kg': 300,
                'water_purification_tablet': 1,
                'water_container': 500
            },
            'personnel': {
                'search_rescue_specialist_per_day': 3000,
                'police_officer_per_day': 2000,
                'coordinator_per_day': 2500,
                'volunteer_per_day': 500,
                'translator_per_day': 1500,
                'sniffer_dog_per_day': 1000
            },
            'logistics': {
                'bus_per_day': 5000,
                'truck_per_day': 4000,
                'fuel_per_liter': 100,
                'excavator_per_day': 15000,
                'crane_per_day': 20000,
                'boat_per_day': 3000,
                'water_pump_per_day': 2000,
                'generator_per_day': 1500
            },
            'equipment': {
                'satellite_phone': 50000,
                'walkie_talkie': 8000,
                'generator': 75000,
                'solar_panel': 25000,
                'led_floodlight': 5000,
                'first_aid_kit': 2000,
                'fire_extinguisher': 3000,
                'life_jacket': 1500,
                'gas_mask': 2500
            }
        }
    
    def _initialize_suppliers(self):
        """Initialize supplier information for different resources"""
        return {
            'medical': {
                'primary': ['Indian Red Cross Society', 'Government Hospitals', 'Private Medical Suppliers'],
                'emergency': ['Armed Forces Medical Services', 'International Medical Aid'],
                'contact_time': '2-4 hours'
            },
            'shelter': {
                'primary': ['NDRF', 'State Disaster Response Force', 'NGOs'],
                'emergency': ['Military Engineering Services', 'International Relief Agencies'],
                'contact_time': '4-8 hours'
            },
            'food_water': {
                'primary': ['Food Corporation of India', 'State Civil Supplies', 'Local Suppliers'],
                'emergency': ['UN World Food Programme', 'International Food Aid'],
                'contact_time': '1-3 hours'
            },
            'personnel': {
                'primary': ['NDRF', 'State Police', 'Fire Services'],
                'emergency': ['Armed Forces', 'Paramilitary Forces'],
                'contact_time': '1-2 hours'
            },
            'logistics': {
                'primary': ['State Transport Corporation', 'Private Transporters'],
                'emergency': ['Military Transport', 'Railway Emergency Services'],
                'contact_time': '2-6 hours'
            }
        }
    
    def generate_resource_plan(self, state, disaster_type, area, severity, duration_days=7):
        """Generate comprehensive resource plan"""
        print(f"🎯 GENERATING RESOURCE PLAN")
        print("=" * 40)
        print(f"Location: {state}")
        print(f"Disaster: {disaster_type}")
        print(f"Area: {area:,} sq km")
        print(f"Severity: {severity}/5")
        print(f"Duration: {duration_days} days")
        
        # Get predictions
        predictions = self.model.predict_victims(state, area, disaster_type, severity)
        ensemble = predictions['ensemble']
        resources = predictions['resources']
        
        # Calculate costs
        costs = self._calculate_resource_costs(resources, duration_days)
        
        # Generate procurement timeline
        timeline = self._generate_procurement_timeline(resources, disaster_type, severity)
        
        # Create deployment plan
        deployment = self._create_deployment_plan(resources, state, area)
        
        return {
            'predictions': ensemble,
            'resources': resources,
            'costs': costs,
            'timeline': timeline,
            'deployment': deployment,
            'suppliers': self.resource_suppliers
        }
    
    def _calculate_resource_costs(self, resources, duration_days):
        """Calculate total costs for all resources"""
        costs = {}
        total_cost = 0
        
        for category, items in resources.items():
            category_cost = 0
            costs[category] = {}
            
            if category in self.resource_costs:
                for item, quantity in items.items():
                    if item in self.resource_costs[category]:
                        unit_cost = self.resource_costs[category][item]
                        
                        # Adjust for daily costs
                        if 'per_day' in item:
                            item_cost = unit_cost * quantity * duration_days
                        else:
                            item_cost = unit_cost * quantity
                        
                        costs[category][item] = {
                            'quantity': quantity,
                            'unit_cost': unit_cost,
                            'total_cost': item_cost,
                            'duration_days': duration_days if 'per_day' in item else 1
                        }
                        category_cost += item_cost
            
            costs[category]['total'] = category_cost
            total_cost += category_cost
        
        costs['grand_total'] = total_cost
        return costs
    
    def _generate_procurement_timeline(self, resources, disaster_type, severity):
        """Generate procurement and deployment timeline"""
        timeline = []
        
        # Immediate response (0-2 hours)
        immediate = []
        if disaster_type.lower() in ['earthquake', 'explosion', 'building_collapse']:
            immediate.extend(['search_rescue_teams', 'sniffer_dogs', 'ambulances', 'doctors'])
        if disaster_type.lower() in ['flood', 'tsunami']:
            immediate.extend(['boats', 'life_jackets', 'water_rescue_specialists'])
        immediate.extend(['police_officers', 'coordinators', 'satellite_phones'])
        
        timeline.append({
            'phase': 'Immediate Response',
            'timeframe': '0-2 hours',
            'priority': 'CRITICAL',
            'resources': immediate,
            'description': 'Life-saving operations and initial response'
        })
        
        # Short-term response (2-8 hours)
        short_term = ['medical_supplies', 'tents', 'blankets', 'food_water_emergency', 
                     'generators', 'communication_equipment', 'transportation']
        
        timeline.append({
            'phase': 'Short-term Response',
            'timeframe': '2-8 hours',
            'priority': 'HIGH',
            'resources': short_term,
            'description': 'Emergency shelter, medical care, and basic needs'
        })
        
        # Medium-term response (8-24 hours)
        medium_term = ['full_medical_facilities', 'mass_shelter', 'food_distribution',
                      'sanitation_facilities', 'heavy_machinery', 'volunteer_coordination']
        
        timeline.append({
            'phase': 'Medium-term Response',
            'timeframe': '8-24 hours',
            'priority': 'MEDIUM',
            'resources': medium_term,
            'description': 'Expanded operations and organized relief'
        })
        
        # Long-term response (1-7 days)
        long_term = ['temporary_housing', 'full_food_supply', 'medical_follow_up',
                    'psychological_support', 'infrastructure_repair', 'recovery_planning']
        
        timeline.append({
            'phase': 'Long-term Response',
            'timeframe': '1-7 days',
            'priority': 'MEDIUM',
            'resources': long_term,
            'description': 'Sustained relief and early recovery'
        })
        
        return timeline
    
    def _create_deployment_plan(self, resources, state, area):
        """Create resource deployment plan"""
        deployment = {
            'command_centers': max(1, int(area / 5000)),  # 1 per 5000 sq km
            'distribution_points': max(3, int(area / 1000)),  # 1 per 1000 sq km
            'medical_posts': max(2, resources['medical']['doctors'] // 3),
            'shelter_camps': max(1, resources['shelter']['tents'] // 50),
            'coordination_hubs': max(1, int(area / 10000))
        }
        
        deployment['deployment_strategy'] = {
            'urban_areas': '40% of resources for high-density areas',
            'rural_areas': '35% of resources for scattered populations',
            'transportation_hubs': '15% of resources for logistics',
            'reserve': '10% kept as strategic reserve'
        }
        
        return deployment
    
    def generate_resource_report(self, plan, output_file=None):
        """Generate comprehensive resource report"""
        report = []
        
        report.append("=" * 80)
        report.append(" DISASTER RESOURCE PLANNING REPORT")
        report.append("=" * 80)
        
        # Predictions Summary
        pred = plan['predictions']
        report.append(f"\n📊 CASUALTY PREDICTIONS:")
        report.append(f"   Deaths: {pred['Deaths']:,}")
        report.append(f"   Injured: {pred['Injured']:,}")
        report.append(f"   Affected: {pred['Affected']:,}")
        report.append(f"   Total Victims: {pred['Total_Victims']:,}")
        
        # Resource Requirements
        report.append(f"\n📦 RESOURCE REQUIREMENTS:")
        for category, items in plan['resources'].items():
            report.append(f"\n   {category.upper().replace('_', ' ')}:")
            for item, quantity in items.items():
                if isinstance(quantity, (int, float)):
                    report.append(f"      {item.replace('_', ' ').title()}: {quantity:,}")
        
        # Cost Analysis
        costs = plan['costs']
        report.append(f"\n💰 COST ANALYSIS:")
        report.append(f"   Total Estimated Cost: ₹{costs['grand_total']:,.2f}")
        
        for category, category_costs in costs.items():
            if category != 'grand_total' and isinstance(category_costs, dict) and 'total' in category_costs:
                report.append(f"   {category.replace('_', ' ').title()}: ₹{category_costs['total']:,.2f}")
        
        # Procurement Timeline
        report.append(f"\n⏰ PROCUREMENT TIMELINE:")
        for phase in plan['timeline']:
            report.append(f"\n   {phase['phase']} ({phase['timeframe']}):")
            report.append(f"      Priority: {phase['priority']}")
            report.append(f"      Description: {phase['description']}")
            report.append(f"      Key Resources: {', '.join(phase['resources'][:5])}")
        
        # Deployment Plan
        deployment = plan['deployment']
        report.append(f"\n🎯 DEPLOYMENT PLAN:")
        report.append(f"   Command Centers: {deployment['command_centers']}")
        report.append(f"   Distribution Points: {deployment['distribution_points']}")
        report.append(f"   Medical Posts: {deployment['medical_posts']}")
        report.append(f"   Shelter Camps: {deployment['shelter_camps']}")
        
        # Supplier Information
        report.append(f"\n📞 KEY SUPPLIERS:")
        for category, supplier_info in plan['suppliers'].items():
            report.append(f"\n   {category.replace('_', ' ').title()}:")
            report.append(f"      Primary: {', '.join(supplier_info['primary'])}")
            report.append(f"      Contact Time: {supplier_info['contact_time']}")
        
        report_text = '\n'.join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to {output_file}")
        
        return report_text
    
    def optimize_resource_allocation(self, multiple_scenarios):
        """Optimize resource allocation across multiple disaster scenarios"""
        print("🔧 OPTIMIZING RESOURCE ALLOCATION")
        print("=" * 40)
        
        total_resources = {}
        total_cost = 0
        scenarios = []
        
        for scenario in multiple_scenarios:
            plan = self.generate_resource_plan(**scenario)
            scenarios.append(plan)
            
            # Aggregate resources
            for category, items in plan['resources'].items():
                if category not in total_resources:
                    total_resources[category] = {}
                
                for item, quantity in items.items():
                    if isinstance(quantity, (int, float)):
                        if item not in total_resources[category]:
                            total_resources[category][item] = 0
                        total_resources[category][item] += quantity
            
            total_cost += plan['costs']['grand_total']
        
        # Calculate savings from bulk procurement
        bulk_savings = total_cost * 0.15  # 15% savings from bulk orders
        optimized_cost = total_cost - bulk_savings
        
        optimization_report = {
            'scenarios_count': len(scenarios),
            'total_resources': total_resources,
            'original_cost': total_cost,
            'bulk_savings': bulk_savings,
            'optimized_cost': optimized_cost,
            'savings_percentage': (bulk_savings / total_cost) * 100
        }
        
        print(f"   Scenarios Analyzed: {len(scenarios)}")
        print(f"   Original Total Cost: ₹{total_cost:,.2f}")
        print(f"   Bulk Procurement Savings: ₹{bulk_savings:,.2f}")
        print(f"   Optimized Cost: ₹{optimized_cost:,.2f}")
        print(f"   Savings: {(bulk_savings / total_cost) * 100:.1f}%")
        
        return optimization_report

def main():
    """Demonstrate resource planning functionality"""
    print("🎯 DISASTER RESOURCE PLANNER DEMO")
    print("=" * 50)
    
    # Initialize model and planner
    data_path = r"c:\Clg projects\SIH\public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv"
    model = DisasterPredictionModel(data_path)
    model.load_model("disaster_prediction_model.pkl")
    
    planner = DisasterResourcePlanner(model)
    
    # Generate resource plan for a major flood
    plan = planner.generate_resource_plan(
        state="West Bengal",
        disaster_type="Flood", 
        area=2000,
        severity=4,
        duration_days=10
    )
    
    # Generate comprehensive report
    report = planner.generate_resource_report(plan, "resource_plan_report.txt")
    print("\n" + report[:1000] + "...")  # Show first 1000 characters
    
    # Demonstrate multi-scenario optimization
    scenarios = [
        {"state": "West Bengal", "disaster_type": "Flood", "area": 1500, "severity": 4},
        {"state": "Gujarat", "disaster_type": "Earthquake", "area": 800, "severity": 5},
        {"state": "Andhra Pradesh", "disaster_type": "Tropical cyclone", "area": 2000, "severity": 4}
    ]
    
    optimization = planner.optimize_resource_allocation(scenarios)
    
    print(f"\n🎉 Resource planning demonstration completed!")
    print(f"📄 Detailed report saved as 'resource_plan_report.txt'")

if __name__ == "__main__":
    main()