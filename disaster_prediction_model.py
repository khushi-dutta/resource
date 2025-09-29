"""
Disaster Victim Prediction Model for India
==========================================

This machine learning model predicts the number of victims (deaths, injuries, affected people)
for natural disasters in India based on:
- Location (state/district)
- Area affected
- Disaster type
- Severity level (derived from magnitude)

Features:
- Data preprocessing and feature engineering
- Multiple ML algorithms (Random Forest, XGBoost, Neural Network)
- Location-based area calculation
- Severity level mapping
- Interactive prediction interface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Neural network training will be skipped.")
    TENSORFLOW_AVAILABLE = False
import joblib
import warnings
warnings.filterwarnings('ignore')

class DisasterPredictionModel:
    def __init__(self, data_path):
        """Initialize the disaster prediction model"""
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Indian states and their approximate areas (in sq km)
        self.state_areas = {
            'Rajasthan': 342239, 'Madhya Pradesh': 308245, 'Maharashtra': 307713,
            'Uttar Pradesh': 240928, 'Gujarat': 196244, 'Karnataka': 191791,
            'Andhra Pradesh': 162968, 'Orissa': 155707, 'Chhattisgarh': 135191,
            'Tamil Nadu': 130060, 'Telangana': 112077, 'Bihar': 94163,
            'West Bengal': 88752, 'Arunachal Pradesh': 83743, 'Jharkhand': 79714,
            'Assam': 78438, 'Himachal Pradesh': 55673, 'Uttarakhand': 53483,
            'Punjab': 50362, 'Haryana': 44212, 'Kerala': 38852,
            'Meghalaya': 22327, 'Manipur': 22327, 'Mizoram': 21081,
            'Nagaland': 16579, 'Tripura': 10486, 'Sikkim': 7096,
            'Delhi': 1484, 'Chandigarh': 114, 'Dadra and Nagar Haveli': 491,
            'Daman and Diu': 112, 'Lakshadweep': 32, 'Puducherry': 479
        }
        
    def load_and_preprocess_data(self):
        """Load and preprocess the disaster data"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} disaster records")
        
        # Display basic info
        print(f"Date range: {self.df['Start Year'].min()} - {self.df['Start Year'].max()}")
        print(f"Disaster types: {self.df['Disaster Type'].nunique()}")
        
        # Clean and prepare data
        self.df = self._clean_data()
        self.df = self._feature_engineering()
        
        return self.df
    
    def _clean_data(self):
        """Clean the dataset"""
        df = self.df.copy()
        
        # Fill missing values
        numeric_columns = ['Total Deaths', 'No. Injured', 'No. Affected', 'Magnitude']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Clean location data
        df['Location'] = df['Location'].fillna('Unknown')
        df['Disaster Type'] = df['Disaster Type'].fillna('Unknown')
        df['Magnitude Scale'] = df['Magnitude Scale'].fillna('Unknown')
        
        # Create total victims column
        df['Total Victims'] = df['Total Deaths'] + df['No. Injured'] + df['No. Affected']
        
        # Filter out records with no victims (for training purposes)
        df = df[df['Total Victims'] > 0]
        
        print(f"After cleaning: {len(df)} records with victims")
        return df
    
    def _feature_engineering(self):
        """Create engineered features"""
        df = self.df.copy()
        
        # Extract state from location
        df['State'] = df['Location'].apply(self._extract_state)
        
        # Calculate area affected (estimate based on location)
        df['Estimated_Area'] = df['State'].map(self.state_areas).fillna(100000)  # Default area
        
        # Create severity levels based on magnitude and disaster type
        df['Severity_Level'] = df.apply(self._calculate_severity, axis=1)
        
        # Create time-based features
        df['Month'] = df['Start Month'].fillna(6)  # Default to monsoon season
        df['Monsoon_Season'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
        
        # Create disaster category groups
        df['Disaster_Category'] = df['Disaster Type'].apply(self._categorize_disaster)
        
        # Log transform for skewed target variables
        df['Log_Deaths'] = np.log1p(df['Total Deaths'])
        df['Log_Injured'] = np.log1p(df['No. Injured'])
        df['Log_Affected'] = np.log1p(df['No. Affected'])
        
        return df
    
    def calculate_required_resources(self, deaths, injured, affected, disaster_type, severity_level, duration_days=7):
        """
        Calculate required resources for disaster response
        
        Parameters:
        - deaths: Number of fatalities
        - injured: Number of injured people
        - affected: Number of affected people
        - disaster_type: Type of disaster
        - severity_level: Severity level (1-5)
        - duration_days: Expected duration of response (default 7 days)
        
        Returns:
        - Dictionary with resource requirements
        """
        resources = {}
        
        # Medical Resources
        resources['medical'] = self._calculate_medical_resources(deaths, injured, affected, disaster_type, severity_level)
        
        # Food and Water
        resources['food_water'] = self._calculate_food_water_resources(injured, affected, duration_days)
        
        # Shelter and Accommodation
        resources['shelter'] = self._calculate_shelter_resources(injured, affected, disaster_type)
        
        # Emergency Personnel
        resources['personnel'] = self._calculate_personnel_resources(deaths, injured, affected, disaster_type, severity_level)
        
        # Transportation and Logistics
        resources['logistics'] = self._calculate_logistics_resources(deaths, injured, affected, disaster_type, severity_level)
        
        # Communication and Equipment
        resources['equipment'] = self._calculate_equipment_resources(affected, disaster_type, severity_level)
        
        return resources
    
    def _calculate_medical_resources(self, deaths, injured, affected, disaster_type, severity_level):
        """Calculate medical resource requirements"""
        medical_resources = {}
        
        # Basic medical supplies per injured person
        basic_supplies_per_injured = {
            'bandages': 5,  # rolls per person
            'antiseptic': 0.5,  # liters per person
            'pain_medication': 10,  # tablets per person
            'antibiotics': 5,  # doses per person
            'iv_fluids': 2,  # bags per person
        }
        
        # Severe injury factor based on disaster type
        severity_multiplier = {
            'earthquake': 2.5,
            'flood': 1.2,
            'cyclone': 1.8,
            'landslide': 2.0,
            'fire': 3.0,
            'explosion': 3.5
        }.get(disaster_type.lower(), 1.5)
        
        # Calculate medical supplies
        for item, base_amount in basic_supplies_per_injured.items():
            medical_resources[item] = int(injured * base_amount * severity_multiplier * (severity_level / 3))
        
        # Medical personnel requirements
        medical_resources['doctors'] = max(1, int(injured / 50) + int(deaths / 20))
        medical_resources['nurses'] = max(2, int(injured / 20) + int(deaths / 10))
        medical_resources['paramedics'] = max(2, int(injured / 30))
        
        # Medical equipment
        medical_resources['ambulances'] = max(1, int(injured / 10) + int(deaths / 5))
        medical_resources['stretchers'] = max(5, int(injured / 5))
        medical_resources['oxygen_cylinders'] = max(2, int(injured / 25))
        medical_resources['defibrillators'] = max(1, medical_resources['doctors'])
        
        # Blood requirements (for severe disasters)
        if severity_level >= 4:
            medical_resources['blood_units'] = int(injured * 0.3)  # 30% may need blood
        
        return medical_resources
    
    def _calculate_food_water_resources(self, injured, affected, duration_days):
        """Calculate food and water resource requirements"""
        food_water_resources = {}
        
        # People needing food and water support
        people_needing_support = injured + int(affected * 0.7)  # 70% of affected need support
        
        # Daily requirements per person
        daily_water_liters = 4  # WHO standard: 4 liters per person per day
        daily_food_kg = 0.5  # 500g food per person per day
        
        # Calculate total requirements
        food_water_resources['water_liters'] = people_needing_support * daily_water_liters * duration_days
        food_water_resources['food_kg'] = people_needing_support * daily_food_kg * duration_days
        
        # Specific food items
        food_water_resources['rice_kg'] = int(food_water_resources['food_kg'] * 0.4)  # 40% rice
        food_water_resources['dal_kg'] = int(food_water_resources['food_kg'] * 0.2)   # 20% pulses
        food_water_resources['vegetables_kg'] = int(food_water_resources['food_kg'] * 0.2)  # 20% vegetables
        food_water_resources['oil_liters'] = int(people_needing_support * 0.05 * duration_days)  # 50ml per person per day
        
        # Baby food and special dietary needs (5% of population)
        food_water_resources['baby_food_kg'] = int(people_needing_support * 0.05 * 0.3 * duration_days)
        food_water_resources['medical_food_kg'] = int(people_needing_support * 0.1 * 0.3 * duration_days)
        
        # Water purification
        food_water_resources['water_purification_tablets'] = int(food_water_resources['water_liters'] / 20)  # 1 tablet per 20L
        food_water_resources['water_containers'] = max(10, int(people_needing_support / 50))  # Storage containers
        
        return food_water_resources
    
    def _calculate_shelter_resources(self, injured, affected, disaster_type):
        """Calculate shelter and accommodation resource requirements"""
        shelter_resources = {}
        
        # People needing temporary shelter
        if disaster_type.lower() in ['earthquake', 'flood', 'cyclone', 'fire']:
            people_needing_shelter = int(affected * 0.6)  # 60% need temporary shelter
        else:
            people_needing_shelter = int(affected * 0.3)  # 30% for other disasters
        
        # Shelter requirements
        people_per_tent = 4  # Average family size
        shelter_resources['tents'] = max(5, int(people_needing_shelter / people_per_tent))
        shelter_resources['tarpaulins'] = int(shelter_resources['tents'] * 1.5)  # Extra for repairs
        
        # Bedding and clothing
        shelter_resources['blankets'] = people_needing_shelter * 2  # 2 blankets per person
        shelter_resources['mattresses'] = people_needing_shelter
        shelter_resources['clothing_sets'] = people_needing_shelter  # 1 set per person
        
        # Sanitation facilities
        shelter_resources['portable_toilets'] = max(2, int(people_needing_shelter / 50))  # 1 per 50 people
        shelter_resources['handwashing_stations'] = max(1, int(people_needing_shelter / 100))
        
        # Cooking facilities
        shelter_resources['cooking_stoves'] = max(2, int(shelter_resources['tents'] / 5))  # 1 per 5 tents
        shelter_resources['cooking_gas_cylinders'] = shelter_resources['cooking_stoves'] * 2
        shelter_resources['cooking_utensils_sets'] = shelter_resources['tents']
        
        return shelter_resources
    
    def _calculate_personnel_resources(self, deaths, injured, affected, disaster_type, severity_level):
        """Calculate emergency personnel resource requirements"""
        personnel_resources = {}
        
        # Search and Rescue
        if disaster_type.lower() in ['earthquake', 'landslide', 'building_collapse']:
            personnel_resources['search_rescue_teams'] = max(2, int(deaths / 10) + int(injured / 50))
            personnel_resources['sniffer_dogs'] = max(1, int(personnel_resources['search_rescue_teams'] / 3))
        else:
            personnel_resources['search_rescue_teams'] = max(1, int(deaths / 20) + int(injured / 100))
            personnel_resources['sniffer_dogs'] = 0
        
        # Security Personnel
        personnel_resources['police_officers'] = max(5, int(affected / 1000) * severity_level)
        personnel_resources['security_guards'] = max(3, int(affected / 2000))
        
        # Administrative Staff
        personnel_resources['coordinators'] = max(2, int(affected / 5000))
        personnel_resources['data_entry_staff'] = max(1, int(affected / 10000))
        personnel_resources['translators'] = max(1, int(affected / 20000))
        
        # Specialized Personnel
        if disaster_type.lower() in ['chemical', 'industrial']:
            personnel_resources['hazmat_specialists'] = max(2, int(severity_level))
        
        if disaster_type.lower() in ['flood', 'tsunami']:
            personnel_resources['water_rescue_specialists'] = max(2, int(injured / 30))
            personnel_resources['boat_operators'] = max(1, int(affected / 500))
        
        # Volunteers
        personnel_resources['trained_volunteers'] = max(10, int(affected / 100))
        personnel_resources['volunteer_coordinators'] = max(1, int(personnel_resources['trained_volunteers'] / 20))
        
        return personnel_resources
    
    def _calculate_logistics_resources(self, deaths, injured, affected, disaster_type, severity_level):
        """Calculate transportation and logistics resource requirements"""
        logistics_resources = {}
        
        # Transportation
        total_people = injured + int(affected * 0.5)  # People needing transportation
        
        logistics_resources['buses'] = max(1, int(total_people / 50))  # 50 people per bus
        logistics_resources['trucks'] = max(2, int(affected / 1000))  # For supplies
        logistics_resources['fuel_liters'] = max(500, int((logistics_resources['buses'] + logistics_resources['trucks']) * 100))
        
        # Heavy machinery (for specific disasters)
        if disaster_type.lower() in ['earthquake', 'landslide']:
            logistics_resources['excavators'] = max(1, int(deaths / 50))
            logistics_resources['cranes'] = max(1, int(deaths / 100))
            logistics_resources['bulldozers'] = max(1, int(affected / 5000))
        
        if disaster_type.lower() in ['flood']:
            logistics_resources['boats'] = max(2, int(affected / 500))
            logistics_resources['water_pumps'] = max(1, int(affected / 1000))
            logistics_resources['generators'] = max(2, int(affected / 2000))
        
        # Storage and distribution
        logistics_resources['warehouses'] = max(1, int(affected / 10000))
        logistics_resources['distribution_points'] = max(2, int(affected / 5000))
        
        return logistics_resources
    
    def _calculate_equipment_resources(self, affected, disaster_type, severity_level):
        """Calculate communication and equipment resource requirements"""
        equipment_resources = {}
        
        # Communication Equipment
        equipment_resources['satellite_phones'] = max(5, int(affected / 5000))
        equipment_resources['walkie_talkies'] = max(10, int(affected / 1000))
        equipment_resources['mobile_towers'] = max(1, int(affected / 20000))
        
        # Power and Lighting
        equipment_resources['generators'] = max(3, int(affected / 2000))
        equipment_resources['solar_panels'] = max(2, int(affected / 5000))
        equipment_resources['led_floodlights'] = max(10, int(affected / 1000))
        equipment_resources['batteries'] = equipment_resources['walkie_talkies'] * 4  # 4 batteries per radio
        
        # Disaster-specific equipment
        if disaster_type.lower() in ['chemical', 'gas_leak']:
            equipment_resources['gas_masks'] = max(20, int(affected / 100))
            equipment_resources['protective_suits'] = max(10, int(affected / 500))
            equipment_resources['air_quality_monitors'] = max(2, int(severity_level))
        
        if disaster_type.lower() in ['flood', 'tsunami']:
            equipment_resources['life_jackets'] = max(50, int(affected / 50))
            equipment_resources['inflatable_boats'] = max(5, int(affected / 1000))
            equipment_resources['water_level_monitors'] = max(3, int(severity_level))
        
        # General emergency equipment
        equipment_resources['first_aid_kits'] = max(10, int(affected / 200))
        equipment_resources['fire_extinguishers'] = max(5, int(affected / 1000))
        equipment_resources['emergency_sirens'] = max(2, int(affected / 10000))
        
        return equipment_resources
    
    def _extract_state(self, location_str):
        """Extract state from location string"""
        if pd.isna(location_str) or location_str == 'Unknown':
            return 'Unknown'
        
        # Common state patterns in the location field
        for state in self.state_areas.keys():
            if state.lower() in location_str.lower():
                return state
        
        # Check for province mentions
        if 'province' in location_str.lower():
            parts = location_str.split(',')
            for part in parts:
                if 'province' in part.lower():
                    state_name = part.replace('province', '').replace('(', '').replace(')', '').strip()
                    # Try to match with known states
                    for state in self.state_areas.keys():
                        if state.lower() in state_name.lower():
                            return state
        
        return 'Unknown'
    
    def _calculate_severity(self, row):
        """Calculate severity level (1-5) based on magnitude and disaster type"""
        magnitude = row['Magnitude']
        disaster_type = row['Disaster Type']
        
        # Default severity based on disaster type
        type_severity = {
            'Earthquake': 4, 'Flood': 3, 'Cyclone': 4, 'Tropical cyclone': 4,
            'Drought': 3, 'Landslide': 3, 'Heat wave': 2, 'Cold wave': 2,
            'Epidemic': 3, 'Storm': 3, 'Wildfire': 3
        }
        
        base_severity = type_severity.get(disaster_type, 2)
        
        # Adjust based on magnitude if available
        if pd.notna(magnitude) and magnitude > 0:
            if disaster_type == 'Earthquake':
                if magnitude >= 7.0: return 5
                elif magnitude >= 6.0: return 4
                elif magnitude >= 5.0: return 3
                elif magnitude >= 4.0: return 2
                else: return 1
            elif 'Cyclone' in disaster_type or 'Storm' in disaster_type:
                if magnitude >= 150: return 5
                elif magnitude >= 120: return 4
                elif magnitude >= 90: return 3
                elif magnitude >= 60: return 2
                else: return 1
            elif 'temperature' in disaster_type.lower():
                if abs(magnitude) >= 45: return 4
                elif abs(magnitude) >= 35: return 3
                elif abs(magnitude) >= 25: return 2
                else: return 1
        
        return base_severity
    
    def _categorize_disaster(self, disaster_type):
        """Categorize disasters into broader groups"""
        if pd.isna(disaster_type):
            return 'Unknown'
        
        disaster_type = disaster_type.lower()
        
        if any(word in disaster_type for word in ['flood', 'landslide', 'mass movement']):
            return 'Hydrological'
        elif any(word in disaster_type for word in ['earthquake', 'volcanic']):
            return 'Geophysical'
        elif any(word in disaster_type for word in ['storm', 'cyclone', 'wind', 'hail', 'lightning']):
            return 'Meteorological'
        elif any(word in disaster_type for word in ['drought', 'wildfire', 'temperature']):
            return 'Climatological'
        elif any(word in disaster_type for word in ['epidemic', 'disease']):
            return 'Biological'
        else:
            return 'Other'
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("Preparing features for machine learning...")
        
        # Select features
        feature_columns = [
            'State', 'Disaster_Category', 'Severity_Level', 'Estimated_Area',
            'Monsoon_Season', 'Month', 'Magnitude'
        ]
        
        # Target variables
        target_columns = ['Total Deaths', 'No. Injured', 'No. Affected']
        
        # Create feature matrix
        X = self.df[feature_columns].copy()
        y = self.df[target_columns].copy()
        
        # Encode categorical variables
        categorical_features = ['State', 'Disaster_Category']
        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
                X[feature] = self.encoders[feature].fit_transform(X[feature].astype(str))
            else:
                X[feature] = self.encoders[feature].transform(X[feature].astype(str))
        
        # Scale features
        if 'feature_scaler' not in self.scalers:
            self.scalers['feature_scaler'] = RobustScaler()
            X_scaled = self.scalers['feature_scaler'].fit_transform(X)
        else:
            X_scaled = self.scalers['feature_scaler'].transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        return X_scaled, y
    
    def train_models(self):
        """Train multiple ML models"""
        print("Training machine learning models...")
        
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Train Neural Network
        print("Training Neural Network...")
        nn_model = self._build_neural_network(X_train.shape[1], y_train.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
        
        nn_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        self.models['neural_network'] = nn_model
        
        # Evaluate models
        self._evaluate_models(X_test, y_test)
        
        return X_test, y_test
    
    def train_models_safe(self):
        """Train models without neural network (safer for cloud deployment)"""
        print("Training machine learning models (safe mode - no neural network)...")
        
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        print("Safe training completed (Random Forest + XGBoost)")
        
        # Evaluate models
        self._evaluate_models_safe(X_test, y_test)
        
        return X_test, y_test
    
    def _build_neural_network(self, input_dim, output_dim):
        """Build neural network model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot build neural network.")
            
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\nModel Evaluation Results:")
        print("=" * 50)
        
        results = {}
        
        for model_name, model in self.models.items():
            if model_name == 'neural_network':
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
            
            print(f"\n{model_name.upper()}:")
            print(f"  MAE: {mae:.2f}")
            print(f"  MSE: {mse:.2f}")
            print(f"  R² Score: {r2:.3f}")
        
        # Find best model
        best_model = min(results.keys(), key=lambda x: results[x]['MAE'])
        print(f"\nBest Model: {best_model.upper()} (lowest MAE)")
        
        return results
    
    def _evaluate_models_safe(self, X_test, y_test):
        """Evaluate trained models (safe mode - no neural network)"""
        print("\nModel Evaluation Results (Safe Mode):")
        print("=" * 50)
        
        results = {}
        
        for model_name, model in self.models.items():
            if model_name == 'neural_network':
                continue  # Skip neural network in safe mode
                
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
            
            print(f"\n{model_name.upper()}:")
            print(f"  MAE: {mae:.2f}")
            print(f"  MSE: {mse:.2f}")
            print(f"  R² Score: {r2:.3f}")
        
        # Find best model
        if results:
            best_model = min(results.keys(), key=lambda x: results[x]['MAE'])
            print(f"\nBest Model: {best_model.upper()} (lowest MAE)")
        
        return results
    
    def predict_victims(self, location, area_sq_km, disaster_type, severity_level):
        """
        Predict number of victims for given disaster parameters
        
        Parameters:
        - location: State name in India
        - area_sq_km: Area affected in square kilometers
        - disaster_type: Type of disaster
        - severity_level: Severity level (1-5)
        
        Returns:
        - Dictionary with predictions from all models
        """
        
        # Map disaster type to category
        disaster_category = self._categorize_disaster(disaster_type)
        
        # Create input feature vector
        input_data = {
            'State': location,
            'Disaster_Category': disaster_category,
            'Severity_Level': severity_level,
            'Estimated_Area': area_sq_km,
            'Monsoon_Season': 1,  # Assume monsoon season for conservative estimate
            'Month': 7,  # July (peak monsoon)
            'Magnitude': severity_level * 2  # Rough magnitude estimate
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for feature in ['State', 'Disaster_Category']:
            if feature in self.encoders:
                try:
                    input_df[feature] = self.encoders[feature].transform(input_df[feature].astype(str))
                except ValueError:
                    # Handle unseen categories
                    input_df[feature] = 0  # Default encoding
        
        # Scale features
        input_scaled = self.scalers['feature_scaler'].transform(input_df)
        
        # Make predictions with all models
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'neural_network':
                    pred = model.predict(input_scaled)[0]
                else:
                    pred = model.predict(input_scaled)[0]
                
                predictions[model_name] = {
                    'Deaths': max(0, int(pred[0])),
                    'Injured': max(0, int(pred[1])),
                    'Affected': max(0, int(pred[2])),
                    'Total_Victims': max(0, int(sum(pred)))
                }
            except Exception as e:
                print(f"Warning: Could not make prediction with {model_name}: {e}")
                continue
        
        # Create ensemble prediction (average of all models)
        if predictions:
            ensemble_pred = {
                'Deaths': int(np.mean([p['Deaths'] for p in predictions.values()])),
                'Injured': int(np.mean([p['Injured'] for p in predictions.values()])),
                'Affected': int(np.mean([p['Affected'] for p in predictions.values()]))
            }
            ensemble_pred['Total_Victims'] = sum([ensemble_pred['Deaths'], 
                                                ensemble_pred['Injured'], 
                                                ensemble_pred['Affected']])
        else:
            # Fallback prediction if no models are available
            base_prediction = max(10, int(area_sq_km * severity_level * 0.1))
            ensemble_pred = {
                'Deaths': base_prediction,
                'Injured': base_prediction * 3,
                'Affected': base_prediction * 10,
                'Total_Victims': base_prediction * 14
            }
        
        predictions['ensemble'] = ensemble_pred
        
        # Calculate required resources
        resources = self.calculate_required_resources(
            ensemble_pred['Deaths'],
            ensemble_pred['Injured'], 
            ensemble_pred['Affected'],
            disaster_type,
            severity_level
        )
        
        # Add resources to prediction output
        predictions['resources'] = resources
        
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'state_areas': self.state_areas
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.encoders = model_data['encoders']
        self.scalers = model_data['scalers']
        self.state_areas = model_data['state_areas']
        print(f"Model loaded from {filepath}")
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("DISASTER PREDICTION MODEL - ANALYSIS REPORT")
        print("="*60)
        
        # Dataset overview
        print(f"\nDATASET OVERVIEW:")
        print(f"Total records: {len(self.df)}")
        print(f"Date range: {self.df['Start Year'].min()} - {self.df['Start Year'].max()}")
        print(f"States covered: {self.df['State'].nunique()}")
        print(f"Disaster types: {self.df['Disaster Type'].nunique()}")
        
        # Disaster statistics
        print(f"\nDISASTER STATISTICS:")
        print(f"Total deaths: {self.df['Total Deaths'].sum():,}")
        print(f"Total injured: {self.df['No. Injured'].sum():,}")
        print(f"Total affected: {self.df['No. Affected'].sum():,}")
        
        # Most affected states
        print(f"\nMOST AFFECTED STATES (by total victims):")
        state_stats = self.df.groupby('State')['Total Victims'].sum().sort_values(ascending=False).head(10)
        for state, victims in state_stats.items():
            print(f"  {state}: {victims:,} victims")
        
        # Most common disasters
        print(f"\nMOST COMMON DISASTERS:")
        disaster_stats = self.df['Disaster Type'].value_counts().head(10)
        for disaster, count in disaster_stats.items():
            print(f"  {disaster}: {count} events")
        
        # Severity distribution
        print(f"\nSEVERITY LEVEL DISTRIBUTION:")
        severity_stats = self.df['Severity_Level'].value_counts().sort_index()
        for level, count in severity_stats.items():
            print(f"  Level {level}: {count} events")

def main():
    """Main function to demonstrate the model"""
    # Initialize model
    model = DisasterPredictionModel(r"c:\Clg projects\SIH\public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv")
    
    # Load and preprocess data
    model.load_and_preprocess_data()
    
    # Train models
    model.train_models()
    
    # Generate analysis report
    model.generate_report()
    
    # Save the model
    model.save_model("disaster_prediction_model.pkl")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Example predictions
    print("\nEXAMPLE PREDICTIONS:")
    print("-" * 30)
    
    # Example 1: Flood in West Bengal
    print("\nExample 1: Flood in West Bengal (1000 sq km, Severity 4)")
    pred1 = model.predict_victims("West Bengal", 1000, "Flood", 4)
    print(f"Ensemble Prediction: {pred1['ensemble']['Total_Victims']} total victims")
    print(f"  Deaths: {pred1['ensemble']['Deaths']}")
    print(f"  Injured: {pred1['ensemble']['Injured']}")
    print(f"  Affected: {pred1['ensemble']['Affected']}")
    
    # Example 2: Earthquake in Gujarat
    print("\nExample 2: Earthquake in Gujarat (500 sq km, Severity 5)")
    pred2 = model.predict_victims("Gujarat", 500, "Earthquake", 5)
    print(f"Ensemble Prediction: {pred2['ensemble']['Total_Victims']} total victims")
    print(f"  Deaths: {pred2['ensemble']['Deaths']}")
    print(f"  Injured: {pred2['ensemble']['Injured']}")
    print(f"  Affected: {pred2['ensemble']['Affected']}")
    
    # Example 3: Cyclone in Andhra Pradesh
    print("\nExample 3: Cyclone in Andhra Pradesh (2000 sq km, Severity 4)")
    pred3 = model.predict_victims("Andhra Pradesh", 2000, "Tropical cyclone", 4)
    print(f"Ensemble Prediction: {pred3['ensemble']['Total_Victims']} total victims")
    print(f"  Deaths: {pred3['ensemble']['Deaths']}")
    print(f"  Injured: {pred3['ensemble']['Injured']}")
    print(f"  Affected: {pred3['ensemble']['Affected']}")
    
    # Show resource requirements
    print("\n" + "="*60)
    print("RESOURCE REQUIREMENTS PREDICTIONS")
    print("="*60)
    
    resources = pred1['resources']
    print("\nFlood in West Bengal - Required Resources:")
    print(f"Medical: {resources['medical']['doctors']} doctors, {resources['medical']['nurses']} nurses")
    print(f"Shelter: {resources['shelter']['tents']} tents, {resources['shelter']['blankets']} blankets")
    print(f"Food: {resources['food_water']['food_kg']:,} kg food, {resources['food_water']['water_liters']:,} L water")
    print(f"Personnel: {resources['personnel']['search_rescue_teams']} rescue teams, {resources['personnel']['police_officers']} police")
    print(f"Logistics: {resources['logistics']['buses']} buses, {resources['logistics']['trucks']} trucks")
    
    resources = pred2['resources']
    print("\nEarthquake in Gujarat - Required Resources:")
    print(f"Medical: {resources['medical']['doctors']} doctors, {resources['medical']['nurses']} nurses")
    print(f"Shelter: {resources['shelter']['tents']} tents, {resources['shelter']['blankets']} blankets")
    print(f"Food: {resources['food_water']['food_kg']:,} kg food, {resources['food_water']['water_liters']:,} L water")
    print(f"Personnel: {resources['personnel']['search_rescue_teams']} rescue teams, {resources['personnel']['police_officers']} police")
    print(f"Logistics: {resources['logistics']['excavators']} excavators, {resources['logistics']['cranes']} cranes")

if __name__ == "__main__":
    main()