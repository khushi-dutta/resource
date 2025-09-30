# India Disaster Prediction System 🚨

A comprehensive machine learning system for predicting natural disaster casualties in India based on location, area, disaster type, and severity level.

## 🎯 Features

- **Multi-Model Prediction**: Uses Random Forest and XGBoost for robust predictions (Neural Networks in local environments)
- **Location-Based Analysis**: Predicts casualties for specific Indian states
- **Severity Assessment**: 5-level severity scale for disaster impact
- **Interactive Web Interface**: User-friendly Streamlit application deployed on cloud
- **Command Line Interface**: Quick predictions via CLI
- **Historical Analysis**: Comprehensive analysis of past disasters
- **Emergency Recommendations**: Actionable response suggestions
- **Cloud Deployment Ready**: Optimized for Streamlit Cloud with automatic compatibility detection

## 📊 Prediction Outputs

The system predicts:
- **Deaths**: Number of fatalities
- **Injured**: Number of injured people  
- **Affected**: Number of people affected (displaced, economic loss, etc.)
- **Total Victims**: Combined count of all casualties
- **Risk Level**: Automated risk assessment (Low/Moderate/High/Critical)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/khushi-dutta/resource.git
   cd resource
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run health check (optional):
   ```bash
   python health_check.py
   ```

## 🚀 Usage

### Option 1: Web Interface (Recommended)
Launch the interactive web application:
```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### 🌐 Live Demo
The application is deployed on Streamlit Cloud and available at: [India Disaster Prediction System](https://resource-hcncmfy4m5ekvrtrodzfxh.streamlit.app/)

> **Note**: First load may take 30-60 seconds as the model trains fresh for optimal cloud compatibility.

#### Web Interface Features:
- **Parameter Selection**: Choose state, disaster type, area, and severity
- **Real-time Predictions**: Instant results with visualization
- **Model Comparison**: See predictions from different ML models
- **Historical Analysis**: Explore past disaster data
- **Risk Assessment**: Automated risk level determination
- **Emergency Recommendations**: Actionable response guidelines

### Option 2: Command Line Interface
For quick predictions via command line:

```bash
# Basic prediction
python cli_predictor.py --state "West Bengal" --disaster "Flood" --area 1000 --severity 4

# Detailed output
python cli_predictor.py --state "Gujarat" --disaster "Earthquake" --area 500 --severity 5 --verbose

# Help and examples
python cli_predictor.py --help
```

#### CLI Parameters:
- `--state`: Indian state name (required)
- `--disaster`: Disaster type (required)
- `--area`: Affected area in sq km (required)
- `--severity`: Severity level 1-5 (required)
- `--verbose`: Show detailed output
- `--model-path`: Custom model file path
- `--data-path`: Custom data file path

### Option 3: Python API
Use the model programmatically:

```python
from disaster_prediction_model import DisasterPredictionModel

# Initialize model
model = DisasterPredictionModel("path/to/your/data.csv")

# Load and train (first time only)
model.load_and_preprocess_data()
model.train_models()

# Make predictions
predictions = model.predict_victims(
    location="West Bengal",
    area_sq_km=1000,
    disaster_type="Flood", 
    severity_level=4
)

print(f"Predicted victims: {predictions['ensemble']['Total_Victims']}")
```

## 📈 Example Predictions

### Flood in West Bengal
```
State: West Bengal
Disaster: Flood
Area: 1,000 sq km
Severity: 4 (High)

Results:
- Deaths: 45
- Injured: 1,200
- Affected: 25,000
- Total Victims: 26,245
- Risk Level: HIGH RISK
```

### Earthquake in Gujarat
```
State: Gujarat
Disaster: Earthquake
Area: 500 sq km  
Severity: 5 (Very High)

Results:
- Deaths: 2,500
- Injured: 8,500
- Affected: 45,000
- Total Victims: 56,000
- Risk Level: CRITICAL RISK
```

### Cyclone in Andhra Pradesh
```
State: Andhra Pradesh
Disaster: Tropical cyclone
Area: 2,000 sq km
Severity: 4 (High)

Results:
- Deaths: 125
- Injured: 3,500
- Affected: 75,000
- Total Victims: 78,625
- Risk Level: CRITICAL RISK
```

## 🎛️ Model Parameters

### Supported Indian States
All 28 states and 8 union territories:
- Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh
- Delhi, Gujarat, Haryana, Himachal Pradesh, Jharkhand
- Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur
- Meghalaya, Mizoram, Nagaland, Orissa, Punjab, Rajasthan
- Sikkim, Tamil Nadu, Telangana, Tripura, Uttar Pradesh
- Uttarakhand, West Bengal

### Disaster Types
- **Hydrological**: Flood, Landslide, Mass movement
- **Geophysical**: Earthquake, Volcanic activity
- **Meteorological**: Storm, Cyclone, Hail, Lightning
- **Climatological**: Drought, Wildfire, Temperature extremes
- **Biological**: Epidemic, Disease outbreaks

### Severity Levels
1. **Very Low**: Minor impact, minimal casualties
2. **Low**: Limited damage and casualties  
3. **Moderate**: Significant but manageable impact
4. **High**: Serious damage, many casualties
5. **Very High**: Catastrophic damage and casualties

## 🧠 Machine Learning Models

The system uses three complementary ML models:

### 1. Random Forest
- **Purpose**: Robust baseline predictions
- **Strengths**: Handles non-linear relationships, feature importance
- **Parameters**: 200 trees, max depth 15

### 2. XGBoost
- **Purpose**: Gradient boosting for improved accuracy
- **Strengths**: Handles missing data, prevents overfitting
- **Parameters**: 200 estimators, learning rate 0.1

### 3. Neural Network (Local Environment Only)
- **Purpose**: Deep learning for complex patterns
- **Architecture**: 4 layers (128→64→32→3 neurons)
- **Features**: Dropout, batch normalization, early stopping
- **Note**: Automatically disabled on cloud deployments for compatibility

### Ensemble Prediction
The final prediction combines all three models using averaging for robust results.

## 📊 Data Source

The model is trained on the **EM-DAT (Emergency Events Database)** dataset containing:
- **Time Period**: 1990-2024
- **Geographic Scope**: India only
- **Disaster Types**: All natural disasters
- **Records**: 400+ disaster events
- **Variables**: 40+ features including casualties, damage, location, magnitude

## � Deployment

### Streamlit Cloud Deployment
The app is optimized for Streamlit Cloud deployment:

1. **Fork the repository** on GitHub
2. **Connect to Streamlit Cloud** at [share.streamlit.io](https://share.streamlit.io)
3. **Deploy** by selecting your repository and `streamlit_app.py`
4. **Automatic optimization** detects cloud environment and adjusts accordingly

### Local Development
```bash
# Quick start
streamlit run streamlit_app.py

# With custom port
streamlit run streamlit_app.py --server.port 8502

# Health check first
python health_check.py
streamlit run streamlit_app.py
```

### Environment Detection
The app automatically detects deployment environment and:
- **Local**: Uses all models including neural networks
- **Cloud**: Skips neural networks for compatibility
- **Fallback**: Provides basic predictions if model training fails

## 🔧 Troubleshooting

### Common Issues

#### "DisasterPredictionModel has no attribute 'train_models_safe'"
- **Solution**: Update to latest version from GitHub
- **Cause**: Old cached version of code

#### "IndexError during model loading"
- **Solution**: Fresh training will resolve TensorFlow compatibility
- **Cause**: Neural network version conflicts

#### App takes long to load
- **Expected**: First load trains models fresh (30-60 seconds)
- **Optimization**: Models are cached after first successful training

### Health Check
Run diagnostics before deployment:
```bash
python health_check.py
```

This checks:
- ✅ Required files present
- ✅ Dependencies installed
- ✅ Model can be loaded
- ✅ Training works correctly

## �🔧 Customization

### Adding New States/Regions
Update the `state_areas` dictionary in `disaster_prediction_model.py`:
```python
self.state_areas = {
    'Your State': area_in_sq_km,
    # ... existing states
}
```

### Adding New Disaster Types
Modify the `_categorize_disaster()` method to include new disaster categories.

### Training with New Data
Replace the CSV file path and retrain:
```python
model = DisasterPredictionModel("path/to/new/data.csv")
model.load_and_preprocess_data()
model.train_models()
model.save_model("new_model.pkl")
```

## 📋 Requirements

### Python Packages
- pandas >= 1.5.0 (Data manipulation)
- numpy >= 1.21.0 (Numerical computations)
- scikit-learn >= 1.3.0 (Machine learning)
- xgboost >= 1.7.0 (Gradient boosting)
- tensorflow >= 2.13.0 (Neural networks)
- streamlit >= 1.28.0 (Web interface)
- plotly >= 5.17.0 (Interactive visualizations)
- matplotlib >= 3.6.0 (Static plots)
- seaborn >= 0.12.0 (Statistical visualizations)

### System Requirements
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space for model and data
- **CPU**: Multi-core recommended for faster training

## 🚨 Emergency Response Integration

The system provides actionable recommendations for:

### Immediate Response
- Search and rescue deployment
- Medical triage setup
- Evacuation procedures
- Emergency shelter establishment

### Resource Coordination
- Personnel allocation
- Equipment deployment  
- Supply distribution
- Communication protocols

### Risk-Based Actions
- **Low Risk**: Standard monitoring
- **Moderate Risk**: Enhanced preparedness
- **High Risk**: Partial mobilization
- **Critical Risk**: Full emergency response

## 📈 Model Performance

### Accuracy Metrics
- **Mean Absolute Error (MAE)**: Measures average prediction error
- **R² Score**: Explains variance in predictions (0-1, higher is better)
- **Cross-Validation**: 5-fold validation for robust evaluation

### Feature Importance
Top predictive factors:
1. Disaster severity level
2. Affected area size
3. Geographic location (state)
4. Disaster type category
5. Seasonal factors (monsoon)

## 🔄 Model Updates

### Automatic Retraining
The model automatically retrains if:
- New data is provided
- Model file is missing
- Significant prediction errors detected

### Manual Updates
```python
# Retrain with new data
model.load_and_preprocess_data()
model.train_models()
model.save_model("updated_model.pkl")
```

## 🎯 Use Cases

### Disaster Management Agencies
- **Pre-disaster Planning**: Resource allocation based on risk
- **Real-time Response**: Rapid casualty estimation
- **Post-disaster Assessment**: Damage validation

### Government Bodies
- **Policy Making**: Evidence-based disaster policies
- **Budget Planning**: Resource allocation for emergency response
- **Inter-state Coordination**: Mutual aid agreements

### Research Organizations
- **Risk Assessment**: Academic disaster risk studies
- **Climate Change Impact**: Future disaster scenario modeling
- **Methodology Development**: ML techniques for disaster prediction

### NGOs and Relief Organizations
- **Resource Deployment**: Optimal aid distribution
- **Volunteer Coordination**: Personnel allocation
- **Fundraising**: Evidence-based aid campaigns

## 🛡️ Limitations and Disclaimers

### Model Limitations
- **Historical Bias**: Based on past events, may not capture future changes
- **Data Quality**: Predictions depend on input data accuracy
- **Regional Variations**: Some areas may have limited historical data
- **Disaster Complexity**: Cannot capture all variables affecting casualties

### Usage Disclaimers
- **Emergency Use**: This is a prediction tool, not a replacement for expert judgment
- **Decision Support**: Should complement, not replace, emergency management expertise
- **Regular Updates**: Model should be retrained periodically with new data
- **Validation Required**: Predictions should be validated against local knowledge

## 🤝 Contributing

### Bug Reports
- Report issues via GitHub or email
- Include error messages and input parameters
- Specify operating system and Python version

### Feature Requests
- Suggest new disaster types or regions
- Propose additional prediction outputs
- Request integration capabilities

### Data Contributions
- Share additional disaster datasets
- Provide local knowledge for validation
- Suggest data quality improvements

## 📞 Support

### Technical Support
- **Documentation**: Comprehensive README and code comments
- **Examples**: Multiple usage examples provided
- **Error Handling**: Detailed error messages and troubleshooting

### Contact Information
- **Issues**: Report via GitHub issues
- **Questions**: Technical questions via email
- **Updates**: Follow project for latest releases

## 📄 License

This project is developed for the Smart India Hackathon (SIH) and is intended for educational and research purposes. Please ensure compliance with local regulations when using for operational disaster management.

## 🙏 Acknowledgments

- **EM-DAT Database**: Centre for Research on the Epidemiology of Disasters (CRED)
- **Scikit-learn**: Machine learning library
- **Streamlit**: Web application framework
- **TensorFlow**: Deep learning framework
- **Smart India Hackathon**: Platform for innovative solutions

---

**Built with ❤️ for India's disaster resilience**#   r e s o u r c e 
 
 