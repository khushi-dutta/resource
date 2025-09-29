# Streamlit Cloud Deployment Guide

## 🚀 Deploy to Streamlit Cloud

Follow these steps to deploy your India Disaster Prediction System to Streamlit Cloud:

### Step 1: Prepare Your Repository
1. Push your code to a GitHub repository
2. Ensure all files are committed including:
   - `streamlit_app.py` (main app file)
   - `requirements.txt` (dependencies)
   - `disaster_prediction_model.pkl` (trained model)
   - `disaster_prediction_model.py` (model class)
   - All data files and supporting modules

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path: `streamlit_app.py`
6. Click "Deploy!"

### Step 3: Configuration
The app includes:
- ✅ Proper page configuration
- ✅ Custom CSS styling
- ✅ Caching for performance
- ✅ Error handling
- ✅ Responsive design

### Step 4: Access Your App
Once deployed, you'll get a URL like: `https://your-app-name.streamlit.app`

## 📝 Deployment Checklist

- [x] `streamlit_app.py` - Main application
- [x] `requirements.txt` - Python dependencies
- [x] `disaster_prediction_model.pkl` - Trained model
- [x] `disaster_prediction_model.py` - Model class
- [x] `.streamlit/config.toml` - Streamlit configuration
- [x] Data files (CSV/Excel)
- [x] Supporting modules

## 🔧 Local Testing
Before deployment, test locally:
```bash
streamlit run streamlit_app.py
```

## 🌐 Alternative: Streamlit Community Cloud
You can also use the newer Streamlit Community Cloud at [streamlit.io/cloud](https://streamlit.io/cloud)

## ⚡ Performance Tips
- Model caching is already implemented
- Large data files are handled efficiently
- UI is optimized for web deployment

## 🔧 Troubleshooting

### Common Issues and Solutions:

1. **IndexError during model loading**
   - The app now uses safe training mode without neural networks
   - Fallback predictions available if models fail
   - Compatible with various TensorFlow versions

2. **File path issues**
   - All hardcoded paths have been removed
   - App looks for data files in the same directory

3. **Package installation errors**
   - Removed problematic packages.txt file
   - Using Python 3.9 for better compatibility

### Health Check
Run the diagnostic script locally:
```bash
python health_check.py
```

This will verify:
- All required files are present
- All dependencies can be imported
- Model can be loaded successfully

Your app is ready for deployment! 🎉