# Flask ML Trainer 🤖

A beautiful web-based machine learning application built with Flask that allows you to upload datasets, train models, and make predictions with an intuitive interface. Can be converted to a standalone executable using PyInstaller.

## 🌟 Features

- **Easy Dataset Upload**: Support for Excel (.xlsx, .xls) and CSV files
- **Interactive Data Preview**: View your data structure and statistics before training
- **Automated ML Training**: Train Random Forest models with configurable parameters
- **Real-time Predictions**: Make predictions with confidence scores and probability distributions
- **Beautiful Interface**: Modern Bootstrap UI with responsive design
- **Sample Dataset**: Built-in Iris dataset for testing and demonstration
- **Standalone Executable**: Convert to .exe file using PyInstaller
- **Model Persistence**: Automatically save and load trained models

## 🚀 Quick Start

### Method 1: Using Batch Files (Recommended)

1. **Clone or download this repository**

2. **Build the application:**
   ```bash
   double-click build.bat
   ```
   This will:
   - Create a virtual environment
   - Install all requirements
   - Build the executable with PyInstaller

3. **Run the application:**
   - For development: double-click `run.bat`
   - For production: run `dist\FlaskMLTrainer.exe`

### Method 2: Manual Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser to:**
   ```
   http://localhost:5000
   ```

## 📊 How to Use

### 1. Upload Your Dataset
- Click "Upload Dataset" on the home page
- Select your Excel or CSV file
- Supported formats: .xlsx, .xls, .csv
- Or use the sample Iris dataset for testing

### 2. Preview and Configure
- Review your data structure
- Select the target column (what you want to predict)
- Configure training parameters:
  - Test size (portion of data for testing)
  - Number of trees (more = better accuracy, slower training)
  - Maximum depth (optional, controls overfitting)

### 3. Train Your Model
- Click "Train Model" to start training
- View training results including:
  - Overall accuracy
  - Detailed classification report
  - Feature importance
  - Model configuration

### 4. Make Predictions
- Navigate to the "Predict" page
- Enter feature values
- Get instant predictions with:
  - Predicted class
  - Confidence percentage
  - Probability distribution for all classes
  - Confidence interpretation

## 🔧 Building Executable

The application comes with a pre-configured PyInstaller spec file for creating a standalone executable:

```bash
# Automatic build (recommended)
build.bat

# Manual build
pyinstaller app.spec
```

The executable will be created in the `dist` folder as `FlaskMLTrainer.exe`.

## 📁 Project Structure

```
flask-ml-trainer/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── app.spec              # PyInstaller configuration
├── build.bat             # Windows build script
├── run.bat               # Windows run script
├── templates/            # HTML templates
│   ├── base.html         # Base template with Bootstrap
│   ├── index.html        # Home page
│   ├── data_preview.html # Dataset preview
│   ├── training_results.html # Training results
│   ├── predict.html      # Prediction input
│   └── prediction_result.html # Prediction output
├── static/               # Static files
│   └── style.css         # Custom CSS
├── uploads/              # Uploaded datasets (auto-created)
└── models/               # Saved models (auto-created)
```

## 🎯 Supported Datasets

The application works best with:
- **Classification datasets** (predicting categories)
- **Numerical and categorical features**
- **Small to medium datasets** (up to 16MB)
- **Common formats**: CSV, Excel

### Example Datasets You Can Try:
- Iris flower classification
- Wine quality prediction
- Customer segmentation
- Medical diagnosis
- Sales prediction

## ⚙️ Technical Details

### Technologies Used:
- **Backend**: Flask, scikit-learn, pandas, numpy
- **Frontend**: Bootstrap 5, Font Awesome, HTML5/CSS3
- **ML Algorithm**: Random Forest Classifier
- **File Processing**: openpyxl, xlrd for Excel files
- **Deployment**: PyInstaller for executable creation

### Features:
- Automatic data preprocessing
- Categorical variable encoding
- Train/test split
- Model persistence with pickle
- Cross-validation metrics
- Responsive web design

## 🚨 Troubleshooting

### Common Issues:

1. **"No module named 'sklearn'"**
   - Run `pip install -r requirements.txt`

2. **"Permission denied" when building executable**
   - Run as administrator or check antivirus settings

3. **"File not found" errors**
   - Ensure all files are in the correct directory structure

4. **Poor model performance**
   - Check if your dataset has enough samples
   - Ensure target column is correctly selected
   - Try adjusting training parameters

### System Requirements:
- Windows 10/11 (for .exe build)
- Python 3.7+
- 4GB RAM minimum
- 500MB disk space

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Submitting pull requests
- Improving documentation

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the error messages in the terminal
3. Ensure all requirements are installed correctly

---

**Made with ❤️ using Flask and scikit-learn**