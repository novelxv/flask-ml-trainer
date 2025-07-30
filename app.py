from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'ml-trainer-super-secret-key-2025-7b8a9c1d2e3f4g5h6i7j8k9l0m1n2o3p'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global variables to store data and model
current_data = None
current_model = None
feature_columns = None
target_column = None
label_encoder = None

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data, feature_columns, target_column
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read the file based on extension
            if filename.endswith('.csv'):
                current_data = pd.read_csv(filepath)
            else:
                current_data = pd.read_excel(filepath)
            
            # Get column information
            columns = current_data.columns.tolist()
            data_info = {
                'shape': current_data.shape,
                'columns': columns,
                'head': current_data.head().to_html(classes='table table-striped'),
                'dtypes': current_data.dtypes.to_dict()
            }
            
            flash('File uploaded successfully!')
            return render_template('data_preview.html', data_info=data_info, columns=columns)
            
        except Exception as e:
            flash(f'Error reading file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload Excel (.xlsx, .xls) or CSV files.')
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    global current_data, current_model, feature_columns, target_column, label_encoder
    
    if current_data is None:
        flash('Please upload a dataset first')
        return redirect(url_for('index'))
    
    target_column = request.form.get('target_column')
    if not target_column:
        flash('Please select a target column')
        return redirect(url_for('index'))
    
    try:
        # Prepare data
        feature_columns = [col for col in current_data.columns if col != target_column]
        X = current_data[feature_columns]
        y = current_data[target_column]
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle target variable if it's categorical
        label_encoder = None
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.astype(str))
        
        # Split data
        test_size = float(request.form.get('test_size', 0.2))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        n_estimators = int(request.form.get('n_estimators', 100))
        max_depth = request.form.get('max_depth')
        max_depth = int(max_depth) if max_depth else None
        
        current_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        current_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = current_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_data = {
            'model': current_model,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'label_encoder': label_encoder
        }
        
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'trained_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Classification report
        if label_encoder:
            target_names = label_encoder.classes_
        else:
            target_names = sorted(list(set(y_test)))
        
        class_report = classification_report(y_test, y_pred, target_names=[str(name) for name in target_names], output_dict=True)
        
        training_results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'model_params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'test_size': test_size
            }
        }
        
        flash('Model trained successfully!')
        return render_template('training_results.html', results=training_results)
        
    except Exception as e:
        flash(f'Error training model: {str(e)}')
        return redirect(url_for('index'))

@app.route('/predict_page')
def predict_page():
    global current_model, feature_columns
    
    if current_model is None:
        # Try to load existing model
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'trained_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                current_model = model_data['model']
                feature_columns = model_data['feature_columns']
                global target_column, label_encoder
                target_column = model_data['target_column']
                label_encoder = model_data['label_encoder']
                flash('Model loaded successfully!')
            except Exception as e:
                flash(f'Error loading model: {str(e)}')
                return redirect(url_for('index'))
        else:
            flash('No trained model found. Please train a model first.')
            return redirect(url_for('index'))
    
    return render_template('predict.html', features=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    global current_model, feature_columns, label_encoder
    
    if current_model is None:
        return jsonify({'error': 'No model available'})
    
    try:
        # Get input values
        input_data = []
        for feature in feature_columns:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({'error': f'Please provide value for {feature}'})
            
            try:
                # Try to convert to float
                input_data.append(float(value))
            except ValueError:
                # If it's a string, we need to handle it (for categorical features)
                input_data.append(value)
        
        # Make prediction
        input_array = np.array(input_data).reshape(1, -1)
        prediction = current_model.predict(input_array)
        prediction_proba = current_model.predict_proba(input_array)
        
        # Get class names
        if label_encoder:
            predicted_class = label_encoder.inverse_transform(prediction)[0]
            classes = label_encoder.classes_
        else:
            predicted_class = prediction[0]
            classes = current_model.classes_
        
        # Get probability for each class
        class_probabilities = {}
        for i, class_name in enumerate(classes):
            class_probabilities[str(class_name)] = float(prediction_proba[0][i])
        
        # Get confidence (max probability)
        confidence = float(max(prediction_proba[0]))
        
        result = {
            'prediction': str(predicted_class),
            'confidence': confidence,
            'probability_percentage': confidence * 100,
            'class_probabilities': class_probabilities,
            'input_values': dict(zip(feature_columns, input_data))
        }
        
        return render_template('prediction_result.html', result=result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/load_sample_data')
def load_sample_data():
    """Load sample Iris dataset for demonstration"""
    global current_data
    
    try:
        # Create sample Iris dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['species'] = iris.target_names[iris.target]
        
        # Save to uploads folder
        sample_path = os.path.join(app.config['UPLOAD_FOLDER'], 'iris_sample.csv')
        iris_df.to_csv(sample_path, index=False)
        
        current_data = iris_df
        
        columns = current_data.columns.tolist()
        data_info = {
            'shape': current_data.shape,
            'columns': columns,
            'head': current_data.head().to_html(classes='table table-striped'),
            'dtypes': current_data.dtypes.to_dict()
        }
        
        flash('Sample Iris dataset loaded successfully!')
        return render_template('data_preview.html', data_info=data_info, columns=columns)
        
    except Exception as e:
        flash(f'Error loading sample data: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
