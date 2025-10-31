import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from retry import retry

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_PATH = 'marine_plastic_classifier.h5'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['organic', 'other', 'plastic']

# Global model variable
model = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Load the trained model - Demo version without TensorFlow"""
    print("Note: Running in demo mode without TensorFlow")
    return None


def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(image_path):
    """Make prediction on the uploaded image - Demo version"""
    try:
        # Demo predictions (random for demonstration)
        # In real version, this uses the actual TensorFlow model
        predictions = np.random.dirichlet(np.ones(3), size=1)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_NAMES[i]: float(predictions[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities
        }
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash('Invalid file type. Allowed types: PNG, JPG, JPEG, GIF', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_image(filepath)
        
        # Add demo notice
        flash('Demo Mode: Using random predictions. Install TensorFlow for real predictions.', 'success')
        
        return render_template('result.html', 
                             filename=filename,
                             result=result)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌊 Marine Plastic Waste Classifier - Demo Mode")
    print("="*60)
    if not os.path.exists(MODEL_PATH):
        print("⚠️  Model file not found. Using demo predictions.")
    print("📝 Note: TensorFlow not fully installed.")
    print("   This is a DEMO version with random predictions.")
    print("   To use real predictions, fix TensorFlow installation.")
    print("\n🚀 Starting Flask server...")
    print("🌐 Open your browser to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
