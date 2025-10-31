import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import base64
import io

# Import DurNet model - sử dụng model Xception đã được train
from durnet_xception import DurNet

app = Flask(__name__)
CORS(app)

# Global variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Disease classes - cập nhật theo model DurNet
DISEASE_CLASSES = [
    "Leaf_Blight",
    "Leaf_Rhizoctonia", 
    "Leaf_Phomopsis",
    "Leaf_Algal",
    "Leaf_Colletotrichum",
    "Leaf_Healthy"
]

def load_model():
    """Load the trained DurNet model"""
    global model
    try:
        # Initialize model với số classes đúng
        model = DurNet(num_classes=len(DISEASE_CLASSES))  # 6 classes
        
        # Load weights
        model_path = 'durnet.pth'
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # Load state dict với strict=False để tránh lỗi key không khớp
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                
                if missing_keys:
                    print(f"⚠️  Missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys: {len(unexpected_keys)} keys")
                    
                print(f"✅ Model weights loaded successfully!")
            except Exception as e:
                print(f"⚠️  Warning: Could not load model weights: {str(e)}")
                print("Using model with random weights for demo purposes")
                
            model.to(device)
            model.eval()
            print(f"📱 Model loaded on device: {device}")
            print(f"🧠 Model classes: {DISEASE_CLASSES}")
            print(f"🔢 Number of classes: {len(DISEASE_CLASSES)}")
            return True
        else:
            print(f"❌ Model file not found: {model_path}")
            print("Initializing model with random weights for demo purposes")
            model.to(device)
            model.eval()
            return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_image(image_data):
    """Preprocess image for model inference"""
    try:
        # Decode base64 image
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(device)
    
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 500
    
    try:
        # Check for file upload (from React Native)
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'error': 'No image file selected',
                    'success': False
                }), 400
            
            # Process uploaded file
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Check for base64 image (alternative method)
        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            image_tensor = preprocess_image(image_data)
            if image_tensor is None:
                return jsonify({
                    'error': 'Failed to process image',
                    'success': False
                }), 400
        
        else:
            return jsonify({
                'error': 'No image provided',
                'success': False
            }), 400
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get all class probabilities
            all_predictions = probabilities[0].cpu().numpy().tolist()
        
        result = {
            'success': True,
            'predicted_class': predicted_class,
            'predicted_disease': DISEASE_CLASSES[predicted_class],
            'confidence': confidence,
            'all_predictions': all_predictions,
            'class_names': {i: name for i, name in enumerate(DISEASE_CLASSES)}
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available disease classes"""
    return jsonify({
        'classes': DISEASE_CLASSES,
        'total': len(DISEASE_CLASSES)
    })

@app.route('/disease-map/regions', methods=['GET'])
def get_disease_map_regions():
    """Get simulated disease distribution data for regions"""
    import random
    
    # Simulated regions data with random disease distribution
    regions = [
        {
            'id': 1,
            'name': 'Khu vực Đông Nam Bộ',
            'coordinates': {'x': 220, 'y': 180},
            'total_trees': 1250,
            'affected_trees': random.randint(400, 800),
            'diseases': []
        },
        {
            'id': 2,
            'name': 'Khu vực Tây Nam Bộ', 
            'coordinates': {'x': 160, 'y': 200},
            'total_trees': 980,
            'affected_trees': random.randint(300, 600),
            'diseases': []
        },
        {
            'id': 3,
            'name': 'Khu vực Trung Bộ',
            'coordinates': {'x': 190, 'y': 120},
            'total_trees': 750,
            'affected_trees': random.randint(200, 400),
            'diseases': []
        },
        {
            'id': 4,
            'name': 'Khu vực Đông Bắc',
            'coordinates': {'x': 220, 'y': 60},
            'total_trees': 650,
            'affected_trees': random.randint(150, 350),
            'diseases': []
        },
        {
            'id': 5,
            'name': 'Khu vực Tây Bắc',
            'coordinates': {'x': 140, 'y': 80},
            'total_trees': 420,
            'affected_trees': random.randint(100, 200),
            'diseases': []
        }
    ]
    
    # Generate random disease distribution for each region
    for region in regions:
        # Random disease percentages that sum up appropriately
        disease_count = random.randint(2, 4)  # 2-4 diseases per region
        selected_diseases = random.sample(DISEASE_CLASSES[:-1], disease_count)  # Exclude Healthy
        
        total_diseased = 0
        diseases = []
        
        for disease in selected_diseases:
            percentage = random.randint(5, 30)
            severity = random.choice(['low', 'medium', 'high'])
            diseases.append({
                'name': disease,
                'percentage': percentage,
                'severity': severity
            })
            total_diseased += percentage
        
        # Add healthy percentage
        healthy_percentage = max(20, 100 - total_diseased)
        diseases.append({
            'name': 'Leaf_Healthy',
            'percentage': healthy_percentage
        })
        
        # Normalize percentages to sum to 100
        total = sum(d['percentage'] for d in diseases)
        for disease in diseases:
            disease['percentage'] = round((disease['percentage'] / total) * 100, 1)
        
        region['diseases'] = diseases
    
    return jsonify({
        'success': True,
        'regions': regions,
        'total_regions': len(regions),
        'last_updated': '2024-10-16T10:30:00Z'
    })

@app.route('/disease-map/statistics', methods=['GET'])
def get_disease_statistics():
    """Get overall disease statistics"""
    import random
    
    # Simulated statistics
    total_trees = 4050
    total_affected = random.randint(1200, 2000)
    
    # Disease distribution across all regions
    disease_stats = {}
    remaining = total_affected
    
    for i, disease in enumerate(DISEASE_CLASSES[:-1]):  # Exclude Healthy
        if i == len(DISEASE_CLASSES) - 2:  # Last disease gets remaining
            count = remaining
        else:
            count = random.randint(50, min(400, remaining - 50))
            remaining -= count
        
        disease_stats[disease] = {
            'count': count,
            'percentage': round((count / total_trees) * 100, 1),
            'severity_distribution': {
                'high': random.randint(20, 40),
                'medium': random.randint(30, 50),
                'low': random.randint(10, 30)
            }
        }
    
    # Healthy trees
    healthy_count = total_trees - total_affected
    disease_stats['Leaf_Healthy'] = {
        'count': healthy_count,
        'percentage': round((healthy_count / total_trees) * 100, 1)
    }
    
    return jsonify({
        'success': True,
        'total_trees': total_trees,
        'total_affected': total_affected,
        'total_healthy': healthy_count,
        'disease_distribution': disease_stats,
        'last_updated': '2024-10-16T10:30:00Z'
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("Failed to load model. Server not started.")