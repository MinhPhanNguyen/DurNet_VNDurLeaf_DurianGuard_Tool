#!/usr/bin/env python3
"""
Test script cho Durian Disease Detection API
Kiểm tra các endpoint và model prediction
"""

import requests
import json
import base64
from PIL import Image
import io
import numpy as np

# Cấu hình
API_BASE_URL = 'http://localhost:5001'
TEST_IMAGE_SIZE = (224, 224)

def create_test_image():
    """Tạo ảnh test đơn giản"""
    # Tạo ảnh RGB ngẫu nhiên
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Chuyển thành bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

def test_health_endpoint():
    """Test health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f'{API_BASE_URL}/health')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check OK")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_classes_endpoint():
    """Test classes endpoint"""
    print("\n🔍 Testing classes endpoint...")
    try:
        response = requests.get(f'{API_BASE_URL}/classes')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Classes endpoint OK")
            print(f"   Total classes: {data.get('total')}")
            print(f"   Classes: {data.get('classes')}")
            return True
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Classes endpoint error: {e}")
        return False

def test_predict_endpoint():
    """Test prediction endpoint"""
    print("\n🔍 Testing prediction endpoint...")
    try:
        # Tạo ảnh test
        test_image = create_test_image()
        
        # Gửi request
        files = {'image': ('test.jpg', test_image, 'image/jpeg')}
        response = requests.post(f'{API_BASE_URL}/predict', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction OK")
            print(f"   Success: {data.get('success')}")
            print(f"   Predicted class: {data.get('predicted_class')}")
            print(f"   Predicted disease: {data.get('predicted_disease')}")
            print(f"   Confidence: {data.get('confidence', 0):.3f}")
            print(f"   All predictions: {[f'{p:.3f}' for p in data.get('all_predictions', [])]}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_base64_prediction():
    """Test prediction với base64 image"""
    print("\n🔍 Testing base64 prediction...")
    try:
        # Tạo ảnh test và chuyển thành base64
        test_image = create_test_image()
        base64_image = base64.b64encode(test_image).decode('utf-8')
        
        # Gửi request
        data = {'image': base64_image}
        response = requests.post(f'{API_BASE_URL}/predict', json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Base64 prediction OK")
            print(f"   Predicted disease: {result.get('predicted_disease')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            return True
        else:
            print(f"❌ Base64 prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Base64 prediction error: {e}")
        return False

def main():
    """Chạy tất cả tests"""
    print("🌿 Durian Disease Detection API Test")
    print("=" * 50)
    
    # Danh sách tests
    tests = [
        test_health_endpoint,
        test_classes_endpoint,
        test_predict_endpoint,
        test_base64_prediction
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # Tổng kết
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
        
    return passed == total

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)