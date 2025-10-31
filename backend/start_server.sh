#!/bin/bash

# Script để khởi động backend và hiển thị IP
echo "🌿 Durian Disease Detection - Backend Startup"
echo "============================================="

# Tìm IP của máy
echo "📱 Tìm IP của máy..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    IP=$(ipconfig getifaddr en0)
    if [ -z "$IP" ]; then
        IP=$(ipconfig getifaddr en1)
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    IP=$(hostname -I | awk '{print $1}')
else
    # Windows (Git Bash)
    IP=$(ipconfig | grep "IPv4" | head -1 | awk '{print $NF}')
fi

if [ -z "$IP" ]; then
    echo "❌ Không thể tìm thấy IP. Vui lòng tìm IP thủ công."
    IP="YOUR_IP_HERE"
else
    echo "✅ IP của máy: $IP"
fi

echo ""
echo "📝 Cập nhật IP trong React Native app:"
echo "   Mở file: DurianDetectorApp/src/services/ApiService.js"
echo "   Thay đổi: const API_BASE_URL = 'http://$IP:5000';"
echo ""

# Kiểm tra Python và dependencies
echo "🐍 Kiểm tra Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 không được tìm thấy. Vui lòng cài đặt Python3."
    exit 1
fi

echo "📦 Kiểm tra dependencies..."
pip3 list | grep -E "(torch|flask|Pillow)" > /dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Một số dependencies có thể thiếu. Cài đặt dependencies..."
    pip3 install -r requirements.txt
fi

echo "🚀 Khởi động Flask server..."
echo "   Server sẽ chạy tại: http://$IP:5000"
echo "   Nhấn Ctrl+C để dừng server"
echo ""

# Khởi động Flask server
cd "$(dirname "$0")"
python3 app.py