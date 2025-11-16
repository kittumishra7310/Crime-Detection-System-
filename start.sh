#!/bin/bash

# Crime Detection System Startup Script

echo "🚀 Starting Crime Detection System..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

echo "✅ Python and Node.js are installed"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd Backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt
cd ..

echo ""
echo "📦 Installing Node.js dependencies..."
npm install

echo ""
echo "🎯 Starting Backend Server (Port 8000)..."
cd Backend
source venv/bin/activate
python3 main.py &
BACKEND_PID=$!
cd ..

echo "⏳ Waiting for backend to start..."
sleep 5

echo ""
echo "🎯 Starting Frontend Server (Port 3000)..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ System is starting up!"
echo ""
echo "📍 Frontend: http://localhost:3000"
echo "📍 Backend API: http://localhost:8000"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "Default Login Credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user interrupt
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
