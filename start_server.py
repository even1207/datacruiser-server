#!/usr/bin/env python3
"""
Startup script for DataCruiser RAG Server
"""

import sys
import os
import subprocess
import time
import signal
import requests
from threading import Thread

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_server_health(port=5080, max_attempts=30):
    """Check if server is healthy"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}/", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

def start_server():
    """Start the server"""
    print("🚀 Starting DataCruiser RAG Server...")
    print("=" * 50)
    
    # Check if data exists
    data_file = "data/data.json"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure data/data.json exists")
        return False
    
    print(f"✅ Data file found: {data_file}")
    
    # Start server in background
    print("🔄 Starting server process...")
    process = subprocess.Popen([
        sys.executable, "run.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for server to start
    print("⏳ Waiting for server to initialize (this may take a few minutes for first run)...")
    
    if check_server_health():
        print("✅ Server started successfully!")
        print("\n🔗 Server is running at: http://localhost:5080")
        print("\n📋 Available endpoints:")
        print("  - GET  /           : Health check")
        print("  - POST /api/ask    : Ask questions about footfall data")
        print("  - GET  /data/info : Data information")
        print("  - POST /cache/clear: Clear cache")
        print("  - GET  /cache/status: Cache status")
        
        print("\n🧪 Testing RAG functionality...")
        try:
            # Test the RAG system
            test_data = {
                "question": "What are the busiest locations?",
                "top_k": 3
            }
            
            response = requests.post(
                "http://localhost:5080/api/ask",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ RAG system working!")
                print(f"   Question: {result.get('question')}")
                print(f"   Answer: {result.get('answer', '')[:100]}...")
                print(f"   Records found: {len(result.get('similar_records', []))}")
            else:
                print(f"⚠️ RAG test failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ RAG test error: {e}")
        
        print("\n🎉 Server is ready! You can now:")
        print("   1. Use the API endpoints directly")
        print("   2. Run: python3 test_rag.py")
        print("   3. Press Ctrl+C to stop the server")
        
        try:
            # Keep server running
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")
            
    else:
        print("❌ Server failed to start properly")
        stdout, stderr = process.communicate()
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        return False
    
    return True

if __name__ == "__main__":
    start_server()
