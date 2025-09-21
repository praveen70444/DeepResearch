#!/usr/bin/env python3
"""
Startup script for Deep Researcher application
Handles both backend and frontend startup for deployment
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting Deep Researcher Backend...")
    try:
        subprocess.run([sys.executable, "api_server_final.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend failed to start: {e}")
        sys.exit(1)

def start_frontend():
    """Start the React frontend"""
    print("🎨 Starting Deep Researcher Frontend...")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return
    
    try:
        # Install dependencies if needed
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        
        # Start the frontend
        subprocess.run(["npm", "start"], cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend failed to start: {e}")

def main():
    """Main startup function"""
    print("🌟 Deep Researcher - Starting Application...")
    
    # Check if we're in the right directory
    if not Path("api_server_final.py").exists():
        print("❌ api_server_final.py not found. Please run from project root.")
        sys.exit(1)
    
    # For deployment, typically you'd start just the backend
    # Frontend would be built and served statically
    if os.getenv("DEPLOYMENT_MODE") == "production":
        print("🏭 Production mode: Starting backend only")
        start_backend()
    else:
        print("🔧 Development mode: Starting both backend and frontend")
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=start_backend, daemon=True)
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Start frontend
        start_frontend()

if __name__ == "__main__":
    main()
