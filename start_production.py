# start_production_fixed.py
import subprocess
import sys
import time
import requests
import socket

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Kill process using specified port (Windows)"""
    try:
        import os
        # Find PID using port
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, 
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                pid = parts[-1]
                print(f"🛑 Killing process {pid} using port {port}")
                subprocess.run(['taskkill', '/PID', pid, '/F'], check=True)
                time.sleep(2)  # Wait for process to terminate
                return True
        return False
    except Exception as e:
        print(f"⚠️ Could not kill process on port {port}: {e}")
        return False

def start_production_servers():
    """Start production servers with port conflict handling"""
    print("🚀 STARTING PRODUCTION SERVERS")
    print("=" * 50)
    
    # Check and clear port 8000
    if is_port_in_use(8000):
        print("🔄 Port 8000 is in use, attempting to clear...")
        if kill_process_on_port(8000):
            print("✅ Port 8000 cleared")
        else:
            print("❌ Could not clear port 8000, trying alternate port...")
            # You could use port 8001 as fallback
            return
    
    # Start FastAPI server (production mode)
    print("🔧 Starting FastAPI Server (Production Mode)...")
    api_process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn', 'run_phase8_optimized:app',
        '--host', '0.0.0.0', '--port', '8001'
    ])
    
    # Wait for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(5)
    
    # Verify API is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            print("✅ FastAPI Server is running")
        else:
            print("❌ FastAPI Server failed to start properly")
    except:
        print("❌ FastAPI Server is not responding")
    
    # Start Streamlit dashboard
    print("📊 Starting Streamlit Dashboard...")
    dashboard_process = subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run', 'run_phase9.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0'
    ])
    
    print("\n✅ PRODUCTION SERVERS STARTED!")
    print("🌐 FastAPI: http://localhost:8001")
    print("📊 Dashboard: http://localhost:8501")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n🎯 TROUBLESHOOTING: If predictions show 'NO', try:")
    print("   • Increase 'Average Rating' to 4.3+")
    print("   • Increase 'Total Rating Count' to 2000+")
    print("   • Increase 'Average Price' to 600+")
    print("\n🛑 Press Ctrl+C to stop servers")
    
    try:
        # Keep servers running
        api_process.wait()
        dashboard_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        api_process.terminate()
        dashboard_process.terminate()
        api_process.wait()
        dashboard_process.wait()
        print("✅ Servers stopped")

if __name__ == "__main__":
    start_production_servers()
